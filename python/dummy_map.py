from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import torch
import time
import numpy as np
import math
import yaml
import shutil
import sys
import random
import re
import os
import transformer
import pickle
from pathlib import Path
import json
import argparse
from transformers import get_linear_schedule_with_warmup
from torchtext.vocab import Vocab, vocab
from collections import OrderedDict, Counter
from dataset import Seq2SeqDataset
from torch.utils.data import DataLoader
from utils import cer, accuracy
import string
import wandb
import pickle
import re
from tqdm import tqdm
from khmernltk import word_tokenize
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(
    description="Train the model on dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "-m", "--model", help="Model architecture: currently only transformer", default="transformer")
parser.add_argument("-i", "--data-input", type=Path,
                    help="Specify file with input sequences", required=True)

parser.add_argument("-t", "--data-target", type=Path,
                    help="Specify file with target sequences", required=True)
## TODO: Test all arguments

args = parser.parse_args()

f = open(args.data_input)
data_input = f.read().split('\n')
if data_input[-1] == '': data_input = data_input[:-1]
f.close()

f = open(args.data_target)
data_target = f.read().split('\n')
if data_target[-1] == '': data_target = data_target[:-1]
f.close()

# ------------- CONSONANTS ----------------

# -----------------------------------------
# ------------- NORMAL FORM ---------------
# -----------------------------------------
# 1 corresponds to a-series
# 0 corresponds to o-series
consonants_ind = {
    'ក': ('k', 1),'ខ': ('kh', 1),'គ': ('k', 0),    
    'ឃ': ('kh', 0),'ង': ('ng', 0),'ច': ('ch', 1),   
    'ឆ': ('ch', 1),'ជ': ('ch', 0),'ឈ': ('ch', 0),
    'ញ': ('nh', 0),'ដ': ('d', 1),'ឋ': ('th', 1),   
    'ឌ': ('d', 0),'ឍ': ('th', 0),'ណ': ('n', 1),
    'ត': ('t', 1),'ថ': ('th', 1),'ទ': ('t', 0),
    'ធ': ('th', 0),'ន': ('n', 0),
    'ប': ('b', 1), # if used with a subscript char the transliteration of the char will be p
    'ផ': ('ph', 1),'ព': ('p', 0),'ភ': ('ph', 0),
    'ម': ('m', 0),'យ': ('y', 0),'រ': ('r', 0),
    'ល': ('l', 0),'វ': ('v', 0),'ស': ('s', 1),
    'ហ': ('h', 1),'ឡ': ('l', 1),
}
# -----------------------------------------
# ----------- SUBSCRIPT FORM --------------
# -----------------------------------------
consonants_sub = {
    '្ក': ('k', 1),'្ខ': ('kh', 1),'្គ': ('k', 0),    
    '្ឃ': ('kh', 0),'្ង': ('ng', 0),'្ច': ('ch', 1),   
    '្ឆ': ('ch', 1),'្ជ': ('ch', 0),'្ឈ': ('ch', 0),
    '្ញ': ('nh', 0),'្ដ': ('d', 1),'្ឋ': ('th', 1),   
    '្ឌ': ('d', 0),'្ឍ': ('th', 0),'្ណ': ('n', 1),
    '្ត': ('t', 1),'្ថ': ('th', 1),'្ទ': ('t', 0),
    '្ធ': ('th', 0),'្ន': ('n', 0),
    '្ប': ('b', 1),
    '្ផ': ('ph', 1),'្ព': ('p', 0),'្ភ': ('ph', 0),
    '្ម': ('m', 0),'្យ': ('y', 0),'្រ': ('r', 0),
    '្ល': ('l', 0),'្វ': ('v', 0),'្ស': ('s', 1),
    '្ហ': ('h', 1)
}


# ---------------- VOWELS -----------------

# -----------------------------------------
# ---------- SUBSCRIPT VOWELS -------------
# -----------------------------------------
vowels_sub = {
    'ា': ('ea', 'a'),
    'ិ': ('i', 'e'),
    'ី': ('i', 'ei'),
    'ឹ': ('ue', 'oe'),
    'ឺ': ('ueu', 'eu'),
    'ុ': ('u', 'o'),
    'ូ': ('u', 'ou'),
    'ួ': ('uo', 'uo'),
    'ើ': ('eu', 'aeu'),
    'ឿ': ('oea', 'oea'),
    'ៀ': ('ie', 'ie'),
    'េ': ('e', 'e'),
    'ែ': ('eae', 'ae'),
    'ៃ': ('ey', 'ai'),
    'ោ': ('ou', 'ao'),
    'ៅ': ('ov', 'au')
}
# -----------------------------------------
# ---------- INDEPENDENT VOWELS -----------
# -----------------------------------------
vowels_ind = {
    'ឥ': 'e',
    'ឦ': 'ei',
    'ឧ': 'o',
    'ឩ': 'ou',
    'ឪ': 'au',
    'ឫ': 'rue',
    'ឬ': 'rueu',
    'ឭ': 'lue',
    'ឮ': 'lueu',
    'ឯ': 'ae',
    'ឰ': 'ai',
    'ឱ': 'ao',
    'ឲ': 'ao',
    'ឳ': 'au'
}
def is_subscript(c, vowels=vowels_sub, consonants=consonants_sub):
    if c in vowels:
        return 'v'
    elif c in consonants:
        return 'c'
    else:
        return None

punct = string.punctuation + '1234567890'
def transliterate_sentence(sentence):
    global consonants_ind
    global vowels_ind
    global consonants_sub
    global vowels_sub
    
    c2t = []
    for i, c in enumerate(sentence):
        if c in punct: c2t.append(None); continue
        if c == ' ': continue
            
#         if c == 'ប':
#             if i+1 < len(sentence):
#                 if not is_subscript(sentence[i+1]) is None:
#                     c2t.append((c, 'p'))
#                 else:
#                     c2t.append((c, 'b'))
#             else:
#                 c2t.append((c, 'b'))
#             continue
        t = is_subscript(c)
        if t is None:
            if c in consonants_ind:
                c2t.append((c, consonants_ind[c][0]))
            elif c in vowels_ind:
                c2t.append((c, vowels_ind[c]))
            else:
                c2t.append((c, ''))
        elif t == 'v':
            el = (c, '')
            for j in range(i-1, -1, -1):
                if sentence[j] in punct:
                    break
                elif sentence[j] in consonants_ind:
                    el = (c, vowels_sub[c][consonants_ind[sentence[j]][1]])
                    break
                elif sentence[j] in consonants_sub:
                    el = (c, vowels_sub[c][consonants_sub[sentence[j]][1]])
                    break
                elif j-1 >= 0 and sentence[j-1:j+1] in consonants_sub:
                    el = (c, vowels_sub[c][consonants_sub[sentence[j-1:j+1]][1]])
                    break
            c2t.append(el)
        else:
            c2t.append((c, consonants_sub[c][0]))
    
    tokens = word_tokenize(sentence, return_tokens=True)
    proc_tokens = []
    j = 0
    for i, token in enumerate(tokens):
        if token in punct: proc_tokens.append(token); j += 1; continue
        
        s = ''
        for c in token:
            s += c2t[j][1]
            j += 1
        proc_tokens.append(s)
    return ' '.join(proc_tokens)

labels = []
preds = []
for inp, tar in tqdm(zip(data_input, data_target)):
    labels.append(''.join(tar.strip().split()))
    preds.append(''.join(transliterate_sentence(inp).split()))
    print(tar.strip())
    print(preds[-1])
    print('----------')
print('CER:', cer(labels, preds))
print('Accuracy:', accuracy(labels, preds))
pd.DataFrame({
    'labels': labels,
    'preds': preds,
    'initial': data_input
}).to_csv('dummy_submission.csv', index=False)

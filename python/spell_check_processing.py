'''
Here we train spell checker model which is Seq2Seq model 
based on different operations in 2 Levenstein len
'''

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
from nltk.tokenize import word_tokenize

SEED = 1337
punctuation = string.punctuation + '1234567890’'
def set_seed(seed = 42, set_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if set_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
set_seed(SEED)

default_hyperparameters = {
    'd_model': 64,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.05,
}

def candidates(word): 
    "Generate possible spelling corrections for word."
    cand = list(set(list(edits1(word)) + list(edits2(word))))
    random.shuffle(cand)
    cand = cand[:100]+[word]
    return cand

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz '
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


# Parse arguments
parser = argparse.ArgumentParser(
    description="Train the model on dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "-m", "--model", help="Model architecture: currently only transformer", default="transformer")
parser.add_argument("-i", "--data-input", type=Path,
                    help="Specify file with input sequences", required=True)

parser.add_argument("-x", "--hyperparameters", type=json.loads,
                    help="Specify model hyperparameters", default='{}')
parser.add_argument("-e", "--max-epochs", type=int,
                    help="Maximum epochs count", default=250)
parser.add_argument("-l", "--log-interval", type=int,
                    help="Training logging interval", default=503)
parser.add_argument("-c", "--checkpoint-every", type=int,
                    help="Save checkpoint every N epochs", default=50)
parser.add_argument("--checkpoint-dir", type=Path,
                    help="Directory where to save checkpoints", default=Path("./checkpoints"))
parser.add_argument("-b", "--batch-size",
                    help="Batch size", type=int, default=32)
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
parser.add_argument("-d", "--device", help="Device: CPU or GPU", default="cuda")

args = parser.parse_args()

try:
    shutil.rmtree('examples/spell-check/')
except:
    pass

os.mkdir('examples/spell-check/')

f = open(args.data_input)
data_input = f.read().split('\n')[:-1]
f.close()
data = []
punct = string.punctuation
for d in data_input:
    d = ''.join([c for c in d if not c in punct])
    data += d.split()

X = []
y = []

for d in tqdm(data):
    cand = list(candidates(d))
    X += [d]*len(cand)
    y += cand

with open('examples/spell-check/input.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(X))
with open('examples/spell-check/target.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(y))
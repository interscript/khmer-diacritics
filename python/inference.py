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
from transformer import Transformer
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
from khmernltk import word_tokenize
import pickle


punc = string.punctuation+'0987654321'

parser = argparse.ArgumentParser(
    description="Train the model on dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-i", "--data-input", type=str,
                    help="Write sentence you want to convert", required=True)
parser.add_argument("-d", "--device", help="Device: CPU or GPU", default="cuda")
parser.add_argument("-en", "--experiment-name", help="Model name", type=str, default="sample")

args = parser.parse_args()
name = args.experiment_name
sentence = args.data_input
device = args.device

ckp = torch.load(f'models/{name}/{name}.bin')

with open(f'models/{name}/khmer.pickle', 'rb') as f:
    khmer_vocab = pickle.load(f)
khmer_vocab.set_default_index(0)
with open(f'models/{name}/latn.pickle', 'rb') as f:
    latn_vocab = pickle.load(f)
max_len = 47

model = Transformer(
    len(khmer_vocab),
    len(latn_vocab),
    d_model = 64,
    nhead = 8,
    num_encoder_layers = 4,
    num_decoder_layers = 4,
    dim_feedforward = 256,
    dropout = 0.05,
    max_len=max_len,
    src_pad_idx=0,
    device=device
)
model.load_state_dict(ckp)
model = model.to(device)

def process_line(line, vcb, pad=True):
    global max_len
    # Just split line
    line = ['<sos>']+list(line)+['<eos>']
    # Pad line to max_len
    if pad:
        line = line+['<pad>']*(max_len-len(line)) 
    # Convert to ids
    line = vcb(line)
    return line
def translate_sentence(model, 
                       sentence, 
                       device=device, 
                       khmer_vocab=khmer_vocab, 
                       latn_vocab=latn_vocab, 
                       max_length=max_len):
    tokens = process_line(sentence, khmer_vocab, False)
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    outputs = [latn_vocab["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
        best_guess = output.argmax(2)[:, -1].item()
        if best_guess == latn_vocab["<eos>"]:
            break
        outputs.append(best_guess)

    latn_itos = dict(map(lambda x: x[::-1], list(latn_vocab.get_stoi().items())))
    translated_sentence = [latn_itos[idx] for idx in outputs]
    return ''.join(translated_sentence[1:])

latn_sentence = ''
for token in word_tokenize(sentence):
    if token in punc or token == ' ':
        latn_sentence += token
    else:
        latn_sentence += translate_sentence(model, token)
print(latn_sentence)
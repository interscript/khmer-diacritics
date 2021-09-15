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
from torch.utils.data import DataLoader
from utils import cer, accuracy
import string
import wandb
import pickle
import re
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('GKLMIP/bert-khmer-base-uncased-tokenized')
vocab = dict(tokenizer.vocab)

def tokenize(s: str, max_len: int) -> torch.Tensor:
    global vocab
    tokens = [vocab['[CLS]']]+[vocab[c] for c in s if c in vocab][:max_len-2]+[vocab['[SEP]']]
    tokens = tokens + (max_len-len(tokens))*[vocab['[PAD]']]
    return torch.tensor(tokens)
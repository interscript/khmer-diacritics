import torch
import time
import numpy
import math
import yaml
import sys
import random
import re
import os
import pickle
from pathlib import Path
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tokenizer import tokenize

class Seq2SeqDataset(Dataset):
    def __init__(self, data, texts):
        self.data = data
        self.texts = texts
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_data': torch.tensor(self.data[idx][0]),
            'target_data': torch.tensor(self.data[idx][1]),
            'input_texts': self.texts[idx][0],
            'target_texts': self.texts[idx][1]
        }

class BERT2BERTDataset(Dataset):
    def __init__(self, data, texts):
        self.X = data
        self.y = texts
        self.tokenizer = BertTokenizer.from_pretrained('GKLMIP/bert-khmer-base-uncased-tokenized')
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.tokenizer(self.X[idx], add_special_tokens=True, return_tensors="pt", padding="max_length", truncation=True, max_length=9).input_ids.view(-1),
            'y': self.tokenizer(self.y[idx], add_special_tokens=True, return_tensors="pt", padding="max_length", truncation=True, max_length=21).input_ids.view(-1),
            'latn': self.y[idx]
        }
    
class BERT2BERTCharDataset(Dataset):
    def __init__(self, data, texts, ml1, ml2):
        self.X = data
        self.y = texts
        self.ml1 = ml1
        self.ml2 = ml2
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': tokenize(self.X[idx], self.ml1),
            'y': tokenize(self.y[idx], self.ml2),
            'latn': self.y[idx]
        }
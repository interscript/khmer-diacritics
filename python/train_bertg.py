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
from dataset import Seq2SeqDataset, BERT2BERTDataset, BERT2BERTCharDataset
from torch.utils.data import DataLoader
from utils import cer, accuracy
import string
import wandb
import pickle
import re
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer

wandb.login()
project = 'khmer-latn'
entity = 'sheminy32'

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
parser.add_argument("-e", "--max-epochs", type=int,
                    help="Maximum epochs count", default=250)
parser.add_argument("-l", "--log-interval", type=int,
                    help="Training logging interval", default=504)
parser.add_argument("-c", "--checkpoint-every", type=int,
                    help="Save checkpoint every N epochs", default=50)
parser.add_argument("--checkpoint-dir", type=Path,
                    help="Directory where to save checkpoints", default=Path("./checkpoints"))
parser.add_argument("-b", "--batch-size",
                    help="Batch size", type=int, default=32)
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
parser.add_argument("-d", "--device", help="Device: CPU or GPU", default="cuda")
parser.add_argument("-en", "--experiment-name", help="Experiment name", type=str, default="sample")
## TODO: Test all arguments

args = parser.parse_args()


f = open(args.data_input)
data_input = f.read().split('\n')
if data_input[-1] == '': data_input = data_input[:-1]
f.close()
data_input = list(map(lambda x: x.strip(), data_input))

f = open(args.data_target)
data_target = f.read().split('\n')
if data_target[-1] == '': data_target = data_target[:-1]
f.close()
data_target = list(map(lambda x: x.strip(), data_target))

MAX_LEN_INPUT = max(list(map(len, data_input)))
MAX_LEN_TARGET = max(list(map(len, data_target)))

device = args.device
lr = args.lr
batch_size = args.batch_size
model_name = args.model
max_epochs = args.max_epochs
log_interval = args.log_interval
checkpoint_every = args.checkpoint_every
checkpoint_dir = args.checkpoint_dir
experiment_name = args.experiment_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

wandb.init(project=project, entity=entity, id=experiment_name)

try:
    shutil.rmtree(f'models/{experiment_name}/')
except:
    pass
os.mkdir(f'models/{experiment_name}/')
# with open(f'models/{experiment_name}/khmer.pickle', 'wb') as f:
#     pickle.dump(khmer_vocab, f)
# with open(f'models/{experiment_name}/latn.pickle', 'wb') as f:
#     pickle.dump(latn_vocab, f)

# Generate train, eval, and test batches
zipped_texts = list(zip(data_input, data_target))
random.Random(SEED).shuffle(zipped_texts)

# # train - 90%, eval - 7%, test - 3%
train_texts = zipped_texts[0:int(len(zipped_texts)*0.9)]
eval_texts = zipped_texts[int(
    len(zipped_texts)*0.9) + 1:int(len(zipped_texts)*0.97)]
test_texts = zipped_texts[int(len(zipped_texts)*0.97):-1]

train_ds = BERT2BERTCharDataset(list(map(lambda x: x[0], train_texts)), list(map(lambda x: x[1], train_texts)), MAX_LEN_INPUT, MAX_LEN_TARGET)
eval_ds = BERT2BERTCharDataset(list(map(lambda x: x[0], eval_texts)), list(map(lambda x: x[1], eval_texts)), MAX_LEN_INPUT, MAX_LEN_TARGET)
test_ds = BERT2BERTCharDataset(list(map(lambda x: x[0], test_texts)), list(map(lambda x: x[1], test_texts)), MAX_LEN_INPUT, MAX_LEN_TARGET)

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=4)
eval_dl = DataLoader(eval_ds, shuffle=False, batch_size=batch_size, num_workers=4)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=4)

tokenizer = BertTokenizer.from_pretrained('GKLMIP/bert-khmer-base-uncased-tokenized')
encoder = BertGenerationEncoder.from_pretrained("GKLMIP/bert-khmer-base-uncased-tokenized", bos_token_id=101, eos_token_id=102)
decoder = BertGenerationDecoder.from_pretrained("GKLMIP/bert-khmer-base-uncased-tokenized", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)

def translate_sentences(model,
                       dl,
                       device=device):
    predictions = []
    for batch in tqdm(dl):
        out = model.generate(batch['X'].to(device))
        for line in out:
            predictions += [''.join([t for t in tokenizer.convert_ids_to_tokens(line) if not t in ['[CLS]', '[PAD]', '[SEP]']])]
    return predictions

## Training
best_model = None
best_val_loss = float("inf")

num_training_steps = batch_size*max_epochs
num_warmup_steps = int(num_training_steps*.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

total_loss = 0.0
sentences = None
true_sentences = None
start_time = time.time()

model = model.to(device)

for epoch in range(max_epochs):
    epoch_start_time = time.time()
    model.train()
    for i, batch in enumerate(train_dl):
        input_data = batch['X'].to(device)
        target_data = batch['y'].to(device)
        output = model(input_ids=input_data, decoder_input_ids=target_data, labels=target_data)
        loss = output.loss
        
        if sentences is None:
            true_sentences = list(batch['latn'])
        else:
            true_sentences += list(batch['latn'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'time: {elapsed:.3f}s | loss: {cur_loss:.4f} | step: {i+1:03d}/{len(train_dl)}')
            wandb.log({'loss/train': cur_loss})
            total_loss = 0
            true_sentences = None
            sentences = None
            start_time = time.time()

    # Evaluate
    model.eval()
    total_loss = 0.0
    true_sentences = None
    sentences = None
    with torch.no_grad():
        for i, batch in enumerate(eval_dl):
            input_data = batch['X'].to(device)
            target_data = batch['y'].to(device)
            output = model(input_ids=input_data, decoder_input_ids=target_data, labels=target_data)
            loss = output.loss

            if sentences is None:
                true_sentences = list(batch['latn'])
            else:
                true_sentences += list(batch['latn'])
            total_loss += loss.item()
        
        pred_sentences = translate_sentences(model, eval_dl)
        print(true_sentences[:2], pred_sentences[:2])
        elapsed = time.time() - start_time
        total_loss = total_loss / len(eval_dl)
        print(f'time: {elapsed:.3f}s | loss: {total_loss:.4f} | cer: {cer(true_sentences, pred_sentences):.4f} | accuracy: {accuracy(true_sentences, pred_sentences):.4f}')
        wandb.log({'loss/val': total_loss, 
                       'cer/val':cer(true_sentences, pred_sentences), 
                       'accuracy/val':accuracy(true_sentences, pred_sentences)})
        total_loss = 0
        true_sentences = []
        sentences = []
        pred_sentences = []
        start_time = time.time()
        
        if total_loss < best_val_loss:
            # Question: here we really save best model or only save reference?
            # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
            best_model = model
            best_val_loss = total_loss

    scheduler.step()


torch.save(model.state_dict(), f'models/{experiment_name}.bin')
wandb.finish()
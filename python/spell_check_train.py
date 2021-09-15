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

default_hyperparameters = {
    'd_model': 64,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.05,
}

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
parser.add_argument("-x", "--hyperparameters", type=json.loads,
                    help="Specify model hyperparameters", default='{}')
parser.add_argument("-e", "--max-epochs", type=int,
                    help="Maximum epochs count", default=250)
parser.add_argument("-l", "--log-interval", type=int,
                    help="Training logging interval", default=5000)
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

f = open(args.data_target)
data_target = f.read().split('\n')
if data_target[-1] == '': data_target = data_target[:-1]
f.close()

device = args.device
lr = args.lr
batch_size = args.batch_size
model_name = args.model
max_epochs = args.max_epochs
log_interval = args.log_interval
checkpoint_every = args.checkpoint_every
checkpoint_dir = args.checkpoint_dir
experiment_name = 'spell-'+args.experiment_name
print(experiment_name)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

config = default_hyperparameters.copy()
config['learning_rate'] = lr
config['batch_size'] = batch_size
config['max_epochs'] = max_epochs
config['model_name'] = model_name
wandb.init(project=project, entity=entity, id=experiment_name, config=config)


def find_checkpoint_index(n):
    matches = re.findall("checkpoint-([0-9]+)", n)
    if len(matches) > 0:
        return int(matches[0])
    else:
        return 0


last_checkpoint = None
sorted_checkpoints = sorted(
    filter(lambda f: re.match(
        "checkpoint-([0-9]+)\.zip", f), os.listdir(checkpoint_dir)),
    key=find_checkpoint_index)
if len(sorted_checkpoints) > 0:
    last_checkpoint = sorted_checkpoints[-1]

if last_checkpoint:
    print(f'Starting from checkpoint {last_checkpoint}')
    initial_epoch = find_checkpoint_index(last_checkpoint) + 1
else:
    initial_epoch = 0

input_texts = []
target_texts = []
input_vocab_counter = {'<pad>': 1, '<sos>': 1, '<eos>': 1}
target_vocab_counter = {'<pad>': 1, '<sos>': 1, '<eos>': 1}
target_vocab_counter.update({
    k:10 for k in 'qwertyuiopasdfghjklzxcvbnm '
})
input_vocab_counter.update({
    k:10 for k in 'qwertyuiopasdfghjklzxcvbnm '
})

for input_text, target_text in zip(data_input, data_target):
    input_text = input_text.strip()
    target_text = target_text.strip().lower()
    # Filter non-latin chars
    target_text = ''.join(filter(lambda x: not x in punctuation, list(target_text)))
    flag = False
    for c in target_text:
        if not c in target_vocab_counter: flag=True; break
    if flag: 
        continue
        
    input_text = ''.join(filter(lambda x: not x in punctuation, list(input_text)))
    flag = False
    for c in input_text:
        if not c in target_vocab_counter: flag=True; break
    if flag: 
        continue
    target_texts.append(target_text)
    input_texts.append(input_text)
print('///Texts are prepared')

khmer_vocab = Vocab(vocab(input_vocab_counter))
latn_vocab = Vocab(vocab(target_vocab_counter))

try:
    shutil.rmtree(f'models/{experiment_name}/')
except:
    pass
os.mkdir(f'models/{experiment_name}/')
with open(f'models/{experiment_name}/real.pickle', 'wb') as f:
    pickle.dump(khmer_vocab, f)
with open(f'models/{experiment_name}/syntetic.pickle', 'wb') as f:
    pickle.dump(latn_vocab, f)

# Generate train, eval, and test batches
zipped_texts = list(zip(input_texts, target_texts))
random.Random(SEED).shuffle(zipped_texts)

# # train - 90%, eval - 7%, test - 3%
data = zipped_texts[:log_interval*6]
train_texts = data[0:int(len(data)*0.9)]
eval_texts = data[int(
    len(data)*0.9) + 1:int(len(data)*0.97)]
test_texts = data[int(len(data)*0.97):-1]

# Max len with <sos>/<eos> tokens
max_len = max(list(map(len, input_texts+target_texts)))+2

# Save model config
with open(f'models/{experiment_name}/config.pickle', 'wb') as f:
    config = default_hyperparameters.copy()
    config['max_len'] = max_len
    config['src_pad_idx'] = 0
    config['src_vocab_size'] = len(khmer_vocab)
    config['trg_vocab_size'] = len(latn_vocab)
    pickle.dump(config, f)

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

train_data = list(zip(
    list(map(lambda x: process_line(x[0], khmer_vocab), train_texts)),
    list(map(lambda x: process_line(x[1], latn_vocab), train_texts))
))
eval_data = list(zip(
    list(map(lambda x: process_line(x[0], khmer_vocab), eval_texts)),
    list(map(lambda x: process_line(x[1], latn_vocab), eval_texts))
))
test_data = list(zip(
    list(map(lambda x: process_line(x[0], khmer_vocab), test_texts)),
    list(map(lambda x: process_line(x[1], latn_vocab), test_texts))
))

train_ds = Seq2SeqDataset(train_data, train_texts)
eval_ds = Seq2SeqDataset(eval_data, eval_texts)
test_ds = Seq2SeqDataset(test_data, test_texts)

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=4)
eval_dl = DataLoader(eval_ds, shuffle=False, batch_size=batch_size, num_workers=4)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=4)

print('///Datasets are prepared')

hyperparameters = {
    **default_hyperparameters,
    **args.hyperparameters,
    "input_vocab_size": len(khmer_vocab),
    "target_vocab_size": len(latn_vocab),
}

with open(Path(checkpoint_dir, 'vocabs.yaml'), 'w') as outfile:
    yaml.dump({
        "input": khmer_vocab.get_stoi(),
        "target": latn_vocab.get_stoi()
    }, outfile)

with open(Path(checkpoint_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump({
        "name": "transformer",
        # transform keys for ruby
        **{f':{k}': v for k, v in hyperparameters.items()}
    }, outfile)


assert model_name == "transformer", "Only transformer model is currently supported"

# load checkpoint data to resume
if last_checkpoint:
    input_zip_file = ZipFile(Path(checkpoint_dir, last_checkpoint))
    assert "metadata.yaml" in input_zip_file.namelist(
    ), "metadata.yaml is missing in model zip!"
    assert "vocabs.yaml" in input_zip_file.namelist(
    ), "vocabs.yaml is missing in model zip!"

    metadata = yaml.safe_load(input_zip_file.read('metadata.yaml'))
    vocabs = yaml.safe_load(input_zip_file.read('vocabs.yaml'))
    input_vocab = Vocab(vocabs['input'])
    target_vocab = Vocab(vocabs['target'])

    if "model.pth" in input_zip_file.namelist():
        model_dict = torch.load(BytesIO(input_zip_file.read('model.pth')))
        new_dict = {}
        # transforming the keys of model dict so that they
        # fit well to torch Transformer
        for key, value in model_dict.items():
            transformed_key = re.sub(r"layer(\d+)\.", r"layers.\1.", key)
            new_dict[transformed_key] = value
    else:
        print("model.pth is missing in model zip!")
        exit()

    model = transformer.Transformer(
        metadata[':input_vocab_size'],
        metadata[':target_vocab_size'],
        d_model=metadata[':d_model'],
        nhead=metadata[':nhead'],
        dim_feedforward=metadata[':dim_feedforward'],
        num_encoder_layers=metadata[':num_encoder_layers'],
        num_decoder_layers=metadata[':num_decoder_layers'],
        dropout=metadata[':dropout'],
        src_pad_idx=0,
        max_len=max_len,
        device=device
    )

    model.load_state_dict(new_dict)
else:
    # do not pass parameters that are not present, for defaults to pick up
    _h = {k: hyperparameters.get(k) for k, v in hyperparameters.items() if k not in [
        'input_vocab_size', 'target_vocab_size'] and v is not None}
    model = transformer.Transformer(
        hyperparameters['input_vocab_size'],
        hyperparameters['target_vocab_size'],
        src_pad_idx=0,
        device=device,
        max_len=max_len,
        **_h
    )

print('///Model is prepared')
    
def save_model(epoch):
    start_saving = time.time()
    state_dict = {k: v.data if isinstance(
        v, torch.nn.Parameter) else v for k, v in model.state_dict().items()}
    torch.save(state_dict, Path(checkpoint_dir, "model.pth"),
               _use_new_zipfile_serialization=True)

    # Zip generation
    input_filenames = ['model.pth', 'metadata.yaml', 'vocabs.yaml']
    zipfile_name = Path(checkpoint_dir, f"checkpoint-{epoch}.zip")
    if os.path.exists(zipfile_name):
        os.remove(zipfile_name)

    output_zip_file = ZipFile(zipfile_name, 'w', compression=ZIP_DEFLATED)

    for filename in input_filenames:
        output_zip_file.write(Path(checkpoint_dir, filename), filename)

    output_zip_file.close()

    print(f">> Saved checkpoint '{checkpoint_dir}/checkpoint-{epoch}.zip' in {round(time.time() - start_saving, 3)}s")

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

def translate_sentences(model, 
                       sentences, 
                       device=device, 
                       khmer_vocab=khmer_vocab, 
                       latn_vocab=latn_vocab, 
                       max_length=max_len):
    try:
        tokens = list(map(lambda x: process_line(x, khmer_vocab), sentences))
    except:
        print(sentences)
        exit()
    # Convert to Tensor
    sentences_tensor = torch.LongTensor(tokens).to(device)

    outputs = torch.LongTensor([[latn_vocab["<sos>"]]]*sentences_tensor.shape[0])
    for i in range(max_length):
        trg_tensor = outputs.to(device)

        with torch.no_grad():
            output = model(sentences_tensor, trg_tensor)
        best_guesses = output.argmax(2)[:, -1].unsqueeze(1).cpu()
        outputs = torch.cat((outputs, best_guesses), 1)

    latn_itos = dict(map(lambda x: x[::-1], list(latn_vocab.get_stoi().items())))
    outputs = list(outputs.numpy())
    translated_sentences = []
    for output in outputs:
        pred_sentence = []
        for idx in output[1:]:
            if idx == latn_vocab["<eos>"]: break
            pred_sentence.append(latn_itos[idx])
        pred_sentence = ''.join(pred_sentence)
        translated_sentences.append(pred_sentence)
    return translated_sentences

## Training
best_model = None
best_val_loss = float("inf")

num_training_steps = batch_size*max_epochs
num_warmup_steps = int(num_training_steps*.1)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

total_loss = 0.0
pred_sentences = []
sentences = []
true_sentences = []
start_time = time.time()

model = model.to(device)

print('///Model training is started!')
for epoch in range(initial_epoch, initial_epoch + max_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    for i, batch in enumerate(train_dl):
        input_data = batch['input_data'].to(device)
        target_data = batch['target_data'].to(device)
        output = model(
            input_data, 
            target_data[:, :-1]
        )
        output = output.reshape(-1, output.shape[2])
        target = target_data[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        sentences += batch['input_texts']
        true_sentences += batch['target_texts']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        if (i+1) == len(train_dl):
            pred_sentences = translate_sentences(model, sentences)
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'time: {elapsed:.3f}s | loss: {cur_loss:.4f} | step: {i+1:03d}/{len(train_dl)} | cer: {cer(true_sentences, pred_sentences):.4f} | accuracy: {accuracy(true_sentences, pred_sentences):.4f}')
            wandb.log({'loss/train': cur_loss, 
                       'cer/train':cer(true_sentences, pred_sentences), 
                       'accuracy/train':accuracy(true_sentences, pred_sentences)})
            total_loss = 0
            true_sentences = []
            sentences = []
            pred_sentences = []
            start_time = time.time()

    if (epoch > 0 and epoch % checkpoint_every == 0) or (max_epochs and epoch == (initial_epoch + max_epochs)):
        save_model(epoch)

    # Evaluate
    model.eval()
    total_loss = 0.0
    true_sentences = []
    sentences = []
    pred_sentences = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dl):
            input_data = batch['input_data'].to(device)
            target_data = batch['target_data'].to(device)
            output = model(
                input_data, 
                target_data[:, :-1]
            )
            output = output.reshape(-1, output.shape[2])
            target = target_data[:, 1:].reshape(-1)

            loss = criterion(output, target)

            sentences += batch['input_texts']
            true_sentences += batch['target_texts']

            total_loss += loss.item()
        
        pred_sentences = translate_sentences(model, sentences)
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
shutil.rmtree(checkpoint_dir)
wandb.finish()
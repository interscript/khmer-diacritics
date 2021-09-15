import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.hid_dim = kwargs['hid_dim']
        self.n_layers = kwargs['n_layers']
        self.input_dim = kwargs['input_dim']
        self.emb_dim = kwargs['emb_dim']
        self.dropout = kwargs['dropout']
        self.bidirectional = kwargs.get('bidirectional', True)
        
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        
        self.rnn = nn.LSTM(self.emb_dim, 
                           self.hid_dim, 
                           num_layers=self.n_layers, 
                           dropout=self.dropout, 
                           bidirectional=self.bidirectional)
        
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.hid_dim = kwargs['hid_dim']
        self.n_layers = kwargs['n_layers']
        self.output_dim = kwargs['output_dim']
        self.emb_dim = kwargs['emb_dim']
        self.dropout = kwargs['dropout']
        self.bidirectional = kwargs.get('bidirectional', True)
        
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        
        self.rnn = nn.LSTM(self.emb_dim, 
                           self.hid_dim, 
                           num_layers=self.n_layers, 
                           dropout=self.dropout, 
                           bidirectional=self.bidirectional)
        self.n_ways = 2 if self.bidirectional else 1
        self.fc_out = nn.Linear(self.hid_dim*self.n_ways, self.output_dim)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src = src.T
        trg = trg.T
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len+1, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_out, hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)

            # replace predictions in a tensor holding predictions for each token
            outputs[t+1] = output

            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions.
            input = output.argmax(1)

        return outputs[1:].permute(1, 0, 2)
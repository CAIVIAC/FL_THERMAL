# RESOURCES
# https://github.com/bentrevett/pytorch-seq2seq/tree/master
# https://github.com/maxbrenner-ai/seq2seq-time-series-forecasting-fully-recurrent/blob/main/notebook.ipynb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
import random
from time import time
import math
from copy import deepcopy
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score
torch.manual_seed(1234)



use_attention = True 
avg_history = 8      
lr = 0.0005          
num_epochs = 100     
batch_size = 64     
num_gru_layers = 1 
grad_clip = 1.0    
dropout = 0.     
metric   = 'mae'
feat_length = 32
pred_length = 5
input_seq_length  = 1
output_seq_length = 1
hidden_size = feat_length*2
target_indices = [0,1,2,3,4]
device = torch.device('cuda')




# DATA PREPARE #############################################################################################################################
def load_data(device, avg_history=20):
    X_data = pd.read_csv('./logs/Data/X.csv')
    X_keys = X_data.keys()
    # print(X_data.shape, X_keys)
    Y_data = pd.read_csv('./logs/Data/Y.csv')
    Y_keys = Y_data.keys()
    # print(Y_data.shape, Y_keys)
    X = X_data.to_numpy().transpose(1, 0)
    Y = Y_data.to_numpy().transpose(1, 0)
    avg_time_series = []
    for time_series_idx in range(X.shape[0]):
        time_series_i = X[time_series_idx]
        avg = np.array([time_series_i[i-avg_history:i].mean(axis=0) for i in range(avg_history, time_series_i.shape[0])])
        avg_time_series.append(avg)
    avg_time_series = np.stack(avg_time_series, 0)
    X = X[:, avg_history:]
    Y = Y[:, avg_history:]
    data = np.concatenate((X, avg_time_series, Y), 0) # [16+16+5, :]
    print('feature-target, time-series', data.shape)
    return data, X_keys, Y_keys

def split_data(data):
    num_timesteps = data.shape[1]
    train_end_idx = round(0.6 * num_timesteps)
    train_data = data[:, : train_end_idx]
    test_end_idx = train_end_idx + round(0.2 * num_timesteps)
    test_data = data[:, train_end_idx : test_end_idx]
    val_data = data[:, test_end_idx : ]
    return train_data, val_data, test_data    
   
def create_sequences(data, feat_length, pred_length, device):
    enc_inputs  = torch.tensor(np.expand_dims(deepcopy(data[:feat_length,:]).transpose(1,0),1), dtype=torch.float).to(device)
    dec_targets = torch.tensor(np.expand_dims(deepcopy(data[-pred_length:,:]).transpose(1,0),1), dtype=torch.float).to(device)
    print('enc_inputs: ', enc_inputs.shape, '    dec_targets: ', dec_targets.shape)
    return {'enc_inputs': enc_inputs, 'dec_targets': dec_targets}

#### ------------------------------------------------------------------------------------------
# Load raw data
data, X_keys, Y_keys = load_data(device, avg_history=avg_history)
# Split into train/val/test
data_splits = split_data(data)
print('train/val/test data splits: ', data_splits[0].shape, data_splits[1].shape, data_splits[2].shape)
# Create the sequences>>   data_splits[0]: train split, data_splits[1]: val split, data_splits[2]: testing split
data_splits = create_sequences(data_splits[0], feat_length, pred_length, device), \
              create_sequences(data_splits[1], feat_length, pred_length, device), \
              create_sequences(data_splits[2], feat_length, pred_length, device)
train_data, val_data, test_data = data_splits[0], data_splits[1], data_splits[2]
print('DATA PRECESSING DONE!\n') ##########################################################




# MODEL #############################################################################################################################
def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0.)
    return layer

class Encoder(nn.Module):
    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(enc_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        
    def forward(self, inputs):
        self.gru.flatten_parameters()
        # inputs: (batch size, input seq len, num enc features) 
        output, hidden = self.gru(inputs)
        # output: (batch size, input seq len, hidden size)     
        # hidden: (num gru layers, batch size, hidden size)    
        return output, hidden

# Decoder superclass whose forward is called by Seq2Seq but other methods filled out by subclasses
class DecoderBase(nn.Module):
    def __init__(self, device, dec_target_size, target_indices, dist_size):
        super().__init__()
        self.device = device
        self.target_indices = target_indices    # [0,1,2,3,4]
        self.target_size = dec_target_size      # 5
        self.dist_size = dist_size              # 1
    
    # Have to run one step at a time unlike with the encoder since sometimes not teacher forcing
    def run_recurrent_step(self, inputs, hidden):
        raise NotImplementedError()
    
    def forward(self, inputs, hidden):
        # inputs: (batch size, input seq len, hidden size)                            
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state  
        outputs, hidden = self.run_recurrent_step(inputs, hidden)
        return outputs           

class DecoderVanilla(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, dist_size, device):
        super().__init__(device, dec_target_size, target_indices, dist_size)
        self.gru = nn.GRU(dec_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out = layer_init(nn.Linear(hidden_size + dec_feature_size, dec_target_size * dist_size))
    
    def run_recurrent_step(self, inputs, hidden):
        self.gru.flatten_parameters()
        # inputs: (batch size, 1, num dec tagets)
        # hidden: (num gru layers, batch size, hidden size)
        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size, num_gru_layers):
        super().__init__()
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        # decoder_hidden_final_layer: (batch size, hidden size)
        # encoder_outputs: (batch size, input seq len, hidden size)
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        # Compare decoder hidden state with each encoder output using a learnable tanh layer
        # energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        weightings = torch.sigmoid(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return weightings
    
class DecoderWithAttention(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, dist_size, device):
        super().__init__(device, dec_target_size, target_indices, dist_size)
        self.attention_model = Attention(hidden_size, num_gru_layers)
        self.gru = nn.GRU(dec_feature_size + hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out = layer_init(nn.Linear(dec_feature_size + hidden_size + hidden_size, dec_target_size * dist_size))

    def run_recurrent_step(self, inputs, hidden):
        self.gru.flatten_parameters()
        # inputs: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden size)
 
        # Get attention weightings
        weightings = self.attention_model(hidden[-1], inputs)
        weighted_in = weightings*inputs

        # output: (batch size, 1, hidden size)
        output, hidden = self.gru(torch.cat((inputs, weighted_in), dim=2), hidden)

        # out input: (batch size, 1, hidden size + hidden size)
        output = self.out(torch.cat((output, weighted_in, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        
        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr, grad_clip):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = torch.optim.Adam(self.parameters(), lr)
        self.grad_clip = grad_clip
    
    def forward(self, enc_inputs):
        # enc_inputs: (batch size, input seq length, num enc features)
        # enc_outputs: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        enc_outputs, hidden = self.encoder(enc_inputs)
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = self.decoder(enc_outputs, hidden)
        return outputs

    def compute_loss(self, prediction, target):
        # prediction: (batch size, dec seq len, num targets, num dist params)
        # target: (batch size, dec seq len, num targets)
        loss = F.mse_loss(prediction.squeeze(-1), target)
        return loss if self.training else loss.item()
    
    def optimize(self, prediction, target):
        # prediction & target: (batch size, seq len, output dim)
        self.opt.zero_grad()
        loss = self.compute_loss(prediction, target)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
        return loss.item()
    
print('MODEL CONSTRCTION DONE!\n')




# Run Training #############################################################################################################################
# New generator every epoch
def batch_generator(data, batch_size, unscale=False, shuffle=True):
    enc_inputs, dec_targets = data['enc_inputs'], data['dec_targets']
    if shuffle:
        indices = torch.randperm(enc_inputs.shape[0])
    else:
        indices = range(enc_inputs.shape[0])
    # breakpoint()
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_enc_inputs = enc_inputs[batch_indices]
        batch_dec_targets = dec_targets[batch_indices]
        if batch_enc_inputs.shape[0] < batch_size:
            break
        yield batch_enc_inputs, batch_dec_targets       

def train(model, train_data, batch_size):
    model.train()
    epoch_loss = 0.
    num_batches = 0
    for batch_enc_inputs, batch_dec_targets in batch_generator(train_data, batch_size):
        output = model(batch_enc_inputs)
        loss = model.optimize(output, batch_dec_targets)
        epoch_loss += loss
        num_batches += 1
    return epoch_loss / num_batches

def evaluate(model, val_data, batch_size):
    model.eval()
    epoch_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for batch_enc_inputs, batch_dec_targets in batch_generator(val_data, batch_size):
            output = model(batch_enc_inputs)
            loss = model.compute_loss(output, batch_dec_targets)
            epoch_loss += loss
            num_batches += 1
    return epoch_loss / num_batches

#### ------------------------------------------------------------------------------------------
dist_size = 1
enc_feature_size = train_data['enc_inputs'].shape[-1]
dec_target_size  = train_data['dec_targets'].shape[-1]
dec_feature_size = hidden_size
print(enc_feature_size, dec_target_size)

encoder = Encoder(enc_feature_size, hidden_size, num_gru_layers, dropout)  # GRU encoder
decoder_args = (dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, dist_size, device)
decoder = DecoderWithAttention(*decoder_args) if use_attention else DecoderVanilla(*decoder_args)
seq2seq = Seq2Seq(encoder, decoder, lr, grad_clip).to(device)
best_val, best_model = float('inf'), None
for epoch in range(num_epochs):
    start_t = time()
    train_loss = train(seq2seq, train_data, batch_size)
    val_loss = evaluate(seq2seq, val_data, batch_size)

    new_best_val = False
    if val_loss < best_val:
        new_best_val = True
        best_val = val_loss
        best_model = deepcopy(seq2seq)                                                           ############ best model
    print(f'Epoch {epoch+1} => Train loss: {train_loss:.5f},',
          f'Val loss: {val_loss:.5f},',
          f'Took {(time() - start_t):.1f} s{"      (NEW BEST)" if new_best_val else ""}')
    
print('TRAINING DONE!\n')




# Test Evaluation #############################################################################################################################
data_to_eval = test_data
best_model.eval()
trained_model_losses = []
for batch_enc_inputs, batch_dec_targets in batch_generator(data_to_eval, 32):
    outputs = best_model(batch_enc_inputs)
    trained_model_loss = best_model.compute_loss(outputs, batch_dec_targets)
    trained_model_losses.append(trained_model_loss)
print(f'Trained model losses: {np.mean(trained_model_losses):.5f}')
print('TESTING DONE!\n')




# Visualize #############################################################################################################################
visual_data = train_data
best_model.eval()
outputs = []
targets = []
with torch.no_grad():
    for batch_enc_inputs, batch_dec_targets in batch_generator(visual_data, 32, unscale=True, shuffle=False):
        outputs.append(best_model(batch_enc_inputs).squeeze())
        targets.append(batch_dec_targets.squeeze())
outputs = torch.cat(outputs)
targets = torch.cat(targets)
# print(outputs.shape, targets.shape)
for idx in range(targets.shape[1]):
    plt.figure(figsize=(10,5))
    x = list(range(targets.shape[0]))
    plt.plot(x, targets[:,idx].cpu())
    plt.plot(x, outputs[:,idx].cpu())
    score = sklearn_metric_loss_score(metric, outputs[:,idx].cpu(), targets[:,idx].cpu())
    print(' training ' + Y_keys[idx] + '  ' + metric + ': ' + f'{score:.3f}')
    plt.title(Y_keys[idx] + '  ' + metric + ': ' + f'{score:.3f}')
    plt.show()
    # plt.savefig('./logs/vis/train_' + Y_keys[idx] + '_S2S1[e' + str(num_epochs) + '].png')
print('- - '*10)

visual_data = test_data
best_model.eval()
outputs = []
targets = []
with torch.no_grad():
    for batch_enc_inputs, batch_dec_targets in batch_generator(visual_data, 32, unscale=True, shuffle=False):
        outputs.append(best_model(batch_enc_inputs).squeeze())
        targets.append(batch_dec_targets.squeeze())
outputs = torch.cat(outputs)
targets = torch.cat(targets)
# print(outputs.shape, targets.shape)
np.save('./logs/vis/S2S1', outputs.cpu().numpy())
for idx in range(targets.shape[1]):
    plt.figure(figsize=(10,5))
    x = list(range(targets.shape[0]))
    plt.plot(x, targets[:,idx].cpu())
    plt.plot(x, outputs[:,idx].cpu())
    score = sklearn_metric_loss_score(metric, outputs[:,idx].cpu(), targets[:,idx].cpu())
    print(' testing  ' + Y_keys[idx] + '  ' + metric + ': ' + f'{score:.3f}')
    plt.title(Y_keys[idx] + '  ' + metric + ': ' + f'{score:.3f}')
    plt.show()
    # plt.savefig('./logs/vis/test_' + Y_keys[idx] + '_S2S1[e' + str(num_epochs) + '].png')

print('VISUALIZATION DONE!\n')
print('SEQ2SEQ DONE !\n\n\n')
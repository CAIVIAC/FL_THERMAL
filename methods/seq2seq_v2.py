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
input_seq_length  = 8 
lr = 0.0005           
num_epochs = 100      
batch_size = 64      
num_gru_layers = 1   
grad_clip = 1.0     
dropout = 0.        
metric   = 'mae'
feat_length = 16
pred_length = 5
output_seq_length = 1
hidden_size = feat_length*2
target_indices = [0,1,2,3,4]
device = torch.device('cuda')


 

# DATA PREPARE #############################################################################################################################
def load_data(device):
    X_data = pd.read_csv('./logs/Data/X.csv') 
    X_keys = X_data.keys()
    # print(X_data.shape, X_keys)
    Y_data = pd.read_csv('./logs/Data/Y.csv') 
    Y_keys = Y_data.keys()
    # print(Y_data.shape, Y_keys)
    X = X_data.to_numpy().transpose(1, 0)   
    Y = Y_data.to_numpy().transpose(1, 0)   
    data = tensor(np.concatenate((X, Y), 0), dtype=torch.float).to(device) # data: (num time series, timesteps)
    return data, X_keys, Y_keys 

def split_data(data):
    num_timesteps = data.shape[1]
    train_end_idx = round(0.6 * num_timesteps)
    train_data = data[:, : train_end_idx]
    test_end_idx = train_end_idx + round(0.2 * num_timesteps)
    test_data = data[:, train_end_idx : test_end_idx]
    val_data = data[:, test_end_idx : ]
    return train_data, val_data, test_data    

def create_sequences(data, input_seq_length, output_seq_length, feat_length, pred_length):
    enc_inputs, dec_targets = [], []
    # Loop over the starting timesteps of the sequences
    for timestep in range(data.shape[1] - (input_seq_length + output_seq_length) + 1):
        enc_inputs.append(data[:feat_length, timestep : timestep+input_seq_length].transpose(1,0)) 
        dec_targets.append(data[-pred_length:, timestep : timestep+input_seq_length+output_seq_length-1].transpose(1,0))
    enc_inputs = torch.stack(enc_inputs);       # enc_inputs: (num seq by time, num time series, input seq len, num features)
    dec_targets = torch.stack(dec_targets);     # dec_targets: (num seq by time, num time series, output seq len, num targets)
    print('enc_inputs: ', enc_inputs.shape, '    dec_targets: ', dec_targets.shape)
    return {'enc_inputs': enc_inputs, 'dec_targets': dec_targets}

#### ------------------------------------------------------------------------------------------
# Load data
data, X_keys, Y_keys = load_data(device)
# Split into train/val/test
data_splits = split_data(data)
print('train/val/test data splits: ', data_splits[0].shape, data_splits[1].shape, data_splits[2].shape)
# Create the sequences>>   data_splits[0]: train split, data_splits[1]: val split, data_splits[2]: testing split
data_splits = create_sequences(data_splits[0], input_seq_length, output_seq_length, feat_length, pred_length), \
              create_sequences(data_splits[1], input_seq_length, output_seq_length, feat_length, pred_length), \
              create_sequences(data_splits[2], input_seq_length, output_seq_length, feat_length, pred_length)
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
        # inputs: (batch size, 5, num dec tagets)
        # hidden: (num gru layers, batch size, hidden size)
        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        # output: (batch size, 5, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size//2)
        self.k = nn.Linear(hidden_size, hidden_size//2)
    
    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        hidden = decoder_hidden_final_layer.unsqueeze(1)
        attention = torch.cosine_similarity(self.q(encoder_outputs).unsqueeze(2), self.k(hidden).unsqueeze(1), dim=-1) 
        weightings = F.log_softmax(attention, dim=1)
        weightings = torch.transpose(weightings, 1, 2)
        return weightings
    
class DecoderWithAttention(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, num_gru_layers, target_indices, dropout, dist_size, device):
        super().__init__(device, dec_target_size, target_indices, dist_size)
        self.attention_model = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out1 = nn.Linear(hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, dec_target_size * dist_size)
    def run_recurrent_step(self, inputs, hidden):
        self.gru.flatten_parameters()
        # inputs: (batch size, input seq len, hidden size)  
        # hidden: (num gru layers, batch size, hidden size) 
        
        # Get attention weightings: (batch size, input seq len, 1)
        weightings = self.attention_model(hidden[-1], inputs) 
        weighted_sum = torch.bmm(weightings, inputs)
                
        # output: (batch size, 1, hidden size)
        output, hidden = self.gru(weighted_sum, hidden)
        output = self.out2(F.relu(self.out1(output)))

        # output = self.out(output)
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
        loss = F.l1_loss(prediction.squeeze(-1), target)
        return loss if self.training else loss.item()
    
    def optimize(self, prediction, target):
        # prediction & target: (batch size, seq len, output dim)
        self.opt.zero_grad()
        loss = self.compute_loss(prediction[:,-1:,], target[:,-1:,])
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
        best_model = deepcopy(seq2seq)
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
        outputs.append(best_model(batch_enc_inputs)[:,-1:,:,:].squeeze())
        targets.append(batch_dec_targets[:,-1:,:].squeeze())
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
    # plt.savefig('./logs/vis/train_' + Y_keys[idx] + '_S2S2[e' + str(num_epochs) + '].png')
print('- - '*10)

visual_data = test_data
best_model.eval()
outputs = []
targets = []
with torch.no_grad():
    for batch_enc_inputs, batch_dec_targets in batch_generator(visual_data, 32, unscale=True, shuffle=False):
        outputs.append(best_model(batch_enc_inputs)[:,-1:,:,:].squeeze())
        targets.append(batch_dec_targets[:,-1:,:].squeeze())
outputs = torch.cat(outputs)
targets = torch.cat(targets)
with open('./logs/vis/RF.npy', 'rb') as f:  RF=np.load(f)
RF = RF[-outputs.shape[0]:,:]
with open('./logs/vis/XGB.npy', 'rb') as f:  XGB=np.load(f)
XGB = XGB[-outputs.shape[0]:,:]
with open('./logs/vis/S2S1.npy', 'rb') as f:  S2S1=np.load(f)
S2S1 = S2S1[-outputs.shape[0]:,:]
# print(outputs.shape, targets.shape)
np.save('./logs/vis/S2S2', outputs.cpu().numpy())
for idx in range(targets.shape[1]):
    plt.figure(figsize=(10,5))
    x = list(range(targets.shape[0]))
    plt.plot(x, targets[:,idx].cpu(), color='deepskyblue')
    plt.plot(x, RF[:,idx], color='tomato')
    plt.plot(x, XGB[:,idx], color='pink')
    plt.plot(x, S2S1[:,idx], color='limegreen')
    plt.plot(x, outputs[:,idx].cpu(), color='green')
    score = sklearn_metric_loss_score(metric, outputs[:,idx].cpu(), targets[:,idx].cpu())
    score_RF = sklearn_metric_loss_score(metric, RF[:,idx], targets[:,idx].cpu())
    score_XGB = sklearn_metric_loss_score(metric, XGB[:,idx], targets[:,idx].cpu())
    score_S2S1 = sklearn_metric_loss_score(metric, S2S1[:,idx], targets[:,idx].cpu())
    print(' testing  ' + Y_keys[idx] + '  ' + metric + ': ' + f'{score:.3f}')
    plt.title(Y_keys[idx] + '  ' + metric + ': ' + f'{score_RF:.3f}(RF)  ' + f'{score_XGB:.3f}(XGB)  ' + f'{score_S2S1:.3f}(S2S1)  ' f'{score:.3f}(S2S2)')
    plt.figtext(0.73, 0.15, "GT", color='deepskyblue')
    plt.figtext(0.76, 0.15, "RF", color='tomato')
    plt.figtext(0.78, 0.15, "XGB", color='pink')
    plt.figtext(0.82, 0.15, "S2S1", color='limegreen')
    plt.figtext(0.86, 0.15, "S2S2", color='green')
    plt.show()
    plt.savefig('./logs/vis/test_' + Y_keys[idx] + '_S2S2[e' + str(num_epochs) + '].png')

print('VISUALIZATION DONE!\n')
print('SEQ2SEQ DONE !\n\n\n')
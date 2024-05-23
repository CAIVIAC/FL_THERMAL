import os
import torch
from torch import tensor
from copy import deepcopy
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from flaml import AutoML
from flaml.automl.data import load_openml_dataset
from flaml.automl.ml import sklearn_metric_loss_score
from flaml.automl.data import get_output_from_log
from torch.utils.tensorboard import SummaryWriter


device = 'cuda'
metric = 'mae' 
budget = 20
output_seq_length = 5
avg_history = 8
raw_data = True
if raw_data:
    input_seq_length = 16
else: # use averaged data
    input_seq_length = 32


# DATA PREPARE #############################################################################################################################
exp_name = 'XGB[b'+ str(budget)+']'
log_dir  = os.path.join('./logs/'+exp_name)
swriter  = SummaryWriter(log_dir)
def writeLog(keys, metricValues, nameStr):
    df = pd.DataFrame({'keys': keys, 'values': metricValues})
    colors = cm.rainbow(np.linspace(0, 1, len(df)))
    fig = plt.figure()
    plt.bar(df['keys'], df['values'], color=colors)    
    swriter.add_figure(nameStr, fig)

def load_data(device, use_raw_data=raw_data, avg_history=20):
    X_data = pd.read_csv('./logs/Data/X.csv')
    X_keys = X_data.keys()
    Y_data = pd.read_csv('./logs/Data/Y.csv')
    Y_keys = Y_data.keys()
    X = X_data.to_numpy().transpose(1, 0)
    Y = Y_data.to_numpy().transpose(1, 0)
    if use_raw_data:
        data = tensor(np.concatenate((X, Y), 0), dtype=torch.float).to(device) # data: (num time series, timesteps)
    else:
        avg_time_series = []
        for time_series_idx in range(X.shape[0]):
            time_series_i = X[time_series_idx]
            avg = np.array([time_series_i[i-avg_history:i].mean(axis=0) for i in range(avg_history, time_series_i.shape[0])])
            avg_time_series.append(avg)
        avg_time_series = np.stack(avg_time_series, 0)
        if X_keys[0]=='時間(s)':
            avg_time_series = avg_time_series[1:,:]
        X = X[:, avg_history:]
        Y = Y[:, avg_history:]
        data = np.concatenate((X, avg_time_series, Y), 0)
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

def create_sequences(data, input_seq_length, output_seq_length, device):
    enc_inputs  = torch.tensor(np.expand_dims(deepcopy(data[:input_seq_length,:]).transpose(1,0),1), dtype=torch.float).to(device) 
    dec_targets = torch.tensor(np.expand_dims(deepcopy(data[-output_seq_length:,:]).transpose(1,0),1), dtype=torch.float).to(device)
    print('enc_inputs: ', enc_inputs.shape, '    dec_targets: ', dec_targets.shape)
    return {'enc_inputs': enc_inputs, 'dec_targets': dec_targets}

#### ------------------------------------------------------------------------------------------
# Load raw data
data, X_keys, Y_keys = load_data(device, use_raw_data=False, avg_history=avg_history)
# Split into train/val/test
data_splits = split_data(data)
print('train/val/test data splits: ', data_splits[0].shape, data_splits[1].shape, data_splits[2].shape)
# Create the sequences
data_splits = create_sequences(data_splits[0], input_seq_length, output_seq_length, device), \
               create_sequences(data_splits[1], input_seq_length, output_seq_length, device), \
               create_sequences(data_splits[2], input_seq_length, output_seq_length, device)
print(data_splits[0]['enc_inputs'].shape) 
print(data_splits[0]['dec_targets'].shape) 
train_data, val_data, test_data = data_splits[0], data_splits[1], data_splits[2]
print(train_data['enc_inputs'].shape, val_data['enc_inputs'].shape, test_data['enc_inputs'].shape)                                   
print(train_data['enc_inputs'].shape, train_data['dec_targets'].shape)
print('DATA PRECESSING DONE!\n')


# Run Training/Testing #############################################################################################################################
X_train = train_data['enc_inputs'].squeeze().cpu().numpy()
Y_train = train_data['dec_targets'].squeeze().cpu().numpy()
X_test  = test_data['enc_inputs'].squeeze().cpu().numpy()
Y_test  = test_data['dec_targets'].squeeze().cpu().numpy()
trainList, testList = [], []
y_pred_train_list, y_pred_test_list = [],[]
for id_Y in range(len(Y_keys)):
    y_train = Y_train[:,id_Y]
    y_test  = Y_test[:,id_Y]
    # model definition
    automl = AutoML()
    settings = {
        "time_budget": budget,          # total running time in seconds
        "metric": metric,               # primary metric for regression
        "estimator_list": ['xgboost'],  # ML learner
        "task": 'regression',           # task type
        "log_file_name": 'tmp.log', # flaml log file
        "seed": 1234,              # random seed
    }
    # logs
    time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    # training
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    # testing
    y_pred_train = automl.predict(X_train)
    y_pred_test  = automl.predict(X_test)
    y_pred_train_list.append(y_pred_train)
    y_pred_test_list.append(y_pred_test)
    trainList.append(sklearn_metric_loss_score(metric, y_pred_train, y_train))
    testList.append(sklearn_metric_loss_score(metric, y_pred_test, y_test))
writeLog(Y_keys, trainList, 'Training')
writeLog(Y_keys, testList, 'Testing')


# Visualize ######################################################################################################################################################
outputs = np.stack(y_pred_train_list).transpose(1, 0)
targets = Y_train
for idx in range(targets.shape[1]):
    plt.figure(figsize=(10,5))
    x = list(range(targets.shape[0]))
    plt.plot(x, targets[:,idx])
    plt.plot(x, outputs[:,idx])
    print(' training ' + Y_keys[idx] + '  ' + metric + ': ' + f'{trainList[idx]:.3f}')
    plt.title(Y_keys[idx] + '  ' + metric + ': ' + f'{trainList[idx]:.3f}')
    plt.show()
    # plt.savefig('./logs/vis/train_' + Y_keys[idx] + '_XGB[b' + str(budget) + '].png')
print('- - '*10)
outputs = np.stack(y_pred_test_list).transpose(1, 0)
targets = Y_test
np.save('./logs/vis/XGB', outputs)
for idx in range(targets.shape[1]):
    plt.figure(figsize=(10,5))
    x = list(range(targets.shape[0]))
    plt.plot(x, targets[:,idx])
    plt.plot(x, outputs[:,idx])
    print(' testing  ' + Y_keys[idx] + '  ' + metric + ': ' + f'{testList[idx]:.3f}')
    plt.title(Y_keys[idx] + '  ' + metric + ': ' + f'{testList[idx]:.3f}')
    plt.show()
    # plt.savefig('./logs/vis/test_' + Y_keys[idx] + '_XGB[b' + str(budget) + '].png')
print('VISUALIZATION DONE!\n')
print("DONE -------------")
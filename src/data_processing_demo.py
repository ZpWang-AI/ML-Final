from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# 特征数量
num_features = 7

n_out = 96
n_in = 96
batch_size = 64

if n_out == 96:
    file = './datas/train_data/EETh1_96_96.csv'
else:
    file = './datas/train_data/EETh1_96_336.csv'

# load dataset
dataset = read_csv(file, header=0, index_col=0)
values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

# split dataset 6:2:2
train_num = int(0.6*len(values))
test_num = int(0.2*len(values))
train = values[:train_num, :]
val = values[train_num:train_num+test_num, :]
test = values[train_num+test_num:, :]

# split into input and outputs
split_point = n_in*num_features
train_X, train_Y = train[:, :split_point], train[:, split_point:]
val_X, val_Y = val[:, :split_point], val[:, split_point:]
test_X, test_Y = test[:, :split_point], test[:, split_point:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_in, num_features))
val_X = val_X.reshape((val_X.shape[0], n_in, num_features))
test_X = test_X.reshape((test_X.shape[0], n_in, num_features))

train_Y = train_Y.reshape((train_Y.shape[0], n_out, num_features))
val_Y = val_Y.reshape((val_Y.shape[0], n_out, num_features))
test_Y = test_Y.reshape((test_Y.shape[0], n_out, num_features))

# to tensor
train_X = torch.from_numpy(train_X).to(torch.float32)
train_Y = torch.from_numpy(train_Y).to(torch.float32)
val_X = torch.from_numpy(val_X).to(torch.float32)
val_Y = torch.from_numpy(val_Y).to(torch.float32)
test_X = torch.from_numpy(test_X).to(torch.float32)
test_Y = torch.from_numpy(test_Y).to(torch.float32)

# dataset
train_set = TensorDataset(train_X, train_Y)
val_set = TensorDataset(val_X, val_Y)
test_set = TensorDataset(test_X, test_Y)

# dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size, False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size, False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, False)

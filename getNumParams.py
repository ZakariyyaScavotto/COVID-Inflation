# Get the number of parameters in the models saved in InterpModels folder
# NNModel.h5, RNNModel.h5, NNRevModel.pt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import keras
from keras.models import load_model

# Load the models
NNModel = load_model('Models/NNModel.h5', compile=False)
RNNModel = load_model('Models/RNNModel.h5', compile=False)
NNRevModel = torch.load('Models/NNRevModel.pt')
LSTMModel = load_model('Models/LSTMModel.h5', compile=False)
GRUModel = load_model('Models/GRUModel.h5', compile=False)
# Get the number of parameters
NNModelNumParams = NNModel.count_params()
RNNModelNumParams = RNNModel.count_params()
LSTMModelNumParams = LSTMModel.count_params()
GRUModelNumParams = GRUModel.count_params()
# count the number of parameters in the NNRev model based on the model_state_dict
NNRevModelNumParams = sum(p.numel() for p in NNRevModel['model_state_dict'].values())

print('NNModelNumParams: ', NNModelNumParams)
print('RNNModelNumParams: ', RNNModelNumParams)
print('NNRevModelNumParams: ', NNRevModelNumParams)
print('LSTMModelNumParams: ', LSTMModelNumParams)
print('GRUModelNumParams: ', GRUModelNumParams)

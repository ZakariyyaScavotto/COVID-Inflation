import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
# import to hide the annoying warnings about numba.jit when importing shap library
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras import layers, metrics, Sequential, models, regularizers
from keras_tuner.tuners import RandomSearch
from sklearn.linear_model import Lasso
import shutil, os
import pickle
from functools import partial
import argparse
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from yellowbrick.regressor.residuals import residuals_plot
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import VotingRegressor

def main():
    # load in the NN model
    model = keras.models.load_model('Models/NNModel.h5')
    # load in the ensemble model
    ensembleModel = pickle.load(open('Models/ensembleModel.pickle', 'rb'))
    # Check to see if the neural network model has the same weights as the neural network in the ensemble model
    nnWeights = model.get_weights()
    ensembleNNWeights = ensembleModel.estimators_[2].model.get_weights()
    # check to see if the weights are the same
    for i in range(len(nnWeights)):
        if np.array_equal(nnWeights[i], ensembleNNWeights[i]):
            print('Same weights')
        else:
            print('Different weights')
    # Check to see if they have the same architecture
    print(model.summary())
    print(ensembleModel.estimators_[2].model.summary())


if __name__ == '__main__':
    main()
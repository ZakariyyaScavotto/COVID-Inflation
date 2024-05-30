import numpy as np
import pandas as pd
import tensorflow as tf 
import keras
from keras.api import metrics
from keras.api.layers import Dense, Dropout
from keras.api.models import Sequential
from walkForwardPrePost2020Dynamic import extendNN

# Create a simple model
model = Sequential()
model.add(Dense(19, input_shape=(19,), activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(0.55))
model.add(Dense(1, activation='sigmoid'))
# Print the model summary
print(model.summary())

# Extend the model
model = extendNN(model, 4)
# Print the model summary
print(model.summary())
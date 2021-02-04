import numpy as np
import tensorflow as tf
from tensorflow import keras

feature_num = 7
lstm_output_num = 10

layer_input = keras.Input(shape=(1,feature_num), name='input')
layer_LSTM = keras.layers.LSTM(lstm_output_num, return_sequences=True, name='LSTM')(layer_input)
layer_output = keras.layers.TimeDistributed(keras.layers.Dense(1), name='output')(layer_LSTM)

model = keras.Model(layer_input, layer_output)
print(model.summary())

model.compile(loss = 'mse', optimizer = 'adam')
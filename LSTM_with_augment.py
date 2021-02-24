import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut

from plot_with_PE_imputation import plot_with_PE_imputation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
from tensorflow import keras
import random as rn
random_state = 1004
np.random.seed(random_state)
rn.seed(random_state)
tf.random.set_seed(random_state)

from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.signal import medfilt

import time

#Load Data
data = pd.read_csv('./facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Store features and labels
# X = data[feature_names].values
# y = data['Facies'].values

# Store well labels and depths
wells = data['Well Name'].values
depth = data['Depth'].values

# Data processing (drop NA / drop F9 well)
DataImp_dropNA = data.dropna(axis = 0, inplace = False)
F9idx = DataImp_dropNA[DataImp_dropNA['Well Name'] == 'Recruit F9'].index
DataImp_dropF9 = DataImp_dropNA.drop(F9idx)
wells_noPE = DataImp_dropF9['Well Name'].values
DataImp = DataImp_dropF9.drop(['Formation', 'Well Name', 'Depth'], axis=1).copy()

Ximp=DataImp.loc[:, DataImp.columns != 'PE'].values
Yimp=DataImp.loc[:, 'PE'].values

from augment_features import augment_features
X_aug, padded_rows = augment_features(Ximp, wells_noPE,depth)
# Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_aug)
Ximp_scaled = scaler.transform(X_aug)

# split data
logo = LeaveOneGroupOut()
R2list = []
mselist = []

feature_num = Ximp_scaled.shape[1]

# GPU 할당 에러 해결
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# parameter
lstm_output_num = 20
batchsize = 10
dropout_rate = 0.0
learning_rate = 0.001


# 각 well에 대해 test

start = time.time()

for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
    well_name = wells_noPE[test[0]]
    print("----------------- 'Well name_test : ", well_name, "' -----------------")

    # Imputation using LSTM

    # Reshape the train, test data
    trainX = Ximp_scaled[train].reshape((Ximp_scaled[train].shape[0], 1, Ximp_scaled[train].shape[1]))
    trainY = Yimp[train]

    testX = Ximp_scaled[test].reshape((Ximp_scaled[test].shape[0], 1, Ximp_scaled[test].shape[1]))
    testY = Yimp[test]

    # Build the LSTM model
    layer_input = keras.Input(shape=(1, feature_num), name='input')
    layer_LSTM = keras.layers.LSTM(lstm_output_num, return_sequences=True, dropout=dropout_rate, name='LSTM')(layer_input)
    layer_output = keras.layers.TimeDistributed(keras.layers.Dense(1), name='output')(layer_LSTM)

    model = keras.Model(layer_input, layer_output)
    keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer='adam')

    # print(model.summary())

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

    # Train model and plot training process
    history = model.fit(trainX, trainY, epochs=150, batch_size=batchsize, validation_data=(testX, testY), verbose=0,
                        shuffle=False, callbacks=[early_stopping])

    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # prediction (R2, mse)
    Yimp_predicted = np.ravel(model.predict(testX))
    ## medfilt
    Yimp_predicted = medfilt(Yimp_predicted, kernel_size=5)

    R2 = r2_score(testY, Yimp_predicted)
    print("Well name_test : ", well_name)
    print("R2: %.4f" %R2)
    R2list.append(R2)

    mse = mean_squared_error(testY, Yimp_predicted)
    print("mse: %.4f" % mse)
    mselist.append(mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data["PE_pred"] = Yimp_predicted

    # plot_with_PE_imputation(predict_data, facies_colors,R2)

average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)

print("total time: %.1f" %(time.time() - start))
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
tf.compat.v1.random.set_random_seed(random_state)
np.random.seed(random_state)
rn.seed(random_state)
tf.random.set_seed(random_state)

from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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

# Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(Ximp)
Ximp_scaled = scaler.transform(Ximp)

# split data
logo = LeaveOneGroupOut()

feature_num = 7

# GPU 할당 에러 해결
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# parameter
# lstm_output_num = 25
# batchsize = 8
# dropout_rate = 0.0
# learning_rate = 0.001

LON_grid = [20,25,30]
BS_grid = [8,10,15]
LR_grid = [0.001]

param_grid = []
for LON in LON_grid:
    for BS in BS_grid:
        for LR in LR_grid:
            param_grid.append({'LON':LON, 'BS':BS, 'LR':LR})

# 각 well에 대해 test
mse_param = []
R2_param = []
df_by_param = pd.DataFrame(columns=["R2","MSE"])

start = time.time()

param_grid_num = len(param_grid)
i = 1

for param in param_grid:
    R2_split = []
    mse_split = []
    print(i,"of", param_grid_num, param, end=" ")
    traintimestart = time.time()

    for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
        well_name = wells_noPE[test[0]]

        # Imputation using LSTM

        # Reshape the train, test data
        trainX = Ximp_scaled[train].reshape((Ximp_scaled[train].shape[0],1,Ximp_scaled[train].shape[1]))
        trainY = Yimp[train]

        testX = Ximp_scaled[test].reshape((Ximp_scaled[test].shape[0],1,Ximp_scaled[test].shape[1]))
        testY = Yimp[test]

        # Build the LSTM model
        layer_input = keras.Input(shape=(1, feature_num), name='input')
        layer_LSTM = keras.layers.LSTM(param['LON'], return_sequences=True, name='LSTM')(layer_input)
        layer_output = keras.layers.TimeDistributed(keras.layers.Dense(1), name='output')(layer_LSTM)

        model = keras.Model(layer_input, layer_output)
        keras.optimizers.Adam(lr=param['LR'])
        model.compile(loss='mse', optimizer='adam')

        # print(model.summary())

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min',verbose=0)

        # Train model and plot training process
        history = model.fit(trainX, trainY, epochs=100, batch_size=param['BS'], validation_data=(testX, testY), verbose=0, shuffle=False, callbacks=[early_stopping])

        # prediction (R2, mse)
        Yimp_predicted = np.ravel(model.predict(testX))
        R2 = r2_score(testY, Yimp_predicted)
        # print("Well name_test : ", well_name)
        # print("R2: %.4f" %R2)

        mse = mean_squared_error(testY, Yimp_predicted)
        # print("mse: %.4f" % mse)

        # predict_data = data[data['Well Name'] == well_name].copy()
        # predict_data["PE_pred"] = Yimp_predicted

        R2_split.append(R2)
        mse_split.append(mse)

        # plot_with_PE_imputation(predict_data, facies_colors,R2)

    R2_param.append(np.mean(R2_split))
    mse_param.append(np.mean(mse_split))
    df_by_param.loc["LON%i/BS%i/LR%f"%(param['LON'],param['BS'],param['LR'])] = [np.mean(R2_split), np.mean(mse_split)]
    i += 1
    print("R2 : %.4f, mse : %.4f" %(np.mean(R2_split), np.mean(mse_split)), end=" ")
    print("param train time : %.1f" %(time.time()-traintimestart))

print(df_by_param)
df_by_param.to_excel("LSTM_grid.xlsx")

best_idx = np.argmin(mse_param)
best_idx2 = np.argmax(R2_param)
param_best = param_grid[best_idx]
param_best2 = param_grid[best_idx2]
mse_best = mse_param[best_idx]
R2_best = R2_param[best_idx2]
print('Best mse = %.4f %s' % (mse_best, param_best))
print('Best R2 = %.4f %s' % (R2_best, param_best2))

print("Gridsearch time : %.1f" %(time.time() - start))

R2list = []
mselist = []
df_by_well = pd.DataFrame(columns=["R2","MSE"])

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
    layer_LSTM = keras.layers.LSTM(param_best['LON'], return_sequences=True, name='LSTM')(layer_input)
    layer_output = keras.layers.TimeDistributed(keras.layers.Dense(1), name='output')(layer_LSTM)

    model = keras.Model(layer_input, layer_output)
    keras.optimizers.Adam(lr=param_best['LR'])
    model.compile(loss='mse', optimizer='adam')

    # print(model.summary())

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

    # Train model and plot training process
    history = model.fit(trainX, trainY, epochs=150, batch_size=param_best['BS'], validation_data=(testX, testY), verbose=0,
                        shuffle=False, callbacks=[early_stopping])

    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # prediction (R2, mse)
    Yimp_predicted = np.ravel(model.predict(testX))
    R2 = r2_score(testY, Yimp_predicted)
    # print("Well name_test : ", well_name)
    # print("R2: %.4f" %R2)
    R2list.append(R2)

    mse = mean_squared_error(testY, Yimp_predicted)
    # print("mse: %.4f" % mse)
    mselist.append(mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data["PE_pred"] = Yimp_predicted

    plot_with_PE_imputation(predict_data, facies_colors,R2)
    df_by_well.loc[well_name] = [R2, mse]

print(df_by_well)
df_by_well.to_excel("LSTM_eachwell.xlsx")

average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)

print("total time: %.1f" %(time.time() - start))
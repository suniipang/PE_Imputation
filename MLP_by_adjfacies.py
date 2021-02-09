import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

from plot_with_PE_imputation import plot_with_PE_imputation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# Imputation
DataImp_dropNA = data.dropna(axis = 0, inplace = False)
F9idx = DataImp_dropNA[DataImp_dropNA['Well Name'] == 'Recruit F9'].index
DataImp_dropF9 = DataImp_dropNA.drop(F9idx)
wells_noPE = DataImp_dropF9['Well Name'].values
DataImp = DataImp_dropF9.drop(['Formation', 'Well Name', 'Depth'], axis=1).copy()

Ximplist = []
Yimplist = []

Ximp=DataImp.loc[:, DataImp.columns != 'PE'].values
Yimp=DataImp.loc[:, 'PE'].values

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(Ximp)
Ximp_scaled = scaler.transform(Ximp)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

logo = LeaveOneGroupOut()
R2list = []
mselist = []

start = time.time()

random_state = 24
df_by_well = pd.DataFrame(columns=["R2","MSE"])

for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
    well_name = wells_noPE[test[0]]

    Xtrain = Ximp_scaled[train]
    Ytrain = Yimp[train]
    trainSSidx = Xtrain[:, 0] <= 0 ##scaling 후 facies 3이 0으로 됨
    trainLSidx = Xtrain[:, 0] > 0

    Xtest = Ximp_scaled[test]
    Ytest = Yimp[test]
    testSSidx = Xtest[:,0] <= 0
    testLSidx = Xtest[:,0] > 0

    ## generate two MLP model
    reg1 = MLPRegressor(hidden_layer_sizes=50, max_iter=1500)
    reg1.fit(Xtrain[trainSSidx],Ytrain[trainSSidx])
    reg2 = MLPRegressor(hidden_layer_sizes=50, max_iter=1500)
    reg2.fit(Xtrain[trainLSidx], Ytrain[trainLSidx])

    ## prediction
    Ypred = np.empty(Ytest.shape,float)
    Ypred[testSSidx] = reg1.predict(Xtest[testSSidx])
    Ypred[testLSidx] = reg2.predict(Xtest[testLSidx])

    R2 = r2_score(Ytest,Ypred)
    mse = mean_squared_error(Ytest,Ypred)

    print("Well name_test : ", well_name, end=" / ")
    print("R2: %.4f" %R2, end = " ")
    print("mse: %.4f" %mse)
    R2list.append(R2)
    mselist.append(mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data["PE_pred"] = Ypred
    df_by_well.loc[well_name] = [R2, mse]

    # plot_with_PE_imputation(predict_data, facies_colors,R2)
    ## 그림 저장하기

print(df_by_well)

average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)

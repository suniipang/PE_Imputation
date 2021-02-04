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

Ximp=DataImp.loc[:, DataImp.columns != 'PE'].values
Yimp=DataImp.loc[:, 'PE'].values

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(Ximp)
Ximp_scaled = scaler.transform(Ximp)

logo = LeaveOneGroupOut()

H_grid = [20,30,40,50,60]
I_grid = [2000,2500,3000,3500,4000]
param_grid = []
for H in H_grid:
    for I in I_grid:
        param_grid.append({'Hiddenlayersizes':H, 'MaxIter':I})
from sklearn.metrics import mean_squared_error


mse_param = []
# score_param = []
for param in param_grid:
    # score_split = []
    mse_split = []
    print(param)
    for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
        well_name = wells_noPE[test[0]]

        # Imputation using linear regression
        reg = MLPRegressor(hidden_layer_sizes=param['Hiddenlayersizes'], max_iter=param['MaxIter'])
        reg.fit(Ximp_scaled[train], Yimp[train])

        # score = reg.score(Ximp_scaled[test],Yimp[test]) # R2
        print("Well name_test : ", well_name)
        # print("acc : %.3f" % score)
        Yimp_predicted = reg.predict(Ximp_scaled[test])
        mse = mean_squared_error(Yimp[test],Yimp_predicted)
        print("mse : %.3f" % mse)

        # score_split.append(score)
        mse_split.append(mse)

    # score_param.append(np.mean(score_split))
    mse_param.append(np.mean(mse_split))

best_idx = np.argmin(mse_param)
param_best = param_grid[best_idx]
mse_best = mse_param[best_idx]
print('\nBest mse = %.3f %s' % (mse_best, param_best))

# best_idx = np.argmax(score_param)
# param_best = param_grid[best_idx]
# score_best = score_param[best_idx]
# print('\nBest score = %.3f %s' % (score_best, param_best))

mselist = []

for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
    well_name = wells_noPE[test[0]]

    # Imputation using MLP
    reg = MLPRegressor(hidden_layer_sizes=param_best['Hiddenlayersizes'], max_iter=param_best['MaxIter'])
    reg.fit(Ximp_scaled[train], Yimp[train])

    # score = reg.score(Ximp_scaled[test], Yimp[test])  # R2
    # print("Well name_test : ", well_name)
    # print("acc : %.3f" % score)
    # acclist.append(score)

    Yimp_predicted = reg.predict(Ximp_scaled[test])
    mse = mean_squared_error(Yimp[test], Yimp_predicted)
    mselist.append(mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data["PE_pred"] = Yimp_predicted

    plot_with_PE_imputation(predict_data, facies_colors,mse)

average_mse = np.mean(mselist)
# print(*acclist)
print("Average MSE : ", average_mse)
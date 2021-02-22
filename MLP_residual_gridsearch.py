import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
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

Ximp=DataImp.loc[:, DataImp.columns != 'PE'].values
Yimp=DataImp.loc[:, 'PE'].values

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(Ximp)
Ximp_scaled = scaler.transform(Ximp)

logo = LeaveOneGroupOut()

H_grid = [10,20,30,40,50,60,70,80,90,100]
I_grid = [1000] ## 1000번이면 수렴
param_grid = []
for H in H_grid:
    for I in I_grid:
        param_grid.append({'Hiddenlayersizes':H, 'MaxIter':I})

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mse_param = []
R2_param = []
df_by_param = pd.DataFrame(columns=["R2","MSE"])

start = time.time()

param_grid_num = len(param_grid)
i = 1
random_state = 24

for param in param_grid:
    R2_split = []
    mse_split = []
    print(i,"of", param_grid_num, param, end=" ")
    traintimestart = time.time()

    for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
        well_name = wells_noPE[test[0]]

        # predict residual by each facies
        facies_scaled = np.unique(Ximp_scaled[:, 0])
        mean_PE = []
        Ytrain = np.zeros(Yimp[train].shape)
        for f in range(9):
            fidx = (Ximp_scaled[train][:, 0] == facies_scaled[f])
            mean_f = Yimp[train][fidx].mean()
            mean_PE.append(mean_f)
            Ytrain[fidx] = Yimp[train][fidx] - mean_f

        # Imputation using MLP
        reg = MLPRegressor(hidden_layer_sizes=param['Hiddenlayersizes'], max_iter=param['MaxIter'],random_state=random_state)
        reg.fit(Ximp_scaled[train][:,1:], Ytrain)

        Y_residual_pred = reg.predict(Ximp_scaled[test][:, 1:])
        Yimp_predicted = np.zeros(Yimp[test].shape)
        for f in range(9):
            fidx = (Ximp_scaled[test][:, 0] == facies_scaled[f])
            Yimp_predicted[fidx] = Y_residual_pred[fidx] + mean_PE[f]

            ## facies별로 정확도 한번 확인해보자
            # if np.any(fidx) == True:
            #     R2_f = r2_score(Yimp[test][fidx], Yimp_predicted[fidx])
            #     print("facies %i R2 : %.4f" % (f + 1, R2_f))

        R2 = r2_score(Yimp[test], Yimp_predicted)  ##R2
        mse = mean_squared_error(Yimp[test], Yimp_predicted)  ##MSE
        # print("Well name_test : ", well_name)
        # print("R2: %.4f" % R2)
        # print("mse: %.4f" % mse)

        R2_split.append(R2)
        mse_split.append(mse)

    R2_param.append(np.mean(R2_split))
    mse_param.append(np.mean(mse_split))
    df_by_param.loc["H%i/I%i"%(param['Hiddenlayersizes'],param['MaxIter'])] = [np.mean(R2_split), np.mean(mse_split)]
    i += 1
    print("R2 : %.4f, mse : %.4f" %(np.mean(R2_split), np.mean(mse_split)), end=" ")
    print("param train time : %.1f" %(time.time()-traintimestart))


print(df_by_param)

best_idx = np.argmin(mse_param)
param_best = param_grid[best_idx]
mse_best = mse_param[best_idx]
print('Best mse = %.4f %s' % (mse_best, param_best))

best_idx2 = np.argmax(R2_param)
param_best2 = param_grid[best_idx2]
R2_best = R2_param[best_idx2]
print('Best R2 = %.4f %s' % (R2_best, param_best2))

print("Gridsearch time : %.1f" %(time.time() - start))

mselist = []
R2list = []
df_by_well = pd.DataFrame(columns=["R2","MSE"])

for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
    well_name = wells_noPE[test[0]]
    print("Well name_test : ", well_name)

    # predict residual by each facies
    facies_scaled = np.unique(Ximp_scaled[:, 0])
    mean_PE = []
    Ytrain = np.zeros(Yimp[train].shape)
    for f in range(9):
        fidx = (Ximp_scaled[train][:, 0] == facies_scaled[f])
        mean_f = Yimp[train][fidx].mean()
        mean_PE.append(mean_f)
        Ytrain[fidx] = Yimp[train][fidx] - mean_f

    # Imputation using MLP
    reg = MLPRegressor(hidden_layer_sizes=param_best['Hiddenlayersizes'], max_iter=param_best['MaxIter'],random_state=random_state)
    reg.fit(Ximp_scaled[train][:, 1:], Ytrain)

    Y_residual_pred = reg.predict(Ximp_scaled[test][:, 1:])
    Yimp_predicted = np.zeros(Yimp[test].shape)
    for f in range(9):
        fidx = (Ximp_scaled[test][:, 0] == facies_scaled[f])
        Yimp_predicted[fidx] = Y_residual_pred[fidx] + mean_PE[f]

        ## facies별로 정확도 한번 확인해보자
        if np.any(fidx) == True:
            R2_f = r2_score(Yimp[test][fidx], Yimp_predicted[fidx])
            print("facies %i R2 : %.4f" % (f + 1, R2_f))

    R2 = r2_score(Yimp[test], Yimp_predicted)  ##R2
    R2list.append(R2)

    mse = mean_squared_error(Yimp[test], Yimp_predicted)  ##MSE
    mselist.append(mse)

    print("R2: %.4f" % R2)
    print("mse: %.4f" % mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data["PE_pred"] = Yimp_predicted

    plot_with_PE_imputation(predict_data, facies_colors,R2)
    df_by_well.loc[well_name] = [R2, mse]

print(df_by_well)

average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)

print("total time: %.1f" %(time.time() - start))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
# depth = data['Depth'].values

# Imputation
DataImp_dropNA = data.dropna(axis = 0, inplace = False)
F9idx = DataImp_dropNA[DataImp_dropNA['Well Name'] == 'Recruit F9'].index
DataImp_dropF9 = DataImp_dropNA.drop(F9idx)

wells_noPE = DataImp_dropF9['Well Name'].values
depth = DataImp_dropF9['Depth'].values

DataImp = DataImp_dropF9.drop(['Formation', 'Well Name', 'Depth'], axis=1).copy()

Ximp=DataImp.loc[:, DataImp.columns != 'PE'].values
Yimp=DataImp.loc[:, 'PE'].values

from augment_features import augment_features
X_aug, padded_rows = augment_features(Ximp, wells_noPE,depth)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_aug)
Ximp_scaled = scaler.transform(X_aug)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
logo = LeaveOneGroupOut()
R2list = []
mselist = []

start = time.time()

for train, test in logo.split(Ximp_scaled, Yimp, groups=wells_noPE):
    well_name = wells_noPE[test[0]]

    # delete padding
    train = np.setdiff1d(train, padded_rows)
    test = np.setdiff1d(test, padded_rows)

    # Imputation using linear regression
    linear_model = LinearRegression()
    linear_model.fit(Ximp_scaled[train], Yimp[train])

    # R2 = linear_model.score(Ximp_scaled[test],Yimp[test]) # R2
    # R2list.append(R2)

    Yimp_predicted = linear_model.predict(Ximp_scaled[test])
    R2 = r2_score(Yimp[test],Yimp_predicted) ##R2
    mse = mean_squared_error(Yimp[test],Yimp_predicted) ##MSE
    R2list.append(R2)
    mselist.append(mse)

    print("Well name_test : ", well_name, end=" ")
    print("R2: %.4f" %R2, end=" ")
    print("mse: %.4f" % mse)

    predict_data = data[data['Well Name'] == well_name].copy()
    predict_data = predict_data.drop([predict_data.index[0],predict_data.index[-1]])
    predict_data["PE_pred"] = Yimp_predicted

    plot_with_PE_imputation(predict_data, facies_colors,R2)
    ## 그림 저장하기



average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

Ximplist = []
Yimplist = []

##Facies 별로 데이터 분할
for i in range(9):
    Ximp = DataImp.loc[DataImp["Facies"] == i+1, DataImp.columns != 'PE'].values
    Yimp = DataImp.loc[DataImp["Facies"] == i+1, "PE"].values
    Ximplist.append(Ximp)
    Yimplist.append(Yimp)
    print("number of facies {} : {}".format(i+1, Ximp.shape[0]))


# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# scaler.fit(Ximp)
# Ximp_scaled = scaler.transform(Ximp)

from sklearn.metrics import mean_squared_error
R2list = []
mselist = []

for i in range(9):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Ximplist[i],Yimplist[i],test_size=0.2, shuffle=True)
    print("Facies {}...".format(i+1))

    # Imputation using linear regression
    linear_model = LinearRegression()
    linear_model.fit(Xtrain, Ytrain)

    R2 = linear_model.score(Xtest,Ytest) # R2
    print("R2: %.4f" %R2)
    R2list.append(R2)

    Yimp_predicted = linear_model.predict(Xtest)

    mse = mean_squared_error(Yimp_predicted,Ytest) ##MSE
    print("mse: %.4f" %mse)
    mselist.append(mse)


average_R2 = np.mean(np.array(R2list))
average_mse = np.mean(np.array(mselist))
print("average R2 : %.4f " % average_R2)
print("average MSE : %.4f " % average_mse)
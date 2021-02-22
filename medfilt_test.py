from scipy.signal import medfilt
import numpy as np

my_array = np.array([1000,10,40,4,5,6,10,2,5,4,63,5,36,5])

filt = medfilt(my_array,kernel_size=3)

for i in range(len(my_array)):
    print(my_array[i],filt[i])
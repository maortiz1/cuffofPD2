import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
from scipy.signal import find_peaks

cwd = os.getcwd()

#Get the names of the .csv file
path_samples_csv= os.path.join(cwd,'mat_to_csv')
files_csv = [f for f in listdir(path_samples_csv) if isfile(join(path_samples_csv, f))]
#read the files
os.chdir(path_samples_csv)

fr = 125

data_train_name = []
data_train_PPG  = np.empty(1)
data_train_ABP  = np.empty(1)
data_train_ECG  = np.empty(1)

data_test_name = []
data_test_PPG  = np.empty(1)
data_test_ABP  = np.empty(1)
data_test_ECG  = np.empty(1)

def fun_Pan_Tompkins(t, ECG):
	dx = (ECG[2:] - ECG[0:-2])/(t[2:]-t[0:-2])
	dx = dx * dx
	w_movil = np.zeros(len(dx))
	w_size = 20
	for i in range(len(dx)-w_size):
		w_movil[i]=np.sum(dx[i:i+w_size])
	peaks = find_peaks(w_movil, width = w_size, height= np.var(w_movil)**0.5)
	t = t[peaks[0]]
	return t

for i in range(1,2):
    #print(files_csv[i])
    Matrix_data = np.genfromtxt(files_csv[i], delimiter=",")
    data_train_name =files_csv[i]
    data_train_PPG = (Matrix_data[0,:])
    data_train_ABP = (Matrix_data[1,:])
    data_train_ECG = (Matrix_data[2,:])
    t_train = np.arange(len(data_train_ECG))/fr
    t_RRtrain = fun_Pan_Tompkins(t_train, data_train_ECG)       

for i in range(2,3):
    #print(files_csv[i])
    Matrix_data = np.genfromtxt(files_csv[i], delimiter=",")
    data_test_name = files_csv[i]
    data_test_PPG = (Matrix_data[0,:])
    data_test_ABP = (Matrix_data[1,:])
    data_test_ECG = (Matrix_data[2,:])
    t_test = np.arange(len(data_test_ECG))/fr
    t_RRtest = fun_Pan_Tompkins(t_test, data_test_ECG)

plt.figure()
plt.plot(t_train, data_train_ECG)
plt.title('ECG train %s'%data_train_name)
plt.show()

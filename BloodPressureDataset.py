import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
from scipy.signal import find_peaks
import pickle

##############################################################
cwd = os.getcwd()
path_samples_csv= os.path.join(cwd,'mat_to_csv')
files_csv = [f for f in listdir(path_samples_csv) if isfile(join(path_samples_csv, f))]
os.chdir(path_samples_csv)
#############################################################
fr = 125

t_RRtrain = []

t_RRtest = []
######################################################################
def fun_Pan_Tompkins(t, ECG):
    """ Algoritmo de Pan-Tompkins para hallar RR
    """
    dx = (ECG[2:] - ECG[0:-2])/(t[2:]-t[0:-2])
    dx = dx * dx
    w_movil = np.zeros(len(dx))
    w_size = 20

    for i in range(len(dx)-w_size):
	    w_movil[i]=np.sum(dx[i:i+w_size])

    peaks = find_peaks(w_movil, width = w_size, height= np.var(w_movil)**0.5)
    t = t[peaks[0]]

    return t
#########################################################################
for i in range(6000):
	print(i)
	data_train_PPG, data_train_ABP, data_train_ECG = np.genfromtxt(files_csv[i], delimiter=",")
	print(data_train_PPG[0],data_train_ABP[0], data_train_ECG[0])
	print(files_csv[i])
	t_train = np.arange(len(data_train_ECG))/fr
	t_RRtrain.append(fun_Pan_Tompkins(t_train, data_train_ECG))       

for i in range(6000,9000):
	print(i)
	data_test_PPG, data_test_ABP, data_test_ECG = np.genfromtxt(files_csv[i], delimiter=",")
	t_test = np.arange(len(data_test_ECG))/fr
	t_RRtest.append(fun_Pan_Tompkins(t_test, data_test_ECG))
	

plt.figure()
plt.plot(t_train, data_train_PPG)
plt.title('PPG train')
plt.show()
#####################################################################

with open('RR_train.mat', 'wb') as fp:
    pickle.dump(t_RRtrain, fp)

with open('RR_test.mat', 'wb') as fp:
    pickle.dump(t_RRtest, fp)

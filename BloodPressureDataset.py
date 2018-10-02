#Arrays
import numpy as np
#Directories
import os
from os import listdir
from os.path import isfile, join
#Plot
import matplotlib.pylab as plt
#Find_peaks
from scipy.signal import find_peaks
#Save variables
import pickle
#Biosignals
import neurokit as nk

##############################################################
cwd = os.getcwd()
path_samples_csv= os.path.join(cwd,'mat_to_csv')
files_csv = [f for f in listdir(path_samples_csv) if isfile(join(path_samples_csv, f))]
os.chdir(path_samples_csv)
#############################################################
fr = 125

t_RR = []
t_PPT = []

#########################################################################
numero = 6

for i in range(numero,numero+1):
    print(i)
    data_PPG, data_ABP, data_ECG = np.genfromtxt(files_csv[i], delimiter=",")
    idx_peaksECG = nk.bio_process(ecg = data_ECG, sampling_rate=125)['ECG']['R_Peaks']
    idx_peaksPPG = []
    idx_peaksDBP = []
    idx_peaksSBP = []
    
    for i in range(len(idx_peaksECG)-1):
        ran_0 = idx_peaksECG[i]
        ran_1 = idx_peaksECG[i+1]
        if not find_peaks(data_PPG[ran_0:ran_1], distance = ran_1-ran_0)[0]:
            idx_peaksPPG.append(int(0))
        else:
            idx_peaksPPG.append(int(find_peaks(data_PPG[ran_0:ran_1], distance = ran_1-ran_0)[0] + ran_0))

        idx_peaksDBP.append(int(np.argmax(data_ABP[ran_0:ran_1]) + ran_0))
        idx_peaksSBP.append(int(np.argmin(data_ABP[ran_0:ran_1]) + ran_0))

    t = (np.arange(len(data_ECG))/fr)
    
plt.figure()
plt.subplot(311)
plt.plot(t, data_PPG)
plt.scatter(t[idx_peaksPPG], data_PPG[idx_peaksPPG],c = 'g')

plt.subplot(312)
plt.plot(t,data_ECG)
plt.scatter(t[idx_peaksECG],data_ECG[idx_peaksECG], c ='r')

plt.subplot(313)
plt.plot(t,data_ABP)
plt.scatter(t[idx_peaksDBP], data_ABP[idx_peaksDBP], c = 'y')
plt.scatter(t[idx_peaksSBP], data_ABP[idx_peaksSBP], c = 'k')

#####################################################################

path_RR = os.path.join(cwd,'RR.pkl')
path_PPT= os.path.join(cwd,'PPT.pkl')

# Saving the objects:
with open(path_RR, 'wb') as f:  
    pickle.dump(t_RR, f)
    
with open(path_PPT, 'wb') as f:  
    pickle.dump(t_PPT, f)




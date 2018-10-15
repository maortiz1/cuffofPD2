#Arrays
import numpy as np
#Directories
import os
from os import listdir
from os.path import isfile, join
#Plot
import matplotlib.pylab as plt
#Scale
from sklearn.preprocessing import scale
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

t_HR  = []
t_PPT = []
SBP   = []
DBP   = []

#########################################################################
for i in range(500):
    data_PPG, data_ABP, data_ECG = np.genfromtxt(files_csv[i], delimiter=",")
    print(i)
    if len(data_PPG)>5*60*fr:      
        data_ECG = scale(data_ECG)
        if np.min(data_ECG)<-3 and np.max(data_ECG)<3:
            data_ECG = data_ECG            
        t = (np.arange(len(data_ECG))/fr)
        idx_peaksECG = nk.bio_process(ecg = data_ECG, sampling_rate=125)['ECG']['R_Peaks']
        t_RR = t[idx_peaksECG]
        HR = 60/np.diff(t_RR)
        idx_peaksPPG = []
        idx_peaksDBP = []
        idx_peaksSBP = []
        idx_del = []        
        for i in range(len(idx_peaksECG)-1):
            
            ran_0 = idx_peaksECG[i]
            ran_1 = idx_peaksECG[i+1]
            
            if find_peaks(data_PPG[ran_0:ran_1], distance = ran_1-ran_0)[0].size>0:
                idx_peaksPPG.append(int(find_peaks(data_PPG[ran_0:ran_1], distance = ran_1-ran_0 )[0] + ran_0))
                idx_peaksDBP.append(int(np.argmax(data_ABP[ran_0:ran_1]) + ran_0))
                idx_peaksSBP.append(int(np.argmin(data_ABP[ran_0:ran_1]) + ran_0))
            else:
                idx_del.append(i)
    
        HR = np.delete(HR, np.array(idx_del))
        t_RR1 =  np.delete(t_RR, np.array(idx_del)+1)
        t_RR2 =  np.delete(t_RR, np.array(idx_del))
        t_PPG = t[idx_peaksPPG]         
        
        if np.var(data_ABP[np.array(idx_peaksSBP)])<15**2 and np.var(data_ABP[np.array(idx_peaksDBP)])<15**2:
            
            PPT1 = (t_PPG-t_RR2[:-1])
            PPT2 = (t_RR1[1: ]-t_PPG)
            if np.mean(PPT1)>np.mean(PPT2):
                t_PPT.append(PPT1)
            else:
                t_PPT.append(PPT2)
#            plt.figure()
#            plt.plot(PPT1,'gx',label='PPG peak - previous RR peak ')
#            plt.plot(PPT2,'rx',label='subsequent RR peak - PPG peak')
#            plt.legend()
              
                
            t_HR.append(HR)    
            SBP.append(data_ABP[np.array(idx_peaksSBP)])
            DBP.append(data_ABP[np.array(idx_peaksDBP)])
#                
#        plt.figure()
#            
#        plt.subplot(311)
#        plt.plot(t, data_PPG)
#        plt.scatter(t[idx_peaksPPG], data_PPG[idx_peaksPPG],c = 'g')
#        
#        plt.subplot(312)
#        plt.plot(t,data_ECG)
#        plt.scatter(t[idx_peaksECG],data_ECG[idx_peaksECG], c ='r')
#        
#        plt.subplot(313)
#        plt.plot(t,data_ABP)
#        plt.scatter(t[idx_peaksDBP], data_ABP[idx_peaksDBP], c = 'y')
#        plt.scatter(t[idx_peaksSBP], data_ABP[idx_peaksSBP], c = 'k')

#####################################################################
os.chdir(cwd)

path_HR = os.path.join(cwd,'HR.pkl')
path_PPT= os.path.join(cwd,'PPT.pkl')
path_SBP= os.path.join(cwd,'SBP.pkl')
path_DBP= os.path.join(cwd,'DBP.pkl')

# Saving the objects:
with open(path_HR, 'wb') as f:  
    pickle.dump(t_HR, f)
    
with open(path_PPT, 'wb') as f:  
    pickle.dump(t_PPT, f)
    
with open(path_SBP, 'wb') as f:  
    pickle.dump(SBP, f)
    
with open(path_DBP, 'wb') as f:  
    pickle.dump(DBP, f)




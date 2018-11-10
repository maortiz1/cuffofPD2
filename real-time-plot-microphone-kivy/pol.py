# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:15:45 2018

@author: m_ana
"""
import scipy
from scipy.signal import filtfilt, kaiserord, lfilter, firwin, freqz
from sklearn.preprocessing import scale
#Biosignals
import neurokit as nk

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

#Leer datos (numeros)
datos_n = np.genfromtxt('Copia de Datos real.txt')
#Leer datos (texto)
datos_t = np.genfromtxt('Copia de Datos real.txt', dtype = str)

nombres = datos_t[::9,0]
t_HR  = []
t_PPT = []

for i in range(nombres.shape[0]):
    #Tener los datos de cada uno de los .txt
    t_HR_cu  = []
    t_PPT_cu = []
    idx, DBP, SBP = datos_n[2+i*9:9+i*9,0], datos_n[2+i*9:9+i*9,1], datos_n[2+i*9:9+i*9,2]
    
    idx = np.cumsum(np.array(idx[-1:0:-1], dtype = int))*250
    ECG = np.genfromtxt('%s_ecg.txt'%nombres[i], delimiter = ',')
    ECG = ECG[:-1]
    ECG = ECG[-1:0:-1]
    PPG = np.genfromtxt('%s_ppg.txt'%nombres[i], delimiter = ',')
    PPG = PPG[:-1]
    PPG = PPG[-1:0:-1]   
    
    idx_a = idx[0]
    
    b_ecg, a_ecg = scipy.signal.butter(1,[0.04,0.92],'bandpass')
    b_ppg, a_ppg = scipy.signal.butter(1,[0.01,0.04],'bandpass')
    
    for j in idx[1:]:
        idx_d = j 
        
        ECG_aux = ECG[idx_a:idx_d]
        PPG_aux = PPG[idx_a:idx_d]
        
        ECG = filtfilt(b_ecg, a_ecg, ECG_aux, method='gust')
        
        PPG = filtfilt(b_ppg, a_ppg, PPG_aux, method='gust')
        t = (np.arange(len(ECG_aux))/250)
        try:
            idx_peaksECG = nk.bio_process(ecg = ECG_aux, sampling_rate=250)['ECG']['R_Peaks']        
            t_RR = t[idx_peaksECG]
        
            HR = 60/np.diff(t_RR)
            
            idx_peaksPPG = []
            idx_del = []   
            
            for i in range(len(idx_peaksECG)-1):
                
                ran_0 = idx_peaksECG[i] 
                ran_1 = idx_peaksECG[i+1] 
                
                if find_peaks(PPG_aux[ran_0:ran_1], distance = ran_1-ran_0)[0].size>0:
                    idx_peaksPPG.append(int(find_peaks(PPG_aux[ran_0:ran_1], distance = ran_1-ran_0 )[0] + ran_0))
                else:
                    idx_del.append(i)
        
            HR    = np.delete(HR, np.array(idx_del))
            t_RR1 =  np.delete(t_RR, np.array(idx_del)+1)
            t_RR2 =  np.delete(t_RR, np.array(idx_del))
            t_PPG = t[idx_peaksPPG]         
            
#            plt.figure()
#            plt.plot(t,ECG_aux)
#            plt.scatter(t_RR,ECG_aux[idx_peaksECG], c='r')
#            plt.figure()
#            plt.plot(t,PPG_aux)
#            plt.scatter(t_PPG,PPG_aux[idx_peaksPPG], c='r')
            
            PPT1 = (t_PPG-t_RR2[:-1])
            PPT2 = (t_RR1[1: ]-t_PPG)
            try:
                t_HR_cu.append(np.mean(HR))
                if np.mean(PPT1)>np.mean(PPT2):
                    t_PPT_cu.append(np.mean(PPT1))
                else:
                    t_PPT_cu.append(np.mean(PPT2))
    #            plt.figure()
    #            plt.plot(PPT1,'gx',label='PPG peak - previous RR peak ')
    #            plt.plot(PPT2,'rx',label='subsequent RR peak - PPG peak')
    #            plt.legend()
            except:
                t_HR_cu.append([])
                t_PPT_cu.append([])
                
        except:
            t_HR_cu.append([])
            t_PPT_cu.append([])
            
        idx_a = j
    t_HR.append(t_HR_cu)
    t_PPT.append(t_PPT_cu)



#datos = np.genfromtxt('https://raw.githubusercontent.com/maortiz1/cuffofPD2/master/real-time-plot-microphone-kivy/11-9-11-17_ecg.txt',delimiter=',')
#plt.figure()
#plt.plot(datos)
##plt.figure()
##plt.plot(datos[:,1])
#
#b, a = scipy.signal.butter(1,[0.04,0.92],'bandpass')
#x = datos[:-1]
## ecg
#y = filtfilt(b, a, x, method='gust')
#
## b, a = scipy.signal.butter(1,[0.01,0.04],'bandpass')
## x = datos[:-1]
## # ppg
## y = filtfilt(b, a, x, method='gust')
##
## # b, a = scipy.signal.butter(1,0.01,'highpass')
## y = filtfilt(y, a, x, method='gust')
#plt.figure()
#plt.plot(x,'r')
#plt.plot(y,'g')
#
#
#fs = 250
#data_ECG = scale(x)
#t = (np.arange(len(data_ECG))/fs)
#idx_peaksECG = nk.bio_process(ecg = data_ECG, sampling_rate=125)['ECG']['R_Peaks']
#t_RR = t[idx_peaksECG]
#
#plt.figure()
#
#plt.plot(t,data_ECG)
#plt.scatter(t[idx_peaksECG],data_ECG[idx_peaksECG], c ='r')
#
#plt.show()

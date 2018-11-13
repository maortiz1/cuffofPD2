# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:19:56 2018

@author: m_ana
"""

#Arrays
import numpy as np
#Directories
#import os
#from os import listdir
#from os.path import isfile, join
#Plot
import matplotlib.pylab as plt
import seaborn as sns
#Scale
from sklearn.preprocessing import scale
#Find_peaks
import scipy
from sklearn import linear_model

#ML
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import mean_squared_error
from math import sqrt

#Save variables
#import pickle
#Biosignals
import neurokit as nk

#Filtros
from scipy.signal import find_peaks, filtfilt
from statsmodels import robust
import pywt
class filter():
    def __init__(self):
        self.becg, self.aecg = scipy.signal.butter(1,[0.08,0.72],'bandpass')
        self.bllnotch, self.allnotch = scipy.signal.iirnotch(0.48,30)
        self.bppg,self.appg= scipy.signal.butter(1,[0.01,0.04],'bandpass')
    def filtrar(self,dataecg,datappg):
        ecg=filtfilt(self.bllnotch, self.allnotch, dataecg, method='gust')
        ppg=filtfilt(self.bllnotch, self.allnotch, datappg, method='gust')
        datafecg=filtfilt(self.becg,self.aecg,ecg,method='gust')
        datafecg=self.waveletFilt(datafecg,"db4",1)
        datafppg=filtfilt(self.bppg,self.appg,ppg,method='gust')
        return datafecg,datafppg
    def hampelFilter(self,data,win,t0,s):
        Th = 1.4826
        eMed = -0.105638066
        pMed = 49.91691522

        if s == "ecg" :
            rMedian=np.median(data)
            diff=np.abs(rMedian-eMed)
            absMedianStd=scipy.signal.medfilt(diff,win)
            th= t0*Th*absMedianStd
            indOutlier=diff>th
            data[indOutlier]=0
        else:
            rMedian=np.median(data)
            diff=np.abs(data-rMedian)
            absMedianStd=scipy.signal.medfilt(diff,win)
            th= t0*Th*absMedianStd
            indOutlier=diff>th
            data[indOutlier]=0
        
        return(data)        
    def waveletFilt(self, x, wavelet, level):
        coeff = pywt.wavedec( x, wavelet, mode="per" )
        sigma = robust.mad( coeff[-level] )
        uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
        coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
        y = pywt.waverec( coeff, wavelet, mode="per" )
        return(y)


## leer los datos
## Tiempo y valores presión
#Leer datos (numeros)
datos_n = np.genfromtxt('Copia de Datos real.txt')
#Leer datos (texto)
datos_t = np.genfromtxt('Copia de Datos real.txt', dtype = str)
fil=filter()
##crear datos presion 
nombres = datos_t[::9,0]
i=1
for k in range(nombres.shape[0]):
    actual=k*9;
    dat = np.delete(datos_n[actual:i*9],(0),axis=0)
    temp=dat[::,0]
    sis=dat[::,1]
    dia=dat[::,2]
    i+=1
    ##tiempo total en segundos
    total=np.cumsum(temp)[-1]
    #numero de datos esperados
    numdat=round(total*250)
    #idx 
    idx=(np.round(np.cumsum(temp)*250))
    
    newsis=np.zeros(int(idx[-1]))
    newdia=np.zeros(int(idx[-1]))
    for d in range(1,idx.shape[0]):
        actual=int(idx[d-1])
        sig=int(idx[d])
        newsis[actual]=sis[d-1]
        newsis[actual+1:sig]=sis[d]
        newdia[actual]=dia[d-1]
        newdia[actual+1:sig]=dia[d]
#se guardan los archivos con los valores de diastole y sistole
#    file = open('%s_dia.txt'%(nombres[k]),'a+')
#    file2=open('%s_sis.txt'%(nombres[k]),'a+')
#    for data in newdia:
#            file.write(str(data) + ',')	
#    for data in newsis:
#            file2.write(str(data) + ',')	
#
#    file.close()
#    file2.close() 
    #recortar ecg
    ECG= np.genfromtxt('%s_ecg.txt'%nombres[k], delimiter = ',')
    lastvalue=ECG.shape[0]-2500
    first=lastvalue-int(idx[-1])
    ECGcut=ECG[first:lastvalue]
    
    #recortar ppg
    PPG= np.genfromtxt('%s_ppg.txt'%nombres[k], delimiter = ',')
    lastvalue=PPG.shape[0]-2500
    first=lastvalue-int(idx[-1])
    PPGcut=PPG[first:lastvalue]
    fecgcut,fppgcut=fil.filtrar(ECGcut,PPGcut)
    #obtener picos RR
    idx_peaksECG = nk.bio_process(ecg = ECGcut, sampling_rate=250)['ECG']['R_Peaks']
    
    
    
    
    
    
    
    
    
        
        
    
    
    
    
        
    












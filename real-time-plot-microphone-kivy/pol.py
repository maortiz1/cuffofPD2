# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:15:45 2018

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

b, a = scipy.signal.iirnotch(0.48,30)

#Hampel Filter for Outliers
def hampelFilter(data,win,t0,s):
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
    
    
def waveletFilt( x, wavelet, level):
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    sigma = robust.mad( coeff[-level] )
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    y = pywt.waverec( coeff, wavelet, mode="per" )
    return(y)

#Leer datos (numeros)
datos_n = np.genfromtxt('Copia de Datos real.txt')
#Leer datos (texto)
datos_t = np.genfromtxt('Copia de Datos real.txt', dtype = str)

nombres = datos_t[::9,0]
nombres_malos = []
t_HR  = []

t_PPT = []
t_DBP = []
t_SBP = []

errorSBP = []
errorDBP = []
coefSBP = []
coefDBP = []

cortar = [150000, 50000, 22000, 105000, 115000, 46000, 64000, 43000, 60000, 70000, 80000, 88000, 72000, 5500, 22000, 46000, 13000, 29000, 65000]


for k in range(nombres.shape[0]):

    #Tener los datos de cada uno de los .txt
    idx, DBP, SBP = datos_n[2+k*9:9+k*9,0], datos_n[2+k*9:9+k*9,1], datos_n[2+k*9:9+k*9,2]

    idx = np.cumsum(np.array(idx[::-1], dtype = int))*250

    ECG = np.genfromtxt('%s_ecg.txt'%nombres[k], delimiter = ',')

    ECG = ECG[cortar[k]:-1]
    
#    plt.subplot(212)
    PPG = np.genfromtxt('%s_ppg.txt'%nombres[k], delimiter = ',')

    PPG = PPG[cortar[k]:-1]

    ECG = filtfilt(b, a, ECG, method='gust')
    b_ecg, a_ecg = scipy.signal.butter(1,[0.08,0.72],'bandpass')
    ECG = filtfilt(b_ecg, a_ecg, ECG, method='gust')
    ECG = waveletFilt(ECG,"db4",1)
    ECG = scale(ECG[1:-1])
    
    PPG = filtfilt(b, a, PPG, method='gust')
    b_ppg, a_ppg = scipy.signal.butter(1,[0.01,0.04],'bandpass')
    PPG = filtfilt(b_ppg, a_ppg, PPG, method='gust')
    PPG = scale(PPG[1:-1])
#    
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(ECG)
#    plt.subplot(212)
#    plt.plot(PPG)


    idx_a = 0
    cont = 0
    t_HR_cu=[]
    t_PPT_cu=[]
    t_DBP_cu=[]
    t_SBP_cu=[]
    for j in idx:       
        


        idx_d = j

        ECG_aux = ECG[idx_a:idx_d]
        PPG_aux = PPG[idx_a:idx_d]
        
#        ECG_aux = filtfilt(b_ecg, a_ecg, ECG_aux, method='gust')
#
#        #PPG_aux = filtfilt(b_ppg, a_ppg, PPG_aux, method='gust')

        t = (np.arange(len(ECG_aux))/250)
        try:
            idx_peaksECG = nk.bio_process(ecg = ECG_aux, sampling_rate=250)['ECG']['R_Peaks']
            #idx_peaksECG = np.delete(idx_peaksECG, np.where(np.diff(t[idx_peaksECG])>1.1))

            t_RR = t[idx_peaksECG]

            HR = 60/np.diff(t_RR)

            idx_peaksPPG = []
            idx_del = []

            for i in range(len(idx_peaksECG)-1):

                ran_0 = idx_peaksECG[i]
                ran_1 = idx_peaksECG[i+1]

                PPG_peak = (find_peaks(PPG_aux[ran_0:ran_1], distance = ran_1-ran_0 )[0] + ran_0)

                if PPG_peak.size>0:#: and PPG_aux[int(PPG_peak[0])]>10:
                    idx_peaksPPG.append(int(PPG_peak[0]))
                else:
                    idx_del.append(i)

           # idx_peaksECG = np.delete(idx_peaksECG, np.array(idx_del))
            HR    = np.delete(HR, np.array(idx_del))
            t_RR1 =  np.delete(t_RR, np.array(idx_del)+1)
            t_RR2 =  np.delete(t_RR, np.array(idx_del))
            t_PPG = t[idx_peaksPPG]

#            plt.figure()
#            plt.subplot(211)
#            plt.plot(t,ECG_aux)
#            plt.scatter(t_RR,ECG_aux[idx_peaksECG], c='r')
#            plt.title(nombres[k])
#            plt.subplot(212)
#            plt.plot(t,PPG_aux,'g')
#            plt.scatter(t_PPG,PPG_aux[idx_peaksPPG], c='y')

            PPT1 = (t_PPG-t_RR2[:-1])
            PPT2 = (t_RR1[1: ]-t_PPG)
            try:
                t_HR_cu.append(np.mean(HR))

                if np.mean(PPT1)>np.mean(PPT2):
                    t_PPT_cu.append(np.mean(PPT1))
                else:
                    t_PPT_cu.append(np.mean(PPT2))
                    
                t_DBP_cu.append(DBP[cont])
                t_SBP_cu.append(SBP[cont])
                cont +=1
    #            plt.figure()
    #            plt.plot(PPT1,'gx',label='PPG peak - previous RR peak ')
    #            plt.plot(PPT2,'rx',label='subsequent RR peak - PPG peak')
    #            plt.legend()
            except:
                print('error dato %s'%nombres[k])
                cont +=1
#                t_HR.append([])
#                t_PPT.append([])

        except:
            print('error dato %s %d'%(nombres[k], j))
#            t_HR.append([])
#            t_PPT.append([])

        idx_a = j
    t_HR.append(t_HR_cu)
    t_PPT.append(t_PPT_cu)
    t_DBP.append(t_DBP_cu)
    t_SBP.append(t_SBP_cu) 
    
    HR_norm     = scale(t_HR[k])
    logPPT_norm = scale((np.log(t_PPT[k])))
    DPB_norm    = scale(t_DBP[k])     
    SBP_norm    = scale(t_SBP[k])
    
    idx = np.ones(np.shape(HR_norm), dtype=bool)
    idx = np.where(abs(HR_norm)<2,idx,False)
    idx = np.where(abs(logPPT_norm)<2,idx,False)   
    idx = np.where(abs(DPB_norm)<2,idx,False)
    idx = np.where(abs(SBP_norm)<2,idx,False)
    
    X = np.transpose(np.array([HR_norm[idx], logPPT_norm[idx]]))
    y1  = np.array(t_DBP[k])[idx]
    y2  = np.array(t_SBP[k])[idx]
    
   
    #regSBP = SVR(kernel='linear', C=3)
    #regDBP = SVR(kernel='linear', C=3)
    
    #regDBP = AdaBoostRegressor(n_estimators=100)
    #regSBP = AdaBoostRegressor(n_estimators=100)
    ##    
    #regDBP = DecisionTreeRegressor()
    #regSBP = DecisionTreeRegressor()
    
    regSBP = linear_model.LinearRegression()
    regDBP = linear_model.LinearRegression()
    
#    kf = KFold(n_splits=1)
#    kf.split(X)    
#    
#    for train, test in kf.split(X):
#        X_train, X_test, y1_train, y1_test, y2_train, y2_test = X[train], X[test], y1[train], y1[test], y2[train], y2[test]
#    
#        regSBP.fit(X_train, y1_train) 
#        regDBP.fit(X_train, y2_train) 
#    
#        errorSBP.append(sqrt(mean_squared_error(y1_test, regSBP.predict(X_test))))
#        errorDBP.append(sqrt(mean_squared_error(y2_test, regDBP.predict(X_test))))

    regSBP.fit(X, y1) 
    regDBP.fit(X, y2) 

    errorSBP.append(sqrt(mean_squared_error(y1, regSBP.predict(X))))
    errorDBP.append(sqrt(mean_squared_error(y2, regDBP.predict(X))))


    coefSBP.append(regSBP.coef_)
    coefDBP.append(regDBP.coef_)
    
    print('Root Mean Squared Error SBP')
    print('mean',np.mean(errorSBP))
    print('Root Mean Squared Error DBP')
    print('mean',np.mean(errorDBP))
    

#    sns.set_style("whitegrid")
#    sns.jointplot(X[:,1], y2, kind ='kde')
#    plt.xlabel('log(PPT)') # Set text for the x axis
#    plt.ylabel('Diastolic Blood Pressure')# Set text for y axi
#    plt.show()
#    
#    sns.set_style("whitegrid")
#    sns.jointplot(X[:,1], y1, kind ='kde')
#    plt.xlabel('log(PPT)') # Set text for the x axis
#    plt.ylabel('Systolic Blood Pressure')# Set text for y axi
#    plt.show()
#    
#    sns.set_style("whitegrid")
#    sns.jointplot(np.array(t_HR[k])[idx], y2, kind ='kde')
#    plt.xlabel('Heart rate') # Set text for the x axis
#    plt.ylabel('Diastolic Blood Pressure')# Set text for y axi
#    plt.show()
#    
#    sns.set_style("whitegrid")
#    sns.jointplot(np.array(t_HR[k])[idx], y1, kind ='kde')
#    plt.xlabel('Heart rate') # Set text for the x axis
#    plt.ylabel('Systolic Blood Pressure')# Set text for y axi
#    plt.show()
# 
plt.show()

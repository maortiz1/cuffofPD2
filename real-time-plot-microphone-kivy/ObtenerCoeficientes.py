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
import sklearn.svm 
from sklearn import preprocessing
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
f=[0]
coefsis=[]
coefdia=[]
coefsis2=[]
coefdia2=[]
pttsvmdia=[]
ppgsvmdia=[]
HRsvmdia=[]
pttsvmsis=[]
ppgsvmsis=[]
HRsvmsis=[]
BPsvmsis=[]
BPsvmdia=[]
#for k in f:
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
#    if len(fecgcut)>len(ECGcut):
#        fecgcut=fecgcut[0:len(ECGcut)-1]
#    if len(fppgcut)>len(PPGcut):
#        fppgcut=fecgcut[0:len(PPGcut)-1]  
    if len(fecgcut)>len(fppgcut):
        fecgcut=fecgcut[0:len(fppgcut)]
    else:
        fppgcut=fppgcut[0:len(fecgcut)]
    #obtener picos RR
    t= np.arange(len(fecgcut))/250
    idx_peaksECG = nk.bio_process(ecg = fecgcut, sampling_rate=250)['ECG']['R_Peaks']
    t_RR = t[idx_peaksECG]
    diff=np.diff(t_RR)
    HR=60/diff
    
    idx_ppgsis = (find_peaks(fppgcut,height=0)[0])
    idx_ppgdia= (find_peaks(-fppgcut,height=0)[0])
    
    pttsis=[]
    pttdia=[]
    ind_ppgsis=[]
    ind_ppgdia=[]
    vec_sis=[]
    vec_dia=[]
    vec_sis.append(newsis[0])
    vec_dia.append(newdia[0])
    for ind in range(len(idx_peaksECG)-1):
        ini=idx_peaksECG[ind]
        fin=idx_peaksECG[ind+1]
        
        ##ppg corte
        idx_ppgsis = (find_peaks(fppgcut[ini:fin],height=0,distance=fin-ini)[0]+ini)
        idx_ppgdia= (find_peaks(-fppgcut[ini:fin],height=0,distance=fin-ini)[0]+ini)
        ind_ppgsis.append(idx_ppgsis[0:-1])
        ind_ppgdia.append(idx_ppgdia[0:-1])
        if idx_ppgsis.size>0:
            ppgcutpeaksis=fppgcut[idx_ppgsis]
            ma=np.argmax(ppgcutpeaksis)           
            ptt=np.abs(t[idx_ppgsis[ma]]-t[ini])
            pttsis.append(np.absolute(ptt))
            vec_sis.append(newsis[idx_ppgsis[ma]])
            
        if idx_ppgdia.size>0:
            ppgcutpeakdia=fppgcut[idx_ppgdia]
            mi=np.argmin(ppgcutpeakdia)           
            ptt=np.abs(t[ini]-t[idx_ppgdia[mi]])
            pttdia.append(np.absolute(ptt))
            vec_dia.append(newdia[idx_ppgdia[mi]])
    #regresion
    #sistole
    #quitar un valor
    
    HR_norm=scale(HR)
    pttsis_norm=scale(np.log(pttsis))  
    pttdia_norm=scale(np.log(pttdia))
    
    sizeHR=len(HR_norm)
    sizepttsis=len(pttsis_norm)
    sizepttdia=len(pttdia_norm)
    sizevecsis=len(vec_sis)
    sizevecdia=len(vec_dia)
    ma=np.min([sizeHR,sizepttsis,sizepttdia])
    HR_norm=HR_norm[0:ma]
    pttsis=pttsis[0:ma]
    pttdia=pttdia[0:ma]
    vec_dian1=vec_dia[0:ma]
    vec_sisn1=vec_sis[0:ma]
    vec_dia0=vec_dia[1:ma+1]
    vec_sis0=vec_sis[1:ma+1]
    
    regSIS=linear_model.LinearRegression();
    xsis=np.transpose(np.array([HR_norm,pttsis,vec_sisn1]))
    regSIS.fit(xsis,vec_sis0)
    
    coefsis.append(regSIS.coef_)
    
    regDIA=linear_model.LinearRegression();
    xdia=np.transpose(np.array([HR_norm,pttdia,vec_dian1]))
    regDIA.fit(xsis,vec_dia0)
    
    coefdia.append(regSIS.coef_)
    regSIS2=linear_model.LinearRegression();
    xsis=np.transpose(np.array([HR_norm,pttsis]))
    regSIS2.fit(xsis,vec_sis0)
    
    coefsis2.append(regSIS.coef_)
    
    regDIA=linear_model.LinearRegression();
    xdia=np.transpose(np.array([HR_norm,pttdia]))
    regDIA.fit(xsis,vec_dia0)
    
    coefdia2.append(regSIS.coef_)
    
    
    
    pttsvmdia.append(np.mean(pttdia))
 
    HRsvmdia.append(np.mean(HR_norm))
    pttsvmsis.append(np.mean(pttsis))
    HRsvmsis.append(np.mean(HR_norm))
    BPsvmdia.append(vec_dia0[0])
    BPsvmsis.append(vec_sis0[0])
    
regSBPA = SVC(kernel='linear', C=300)
    
regSBPB = SVC(kernel='linear', C=300)
    
regSBPC = SVC(kernel='linear', C=300)
regDBPA = SVC(kernel='linear', C=300)   
regDBPB = SVC(kernel='linear', C=300)   
regDBPC = SVC(kernel='linear', C=300)   



coefAsis=[]
coefBsis=[]
coefCsis=[]
coefAdia=[]
coefBdia=[]
coefCdia=[]
for a in coefsis:
    coefAsis.append(a[0])
    coefBsis.append(a[1])
    coefCsis.append(a[2])
for b in coefdia:
    coefAdia.append(b[0])
    coefBdia.append(b[1])
    coefCdia.append(b[2])    




lab_enc1=preprocessing.LabelEncoder()
lab_enc2=preprocessing.LabelEncoder()
lab_enc3=preprocessing.LabelEncoder()
lab_enc4=preprocessing.LabelEncoder()
lab_enc6=preprocessing.LabelEncoder()
lab_enc5=preprocessing.LabelEncoder()
X=np.transpose(np.array([HRsvmsis,pttsvmsis,BPsvmsis]))    
coefAsis = lab_enc1.fit_transform(coefAsis)    
coefBsis = lab_enc2.fit_transform(coefBsis) 
coefCsis = lab_enc3.fit_transform(coefCsis)     
  
regSBPA.fit(X[0:-1],coefAsis[0:-1])   
regSBPB.fit(X[0:-1],coefBsis[0:-1])
regSBPC.fit(X[0:-1],coefCsis[0:-1])      

coefAdia = lab_enc4.fit_transform(coefAdia)    
coefBdia = lab_enc5.fit_transform(coefBdia) 
coefCdia = lab_enc6.fit_transform(coefCdia)

X2=np.transpose(np.array([HRsvmdia,pttsvmdia,BPsvmdia]))    
regDBPA.fit(X2[0:-1],coefAdia[0:-1])   
regDBPB.fit(X2[0:-1],coefBdia[0:-1])           
regDBPC.fit(X2[0:-1],coefCdia[0:-1])       

Asis=regSBPA.predict([X[-1]])  

Bsis=regSBPB.predict([X[-1]])  
       

Csis=regSBPC.predict([X[-1]])  
   

Cdia=regDBPC.predict([X2[-1]])     
Bdia=regDBPB.predict([X2[-1]])       
Adia=regDBPA.predict([X2[-1]])           
    
Asis=lab_enc1.inverse_transform(Asis)
Bsis=lab_enc2.inverse_transform(Bsis)
Csis=lab_enc3.inverse_transform(Csis)

Adia=lab_enc4.inverse_transform(Adia)
Bdia=lab_enc5.inverse_transform(Bdia)
Cdia=lab_enc6.inverse_transform(Cdia)







# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:15:45 2018

@author: m_ana
"""
import numpy as np
from statsmodels import robust
import matplotlib.pylab as plt
import scipy
from scipy.signal import filtfilt
import pywt

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

    
    
ecg = np.genfromtxt('https://raw.githubusercontent.com/maortiz1/cuffofPD2/master/real-time-plot-microphone-kivy/11-9-12-44_ecg.txt',delimiter=',')
ppg = np.genfromtxt('https://raw.githubusercontent.com/maortiz1/cuffofPD2/master/real-time-plot-microphone-kivy/11-9-12-44_ppg.txt',delimiter=',')

#Remove nan Values for filter
ecg = ecg[~np.isnan(ecg)]
ppg = ppg[~np.isnan(ppg)]

b, a = scipy.signal.iirnotch(0.48,30)
ecg = filtfilt(b, a, ecg, method='gust')
ppg = filtfilt(b, a, ppg, method='gust')

#ECG Filter
b, a = scipy.signal.butter(1,[0.08,0.72],'bandpass')
x = filtfilt(b, a, ecg, method='gust')
x2 = waveletFilt(x,"db4",1)
#PPG Filter
b, a = scipy.signal.butter(1,[0.01,0.04],'bandpass')
y = filtfilt(b, a, ppg, method='gust')

#Filter Outliers
eMed = -0.105638066
eDev = 0.143986308
pMed = 49.91691522
indOutlier=(1000)<np.abs(x2)
x2[indOutlier]=0
indOutlier=(5000)<np.abs(y)
y[indOutlier]=0


plt.figure()
#plt.plot(ecg,'r')
#plt.plot(x,'g')
plt.plot(x2,'b')
#plt.plot(oX,'b')

plt.figure()
#plt.plot(ppg,'r')
plt.plot(y,'g')
#plt.plot(oY,'b')

plt.show()


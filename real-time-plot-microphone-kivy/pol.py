# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:15:45 2018

@author: m_ana
"""
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.signal import filtfilt, kaiserord, lfilter, firwin, freqz
from sklearn.preprocessing import scale
#Biosignals
import neurokit as nk

datos = np.genfromtxt('https://raw.githubusercontent.com/maortiz1/cuffofPD2/master/real-time-plot-microphone-kivy/11-9-11-17_ecg.txt',delimiter=',')
plt.figure()
plt.plot(datos)
#plt.figure()
#plt.plot(datos[:,1])

b, a = scipy.signal.butter(1,[0.04,0.92],'bandpass')
x = datos[:-1]
# ecg
y = filtfilt(b, a, x, method='gust')

# b, a = scipy.signal.butter(1,[0.01,0.04],'bandpass')
# x = datos[:-1]
# # ppg
# y = filtfilt(b, a, x, method='gust')
#
# # b, a = scipy.signal.butter(1,0.01,'highpass')
# y = filtfilt(y, a, x, method='gust')
print(y)
plt.figure()
plt.plot(x,'r')
plt.plot(y,'g')


fs = 250
data_ECG = scale(x)
t = (np.arange(len(data_ECG))/fs)
idx_peaksECG = nk.bio_process(ecg = data_ECG, sampling_rate=125)['ECG']['R_Peaks']
t_RR = t[idx_peaksECG]

plt.figure()

plt.plot(t,data_ECG)
plt.scatter(t[idx_peaksECG],data_ECG[idx_peaksECG], c ='r')

plt.show()

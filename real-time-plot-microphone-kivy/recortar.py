import numpy as np
#Directories
#import os
#from os import listdir
#from os.path import isfile, join
#Plot
import matplotlib.pylab as plt

datos_t = np.genfromtxt('Copia de Datos real.txt', dtype = str)

nombres = datos_t[::9,0]

cortar = [150000, 50000, 22000, 105000, 115000, 46000, 64000, 43000, 60000, 70000, 80000, 88000, 72000, 5500, 22000, 46000, 13000, 29000, 65000]

for i in range(nombres.shape[0]):
    print(nombres[i])
    ECG = np.genfromtxt('%s_ecg_copy.txt'%nombres[i], delimiter = ',')
    print(ECG.shape)
#    ECG = ECG[cortar[i]:-1]
#    print(ECG.shape)
#    file = open('%s_ecg_copy.txt'%(nombres[i]),'a+')
#
#    for j in range(ECG.shape[0]):
#        file.write(str( '%d \n'%ECG[j]))

#    file.close()
#   
    PPG = np.genfromtxt('%s_ppg_copy.txt'%nombres[i], delimiter = ',')
    print(PPG.shape)
#    PPG = PPG[cortar[i]:-1]
#    print(PPG.shape)

#    file1 = open('%s_ppg_copy.txt'%(nombres[i]),'a+')
#
#    for j in range(PPG.shape[0]):
#        file1.write(str( '%d \n'%PPG[j]))
#
#    file1.close()


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

for i in range(nombres.shape[0]-1, nombres.shape[0]):
    plt.figure()
    ECG = np.genfromtxt('%s_ecg.txt'%nombres[i], delimiter = ',')
    plt.subplot(211)
    plt.plot(ECG[:-1])
    ECG = ECG[cortar[i]:-1]
    plt.plot(ECG)
    file = open('%s_ecg_copy.txt'%(nombres[i]),'a+')

    for j in range(ECG.shape[0]):
        file.write(str( '%d \n'%ECG[j]))

    file.close()
    plt.subplot(212)
    PPG = np.genfromtxt('%s_ppg.txt'%nombres[i], delimiter = ',')
    plt.plot(PPG[:-1])
    PPG = PPG[cortar[i]:-1]
    plt.plot(PPG)

    file1 = open('%s_ppg_copy.txt'%(nombres[i]),'a+')

    for j in range(PPG.shape[0]):
        file1.write(str( '%d \n'%PPG[j]))

    file1.close()

plt.show()

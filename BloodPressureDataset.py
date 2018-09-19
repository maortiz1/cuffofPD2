import numpy as np
import os
from os import listdir
from os.path import isfile, join

cwd = os.getcwd()

#Get the names of the .csv file
path_samples_csv= os.path.join(cwd,'mat_to_csv')
files_csv = [f for f in listdir(path_samples_csv) if isfile(join(path_samples_csv, f))]
#read the files
os.chdir(path_samples_csv)

data_train_name = []
data_train_PPG  = []
data_train_ABP  = []
data_train_ECG  = []

data_test_name = []
data_test_PPG  = []
data_test_ABP  = []
data_test_ECG  = []

for i in range(len(files_csv)-6000):
    #print(files_csv[i])
    Matrix_data = np.genfromtxt(files_csv[i], delimiter=",")
    data_train_name.append(files_csv[i])
    data_train_PPG.append(Matrix_data[0,:])
    data_train_ABP.append(Matrix_data[1,:])
    data_train_ECG.append(Matrix_data[2,:])       

for i in range(6000,12000):
    #print(files_csv[i])
    Matrix_data = np.genfromtxt(files_csv[i], delimiter=",")
    data_test_name.append(files_csv[i])
    data_test_PPG.append(Matrix_data[0,:])
    data_test_ABP.append(Matrix_data[1,:])
    data_test_ECG.append(Matrix_data[2,:])    
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
#from scipy.signal import find_peaks
import pickle

##############################################################
cwd = os.getcwd()
path_RR = os.path.join(cwd,'RR.pkl')


with open(path_RR, 'rb') as f:
    t_RRtrain = pickle.load(f)


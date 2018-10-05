#Arrays
import numpy as np
#Directories
import os
#Plot
import matplotlib.pylab as plt
#Save variables
import pickle

from sklearn import linear_model

##############################################################
cwd = os.getcwd()
path_HR = os.path.join(cwd,'HR.pkl')
path_PPT= os.path.join(cwd,'PPT.pkl')
path_SBP= os.path.join(cwd,'SBP.pkl')
path_DBP= os.path.join(cwd,'DBP.pkl')

with open(path_HR, 'rb') as f:
    HRtrain = pickle.load(f)

with open(path_PPT, 'rb') as f:
    PPTtrain = pickle.load(f)
    
with open(path_SBP, 'rb') as f:
    SBPtrain = pickle.load(f)
    
with open(path_DBP, 'rb') as f:
    DBPtrain = pickle.load(f)
    
#########################################################################
n = len(HRtrain)
    
aSBP = np.empty(n)
bSBP = np.empty(n)

aDBP = np.empty(n)
bDBP = np.empty(n)

errorSBP = np.empty(n)
errorDBP = np.empty(n)

##########################################################################

for i in range(n):
    
    HR_norm = (HRtrain[i]-np.mean(HRtrain[i]))/(np.var(HRtrain[i])**0.5)
    logPPT_norm = (np.log(PPTtrain[i])-np.mean(np.log(PPTtrain[i])))/(np.var(np.log(PPTtrain[i]))**0.5)
    
    idx = np.ones(np.shape(HR_norm), dtype=bool)
    idx = np.where(abs(HR_norm)<2,idx,False)
    idx = np.where(abs(logPPT_norm)<2,idx,False)    

    X = np.transpose(np.array([HR_norm[idx], logPPT_norm[idx]]))
    y1  = SBPtrain[i][idx]
    y2  = DBPtrain[i][idx]
    
    regSBP = linear_model.LinearRegression()
    regDBP = linear_model.LinearRegression()
    
    regSBP.fit(X, y1)
    regDBP.fit(X, y2)
    
    aSBP[i], bSBP[i] = regSBP.coef_
    aDBP[i], bDBP[i] = regDBP.coef_
    
    errorSBP[i] = 1/(len(y1)) * sum((regSBP.predict(X)-y1)**2)
    errorDBP[i] = 1/(len(y2)) * sum((regDBP.predict(X)-y2)**2)
    
    
    
    

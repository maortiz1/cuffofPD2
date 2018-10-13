#Arrays
import numpy as np
#Directories
import os
#Plot
import matplotlib.pylab as plt
#Scale
from sklearn.preprocessing import scale
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
    
    HR_norm     = scale(HRtrain[i])
    logPPT_norm = scale(np.log(PPTtrain[i]))
    DPB_norm    = scale(DBPtrain[i])     
    SBP_norm    = scale(SBPtrain[i])
    
    idx = np.ones(np.shape(HR_norm), dtype=bool)
    idx = np.where(abs(HR_norm)<2,idx,False)
    idx = np.where(abs(logPPT_norm)<2,idx,False)   
    idx = np.where(abs(DPB_norm)<2,idx,False)
    idx = np.where(abs(SBP_norm)<2,idx,False)
    
    
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
    
############################################################################ 
    
idx_coef = np.ones(n, dtype=bool)
idx_coef = np.where(abs(aSBP)<2000,idx_coef,False)
idx_coef = np.where(abs(bSBP)<2000,idx_coef,False)
idx_coef = np.where(abs(aDBP)<2000,idx_coef,False)
idx_coef = np.where(abs(bDBP)<2000,idx_coef,False)

print('aSBP')
print('mean',np.mean(aSBP[idx_coef]) , 'std', np.var(aSBP[idx_coef])**0.5)
print('bSBP')
print('mean',np.mean(bSBP[idx_coef]) , 'std', np.var(bSBP[idx_coef])**0.5)
print('Mean Squared Error SBP')
print(np.mean(errorSBP[idx_coef]))
print('aDBP')
print('mean',np.mean(aDBP[idx_coef]) , 'std', np.var(aDBP[idx_coef])**0.5)
print('bDBP')
print('mean',np.mean(bDBP[idx_coef]) , 'std', np.var(bDBP[idx_coef])**0.5)
print('Mean Squared Error DBP')
print(np.mean(errorDBP[idx_coef]))

l = errorSBP[idx_coef]

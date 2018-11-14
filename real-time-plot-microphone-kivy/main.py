"""Real time plotting of Microphone level using kivy
"""

from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.garden.graph import MeshLinePlot, LinePlot
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from pylsl import StreamInfo, StreamOutlet
from functools import partial
import sys
import atexit
import random
import threading
import time
import datetime
import numpy as np
from statsmodels import robust
import matplotlib.pylab as plt
import scipy
from scipy.signal import filtfilt
import pywt
from kivy.clock import Clock
from threading import Thread
from sklearn.preprocessing import scale
#Find_peaks
import scipy
from sklearn import linear_model

#ML

from sklearn import preprocessing
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error
from math import sqrt
import neurokit as nk

from scipy.signal import find_peaks, filtfilt
#import audioop
#import pyaudio
import open_bci_v3 as bci

class Logic(BoxLayout):
    random_number = StringProperty()
    minecg = NumericProperty()
    maxecg = NumericProperty()
    minppg = NumericProperty()
    maxppg = NumericProperty()
    strDBP = StringProperty()
    strSBP = StringProperty()
    strHR  = StringProperty()
    strPO  = StringProperty()
    def __init__(self,):
        super(Logic, self).__init__()
        self.plot  = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        self.plot2 = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        
        self.random_number = str('')
        self.strDBP = str('')
        self.strSBP = str('')
        self.strHR  = str('')
        self.strPO  = str('96')
        self.minecg=0
        self.maxecg=1000
        self.minppg=0
        self.maxppg=1000
        self.file = []
        self.HR = [82]
#        try:

        self.bcbB = bciBoardConnect()
        self.bcbB.createlsl()
        self.fil=filter()
        self.guesser= estimatePressure();
        self.DiaA=[80];
        self.SisA=[120];

#        except:
#            popup=Popup(title='Error',
#                content=Label(text='Ocurrio un error conectandose al OPENBCI \n Desconecte el modulo y vuelva a intentar'),
#                size_hint=(None,None), size=(200,200))
#
#            popup.open()
#            time.sleep(.5)
#            App.get_running_app().stop()
#            sys.exit()

    def change_text(self, dt):
        self.strPO = str(80+random.randint(1, 19))


        #Cuando funcionen los calculos toca cambiar lo de random....

    def start(self):

        self.ids.ppg_graph.add_plot(self.plot2)
        self.ids.ecg_graph.add_plot(self.plot)
        self.bcbB.startstreaming()

        Clock.schedule_interval(self.get_value, 1/250)
        Clock.schedule_interval(self.calculartod, 10)
        Clock.schedule_interval(self.change_DBP, 2/250)
        Clock.schedule_interval(self.change_text, 20)
        
        

    def stop(self):

        Clock.unschedule(self.get_value)
        Clock.unschedule(self.change_DBP)
        self.bcbB.stopstreaming()

    def start_recording(self):
        now = datetime.datetime.now()
#        self.datanow=[];
        self.bcbB.chSave()
        self.file = open('%i-%i-%i-%i_ecg.txt'%(now.month, now.day, now.hour ,now.minute),'a+')
        self.file1 = open('%i-%i-%i-%i_ppg.txt'%(now.month, now.day, now.hour ,now.minute),'a+')
#        Clock.schedule_interval((self.putdataontxt), 1/250)

    def stop_recording(self):
#        Clock.unschedule(partial(self.putdataontxt, self.file))
        ecg=self.bcbB.retecgT()
        ppg= self.bcbB.retppgT()

        for s in ecg:
            self.file.write(str(s) + ',')
        for d in ppg:
            self.file1.write(str(d) + ',')

        self.bcbB.flush()
        self.file.close()
        self.file1.close()
        self.bcbB.chSave()

    def get_value(self, dt):

        if len(self.bcbB.retdataecgchunk())>2 and (min(self.bcbB.retdataecgchunk()) < max(self.bcbB.retdataecgchunk())):
            try:
                self.minecg=min(self.bcbB.retdataecgchunk())
                self.maxecg=max(self.bcbB.retdataecgchunk())
            except:
                print('nop')
        if len(self.bcbB.retdatappgchunk())>2 and (min(self.bcbB.retdatappgchunk()) < max(self.bcbB.retdatappgchunk())):
            try:
                self.minppg=min(self.bcbB.retdatappgchunk())
                self.maxppg=max(self.bcbB.retdatappgchunk())
            except:
                print('nop2')

        self.plot2.points = [(index/250,value) for index, value in enumerate(self.bcbB.retdatappgchunk())]

        self.plot.points = [(index/250, value) for index, value in enumerate(self.bcbB.retdataecgchunk())]


    def change_DBP(self, dt):
  
        self.strDBP = str(int(self.DiaA[-1]))
        self.strSBP = str(int(self.SisA[-1]))
        if len(self.HR) is not 0:
            self.strHR  = str(int(self.HR[-1]))
        else:
            self.strHR  = str('')
         
        
    def calculartod(self,dt):
        self.ecg30,self.ppg30=self.bcbB.ret30()
        if len(self.ecg30)>7000:
            fecgcut,fppgcut=self.fil.filtrar(self.ecg30,self.ppg30) 
            if len(fecgcut)>len(fppgcut):
                fecgcut=fecgcut[0:len(fppgcut)]
            else:
                fppgcut=fppgcut[0:len(fecgcut)]
                
            t= np.arange(len(fecgcut))/250
            idx_peaksECG = nk.bio_process(ecg = fecgcut, sampling_rate=250)['ECG']['R_Peaks']
            t_RR = t[idx_peaksECG]
            diff=np.diff(t_RR)
            HRs=60/diff;
            
            HRs[np.isnan(HRs)]=60
            
            print(HRs)
            self.HR.append(np.mean(HRs))
            pttsis=[]
            pttdia=[]
            ind_ppgsis=[]
            ind_ppgdia=[]
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
                
            
                if idx_ppgdia.size>0:
                    ppgcutpeakdia=fppgcut[idx_ppgdia]
                    mi=np.argmin(ppgcutpeakdia)           
                    ptt=np.abs(t[ini]-t[idx_ppgdia[mi]])
                    pttdia.append(np.absolute(ptt))
                
            HR_norm=scale(HRs)
            pttsis_norm=scale(np.log(pttsis))  
            pttdia_norm=scale(np.log(pttdia))
            
            sizeHR=len(HR_norm)
            sizepttsis=len(pttsis_norm)
            sizepttdia=len(pttdia_norm)
            
            ma=np.min([sizeHR,sizepttsis,sizepttdia])
            HR_norm=HR_norm[0:ma]
            pttsis=pttsis_norm[0:ma]
            pttdia=pttdia_norm[0:ma] 
            self.meanpttsis=np.mean(pttsis)
            self.meanpttdia=np.mean(pttdia)
            self.HRnormena=np.mean(HR_norm)
            diaAn=np.mean(self.DiaA)
            sisAn=np.mean(self.SisA)
            aSis,bSis,cSis,aDias,bDias,cDias=self.guesser.predecir(self.HRnormena,self.meanpttsis,self.meanpttdia,sisAn,diaAn)
            diaHoy=aDias*self.HRnormena+bDias*self.meanpttdia+cDias*diaAn
            sisHoy=aSis*self.HRnormena+bSis*self.meanpttdia+cSis*sisAn
            if diaHoy<150 and sisHoy<250:
                self.DiaA.append(int(diaHoy))
                self.SisA.append(int(sisHoy))
            print(diaHoy)
            print(sisHoy)
         
            

class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")
class bciBoardConnect():
    def __init__(self):
        self.board=bci.OpenBCIBoard(port='COM3')
        self.eeg_channels = self.board.getNbEEGChannels()
        self.aux_channels = self.board.getNbAUXChannels()
        self.sample_rate = self.board.getSampleRate()
        self.save=False
        self.graphppg=[];
        self.graphecg=[];
        self.ecg=[];
        self.ppg=[];
        self.ecg30=[];
        self.ppg30=[];
       #setting channel 6
        beg='x6000000X';
        #setting default and reseting board
        s='sv'
        s=s+'d'
        #writing data to board
        s=s+'F'

        for c in s:
            if sys.hexversion > 0x03000000:
                self.board.ser.write(bytes(c, 'utf-8'))
            else:
                self.board.ser.write(bytes(c))
                time.sleep(0.100)
        #writing channel six data to board
        time.sleep(0.100)
#        for dat in beg:
        for x in beg:
            if sys.hexversion > 0x03000000:
               self.board.ser.write(bytes(x, 'utf-8'))
            else:
                self.board.ser.write(bytes(x))
                time.sleep(0.100)

        self.ecg = []
        self.ppg = []
        self.ecgT=[]
        self.ppgT=[]

    #function to callback while board streaming data in and save it
    def send(self,sample):
        #print(sample.channel_data)
        self.ecg = (sample.channel_data[3])
        self.ppg = (sample.channel_data[5])

        if len(self.graphppg)<1250:
            self.graphecg.append(sample.channel_data[3])
            self.graphppg.append(sample.channel_data[5])
        else:
            self.graphppg=[]
            self.graphecg=[]
        self.outlet_eeg.push_sample(sample.channel_data)
        self.outlet_aux.push_sample(sample.aux_data)
        if self.save:
            self.ecgT.append(sample.channel_data[3])
            self.ppgT.append(sample.channel_data[5])
        if len(self.ppg30)<7500:
            self.ppg30.append(sample.channel_data[5])
            self.ecg30.append(sample.channel_data[3])
        else:
            self.ppg30=[]
            self.ppg30=[]

    def createlsl(self):
        info_eeg = StreamInfo("OpenBCI_EEG", 'EEG', self.eeg_channels, self.sample_rate,'float32',"openbci_eeg_id1");
        info_aux = StreamInfo("OpenBCI_AUX", 'AUX', self.aux_channels,self.sample_rate,'float32',"openbci_aux_id1")
        self.outlet_eeg = StreamOutlet(info_eeg)
        self.outlet_aux = StreamOutlet(info_aux)
    def clean(self):
        self.board.disconnect()
        self.outlet_eeg.close_stream()
        self.outlet_aux.close_stream()
        atexit.register(clean)

    def startstreaming(self):

        self.boardThread=threading.Thread(target=self.board.start_streaming,args=(self.send,-1))
        self.boardThread.daemon=True
        try:
            self.boardThread.start()

        except:
            raise
    def stopstreaming(self):
        self.board.stop()
        time.sleep(.1)
        line=''
        while self.board.ser.inWaiting():
            c=self.board.ser.read().decode('utf-8',errors='replace')
            line+=c
            time.sleep(0.001)
            if(c=='\n'):
                line=''
    def retdataecgchunk(self):
        return self.graphecg
    def retdatappgchunk(self):
        return self.graphppg
    def retbothdata(self):
        return self.ecg, self.ppg
    def retecgT(self):
        return self.ecgT
    def retppgT(self):
        return self.ppgT
    def flush(self):
        self.ppgT=[]
        self.ecgT=[]
    def chSave(self):
        if self.save==False:
            self.save=True
        else:
            self.save=False
            
    def ret30(self):
        return self.ecg30,self.ppg30
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
    def waveletFilt( self,x, wavelet, level):
        coeff = pywt.wavedec( x, wavelet, mode="per" )
        sigma = robust.mad( coeff[-level] )
        uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
        coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
        y = pywt.waverec( coeff, wavelet, mode="per" )
        return(y)
class estimatePressure():
    def __init__(self):
        self.datos=np.genfromtxt('SVMmod.txt')
        self.HRsvmsis=self.datos[::,0]
        self.BPsvmsis=self.datos[::,1]
        self.pttsvmsis=self.datos[::,2]
        self.HRsvmdia=self.datos[::,3]
        self.pttsvmdia=self.datos[::,4]
        self.BPsvmdia=self.datos[::,5]
        self.coefAsis=self.datos[::,6]
        self.coefBsis=self.datos[::,7]
        self.coefCsis=self.datos[::,8]
        self.coefAdia=self.datos[::,9]
        self.coefBdia=self.datos[::,10]
        self.coefCdia=self.datos[::,11]
        
        
        self.lab_enc1=preprocessing.LabelEncoder()
        self.lab_enc2=preprocessing.LabelEncoder()
        self.lab_enc3=preprocessing.LabelEncoder()
        self.lab_enc4=preprocessing.LabelEncoder()
        self.lab_enc6=preprocessing.LabelEncoder()
        self.lab_enc5=preprocessing.LabelEncoder()
        self.X=np.transpose(np.array([self.HRsvmsis,self.pttsvmsis,self.BPsvmsis]))    
        self.coefAsis = self.lab_enc1.fit_transform(self.coefAsis)    
        self.coefBsis = self.lab_enc2.fit_transform(self.coefBsis) 
        self.coefCsis = self.lab_enc3.fit_transform(self.coefCsis)     

        self.regSBPA = SVC(kernel='linear', C=300)
            
        self.regSBPB = SVC(kernel='linear', C=300)
            
        self.regSBPC = SVC(kernel='linear', C=300)
        self.regDBPA = SVC(kernel='linear', C=300)   
        self.regDBPB = SVC(kernel='linear', C=300)   
        self.regDBPC = SVC(kernel='linear', C=300)   



        self.regSBPA.fit(self.X,self.coefAsis)   
        self.regSBPB.fit(self.X,self.coefBsis)
        self.regSBPC.fit(self.X,self.coefCsis)      

        self.coefAdia = self.lab_enc4.fit_transform(self.coefAdia)    
        self.coefBdia = self.lab_enc5.fit_transform(self.coefBdia) 
        self.coefCdia = self.lab_enc6.fit_transform(self.coefCdia)

        self.X2=np.transpose(np.array([self.HRsvmdia,self.pttsvmdia,self.BPsvmdia]))    
        self.regDBPA.fit(  self.X2,  self.coefAdia)   
        self.regDBPB.fit(  self.X2,  self.coefBdia)           
        self.regDBPC.fit(  self.X2,  self.coefCdia)    
        
    def predecir(self,HR,pptsis,pptdia,bpasis,bpadias):
        self.X=[HR,pptsis,bpasis]
        self.X2=[HR,pptdia,bpadias]
        self.Asis=self.regSBPA.predict([self.X])  
        self.Bsis=self.regSBPB.predict([self.X])        
        self.Csis=self.regSBPC.predict([self.X])
        self.Cdia=self.regDBPC.predict([self.X2])     
        self.Bdia=self.regDBPB.predict([self.X2])       
        self.Adia=self.regDBPA.predict([self.X2])  
        ##decode
        self.Asis=self.lab_enc1.inverse_transform(self.Asis)
        self.Bsis=self.lab_enc2.inverse_transform(self.Bsis)
        self.Csis=self.lab_enc3.inverse_transform(self.Csis)
        self.Adia=self.lab_enc4.inverse_transform(self.Adia)
        self.Bdia=self.lab_enc5.inverse_transform(self.Bdia)
        self.Cdia=self.lab_enc6.inverse_transform(self.Cdia)
        return self.Asis,self.Bsis,self.Csis,self.Adia,self.Bdia,self.Cdia
    
if __name__ == "__main__":
    levels = []  # store levels of microphone
#    get_level_thread = Thread(target = get_microphone_level)
#    get_level_thread.daemon = True
#    get_level_thread.start()
    RealTimeMicrophone().run()

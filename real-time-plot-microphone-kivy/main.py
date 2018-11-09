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

from kivy.clock import Clock
from threading import Thread

#import audioop
#import pyaudio
import open_bci_v3 as bci

class Logic(BoxLayout):
    random_number = StringProperty()
    minecg = NumericProperty()
    maxecg = NumericProperty()
    minppg = NumericProperty()
    maxppg = NumericProperty()
    def __init__(self,):
        super(Logic, self).__init__()
        self.plot = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        self.plot2 = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        self.random_number = str('')
        self.minecg=0
        self.maxecg=1000
        self.minppg=0
        self.maxppg=1000
        self.file = []
#        try:

        self.bcbB = bciBoardConnect()
        self.bcbB.createlsl()

#        except:
#            popup=Popup(title='Error',
#                content=Label(text='Ocurrio un error conectandose al OPENBCI \n Desconecte el modulo y vuelva a intentar'),
#                size_hint=(None,None), size=(200,200))
#
#            popup.open()
#            time.sleep(.5)
#            App.get_running_app().stop()
#            sys.exit()

    def change_text(self):
        self.random_number = str(random.randint(1, 100))
    def start(self):

        self.ids.ppg_graph.add_plot(self.plot2)
        self.ids.ecg_graph.add_plot(self.plot)
        self.bcbB.startstreaming()

        Clock.schedule_interval(self.get_value, 1/250)
        Clock.schedule_interval(self.change_DBP, 0.5)

    def stop(self):

        Clock.unschedule(self.get_value)
        Clock.unschedule(self.change_DBP)
        self.bcbB.stopstreaming()

    def start_recording(self):
        now = datetime.datetime.now()
        self.file = open('%i-%i-%i-%i.txt'%(now.month, now.day, now.hour ,now.minute),'w')
        Clock.schedule_interval(partial(self.putdataontxt, self.file), 1/250)

    def stop_recording(self):
        Clock.unschedule(partial(self.putdataontxt, self.file))

    def putdataontxt(self, file, dt):
        file.write('\n %d  %d'%(self.bcbB.retbothdata()[0],self.bcbB.retbothdata()[1]))
        print('%d  %d'%(self.bcbB.retbothdata()[0],self.bcbB.retbothdata()[1]))

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

        self.plot2.points = [(index/259,value) for index, value in enumerate(self.bcbB.retdatappgchunk())]

        self.plot.points = [(index/250, value) for index, value in enumerate(self.bcbB.retdataecgchunk())]


    def change_DBP(self, dt):
        self.random_number = str(random.randint(1, 100))

class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")
class bciBoardConnect():
    def __init__(self):
        self.board=bci.OpenBCIBoard(port='COM5')
        self.eeg_channels = self.board.getNbEEGChannels()
        self.aux_channels = self.board.getNbAUXChannels()
        self.sample_rate = self.board.getSampleRate()

        self.graphppg=[];
        self.graphecg=[];
        self.ecg=[];
        self.ppg=[];
       #setting channel 6
        beg='x6000000X';
        #setting default and reseting board
        s='sv'
        s=s+'d'
        #writing data to board


        for c in s:
            if sys.hexversion > 0x03000000:
                self.board.ser.write(bytes(c, 'utf-8'))
            else:
                self.board.ser.write(bytes(c))
                time.sleep(0.100)
        #writing channel six data to board
        time.sleep(0.100)
        for x in beg:
            if sys.hexversion > 0x03000000:
                self.board.ser.write(bytes(x, 'utf-8'))
            else:
                self.board.ser.write(bytes(x))
                time.sleep(0.100)

        self.ecg = []
        self.ppg = []

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



if __name__ == "__main__":
    levels = []  # store levels of microphone
#    get_level_thread = Thread(target = get_microphone_level)
#    get_level_thread.daemon = True
#    get_level_thread.start()
    RealTimeMicrophone().run()

"""Real time plotting of Microphone level using kivy
"""

from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.garden.graph import MeshLinePlot, LinePlot
from kivy.properties import StringProperty

import random

from kivy.clock import Clock
from threading import Thread
import audioop
import pyaudio

def get_microphone_level():
    """
    source: http://stackoverflow.com/questions/26478315/getting-volume-levels-from-pyaudio-for-use-in-arduino
    audioop.max alternative to audioop.rms
    """
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    

    s = p.open(format=FORMAT,
               channels=CHANNELS,
               rate=RATE,
               input=True,
               frames_per_buffer=chunk)
    global levels
    while True:
        data = s.read(chunk)
        mx = audioop.rms(data, 2)
        if len(levels) >= 100:
            levels = []
        levels.append(mx)


class Logic(BoxLayout):
    random_number = StringProperty()
    def __init__(self,):
        super(Logic, self).__init__()
        self.plot = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        self.plot2 = MeshLinePlot(color=[0.09, 0.63, 0.8, 1])
        
        self.random_number = str('')

    def change_text(self):
        self.random_number = str(random.randint(1, 100))
    def start(self):
        self.ids.graph2.add_plot(self.plot2)
        self.ids.graph.add_plot(self.plot)
        
        
        Clock.schedule_interval(self.get_value, 1/250)
        Clock.schedule_interval(self.change_DBP, 0.5)

    def stop(self):
        Clock.unschedule(self.get_value)
        Clock.unschedule(self.change_DBP)

    def get_value(self, dt):
        self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
        self.plot2.points = [(i, j/5) for i, j in enumerate(levels)]
       
        
    def change_DBP(self, dt):
        self.random_number = str(random.randint(1, 100))


class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")

if __name__ == "__main__":
    levels = []  # store levels of microphone
    get_level_thread = Thread(target = get_microphone_level)
    get_level_thread.daemon = True
    get_level_thread.start()
    RealTimeMicrophone().run()
    

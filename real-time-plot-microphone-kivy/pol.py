# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:15:45 2018

@author: m_ana
"""
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.signal import kaiserord, lfilter, firwin, freqz

datos = np.genfromtxt('11-9-9-1.txt')
plt.figure()
plt.plot(datos[:,0])
plt.figure()
plt.plot(datos[:,1])

fs = 250
N = datos.shape[0]
amp = 2*np.sqrt(2)
t = np.array(N)/fs

x = datos[:,0]
nyq_rate = fs / 2.0
width = 5.0/nyq_rate
ripple_db = 60.0

N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
cutoff_hz = 10.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)


# Plot the original signal.
plt.plot(t, x)
delay = 0.5 * (N-1) / fs
# Plot the filtered signal, shifted to compensate for the phase delay.
plt.plot(t-delay, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plt.plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

plt.xlabel('t')
plt.grid(True)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import random
from signal_generator import matrix
from signal_generator import detect_peaks
from signal_generator import accuracy


def ecg_signal_generator(heartbeat, duration):
    # Sampling frequency
    fs = 300

    # Duration of a single beat
    beat_duration = 60 / heartbeat  # Rough duration for one heart beat

    # Define time array for a single beat
    t_single = np.linspace(0, beat_duration, int(fs*beat_duration), endpoint=False)

    P_a = 0.15
    Q_a = 0.35
    R_a = 0.4
    S_a = 0.45
    T_a = 0.6

    # Gaussian models for P, Q, R, S, T waves
    P = np.exp(-(t_single-P_a)**2/(2*0.03**2))
    Q = np.exp(-(t_single-Q_a)**2/(2*0.005**2))
    R = np.exp(-(t_single-R_a)**2/(2*0.02**2))
    S = np.exp(-(t_single-S_a)**2/(2*0.005**2))
    T = np.exp(-(t_single-T_a)**2/(2*0.02**2))

    position_matrix = [[P_a],[Q_a],[R_a],[S_a],[T_a]]


    # Calculate the number of beats in the duration
    num_beats = int(duration / beat_duration)

    ecg_signal = []

    for i in range(num_beats+1):
        # Construct a single heartbeat waveform
        P_height = 0.001*random.randrange(5,25)
        R_height = 0.01*random.randrange(23,26)
        Q_height = 0.0001*random.randrange(int(1500*R_height), int(2000*R_height))
        S_height = 0.001*random.randrange(20,30)
        T_height = 0.0001*random.randrange(int(1200*R_height), int(2000*R_height))


        single_beat = P_height*P - Q_height*Q + R_height*R - S_height*S + T_height*T
        ecg_signal.extend(single_beat)
    

    # Create the ECG signal by tiling the single beat
    #ecg_signal = np.tile(single_beat, num_beats)[:int(fs*duration)]
    

    # Adjust the time vector t to match the length of ecg_signal
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)

    # To match arrays
    difference = len(t)-len(ecg_signal)
    ecg_signal = ecg_signal[:difference]

    for beat in range(num_beats):
        for peak in position_matrix:
                peak.append(peak[-1]+beat_duration)

    return t, ecg_signal, position_matrix

def add_noise(signal, scale):
        noise = np.random.normal(0, scale, len(signal))
        noisy_signal = signal + noise
        return noisy_signal


def plot_ecg_waves(position_matrix):
    colours = ["r","b","g","c","m"]
    for i, wave in enumerate(position_matrix):
        for position in wave:
            plt.axvline(x = position, color = colours[i], label = 'axvline')

plt.figure(figsize=[10,5])
    
t, ecg_signal, position_matrix = ecg_signal_generator(70, 4)
noisy_ecg = add_noise(ecg_signal, 0.004)

plot_ecg_waves(position_matrix)

plt.plot(t, noisy_ecg, color = "k")
plt.show()


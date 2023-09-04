import numpy as np
import matplotlib.pyplot as plt

def ecg_signal_generator(heartbeat, duration):
    # Sampling frequency
    fs = 500

    # Duration of a single beat
    beat_duration = 60 / heartbeat  # Rough duration for one heart beat

    # Define time array for a single beat
    t_single = np.linspace(0, beat_duration, int(fs*beat_duration), endpoint=False)

    # Gaussian models for P, Q, R, S, T waves
    P = np.exp(-(t_single-0.15)**2/(2*0.01**2))
    Q = np.exp(-(t_single-0.2)**2/(2*0.005**2))
    R = np.exp(-(t_single-0.25)**2/(2*0.01**2))
    S = np.exp(-(t_single-0.3)**2/(2*0.005**2))
    T = np.exp(-(t_single-0.35)**2/(2*0.01**2))

    # Construct a single heartbeat waveform
    single_beat = 0.3*P - 0.2*Q + R - 0.3*S + 0.2*T

    # Calculate the number of beats in the duration
    num_beats = int(duration / beat_duration)

    # Create the ECG signal by tiling the single beat
    ecg_signal = np.tile(single_beat, num_beats)[:int(fs*duration)]
    
    # Adjust the time vector t to match the length of ecg_signal
    t = np.linspace(0, duration, len(ecg_signal), endpoint=False)

    return t, ecg_signal

t, ecg_signal = ecg_signal_generator(30, 10)

plt.figure(figsize=[10,5])
plt.plot(t, ecg_signal)
plt.show()
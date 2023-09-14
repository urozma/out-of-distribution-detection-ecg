import numpy as np
import matplotlib.pyplot as plt
import random


def ecg_signal_generator(heartbeat, duration):
    # Sampling frequency
    fs = 300

    # Duration of a single beat
    beat_duration = 60 / heartbeat  # Rough duration for one heart beat

    # Define time array for a single beat
    t_single = np.linspace(0, beat_duration, int(fs*beat_duration), endpoint=False)

    # For whole duration
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)

    # Timestamps where the means of the waves will be
    means = [0.15, 0.35, 0.4, 0.45, 0.6]

    # Gaussian models for P, Q, R, S, T waves
    P = np.exp(-(t_single-means[0])**2/(2*0.03**2))
    Q = np.exp(-(t_single-means[1])**2/(2*0.005**2))
    R = np.exp(-(t_single-means[2])**2/(2*0.02**2))
    S = np.exp(-(t_single-means[3])**2/(2*0.005**2))
    T = np.exp(-(t_single-means[4])**2/(2*0.02**2))

    # Turn timestamp array into indices
    def get_closest_index(time, t):
        return min(range(len(t)), key=lambda i: abs(t[i]-time))

    # Calculate the number of beats in the duration
    num_beats = int(duration / beat_duration)
    
    # Create a target array for the first beat
    target_single = [0] * len(t_single)
    for i, a in enumerate(means):
        index = get_closest_index(a,t)
        target_single[index] = i+1

    beat_duration_index = get_closest_index(beat_duration, t)
    index = 0

    ecg_signal = []
    target = []

    for i in range(num_beats+1):
        # Construct a single heartbeat waveform
        P_height = 0.001*random.randrange(5,25)
        R_height = 0.01*random.randrange(23,26)
        Q_height = 0.0001*random.randrange(int(1500*R_height), int(2000*R_height))
        S_height = 0.001*random.randrange(20,30)
        T_height = 0.0001*random.randrange(int(1200*R_height), int(2000*R_height))


        single_beat = P_height*P - Q_height*Q + R_height*R - S_height*S + T_height*T
        ecg_signal.extend(single_beat)
        target.extend(target_single)

    # To match arrays
    difference = len(t)-len(ecg_signal)
    ecg_signal = ecg_signal[:difference]
    target = target[:difference]

    # Add noise to signal
    def add_noise(signal, scale):
        noise = np.random.normal(0, scale, len(signal))
        noisy_signal = signal + noise
        return noisy_signal
    
    noisy_ecg = add_noise(ecg_signal, 0.004)
    return t, noisy_ecg, target

# Create a dataset with the created ECG function
def dataset(samples):
    signal_matrix = []
    target_matrix = []

    for s in range(samples):
        normal_heartbeat = random.randrange(60,100)
        t, ecg_signal, target = ecg_signal_generator(normal_heartbeat, 5)

        signal_matrix.append(ecg_signal)
        target_matrix.append(target)

    signal_matrix = np.array(signal_matrix)
    target_matrix = np.array(target_matrix)

    return t, signal_matrix, target_matrix

# Split dataset into pairs of training, validation and test data with their target values
def split_dataset(dataset, target, split):
    train_data = dataset[:split[0]]
    train_target = target[:split[0]]
    train_tuple = (train_data, train_target)

    val_data = dataset[split[0]:split[0]+split[1]]
    val_target = target[split[0]:split[0]+split[1]]
    val_tuple = (val_data, val_target)

    test_data = dataset[split[0]+split[1]:split[0]+split[1]+split[2]]
    test_target = target[split[0]+split[1]:split[0]+split[1]+split[2]]
    test_tuple = (test_data, test_target)

    return train_tuple, val_tuple, test_tuple

t, signal_matrix, target_matrix = dataset(1600)

train_tuple, val_tuple, test_tuple = split_dataset(signal_matrix, target_matrix, [1000, 100, 500])
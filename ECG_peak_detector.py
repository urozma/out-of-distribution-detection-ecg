import numpy as np
import matplotlib.pyplot as plt
from signal_generator import accuracy
from ECG_signal_generator import dataset
from ECG_signal_generator import split_dataset
from signal_generator import accuracy

def detect_R(dataset):
    peak_matrix = []
    for sample in dataset:
        peaks = [0]
        for i in range(1, len(sample)-1):
            if sample[i] > max(sample)*0.3 and sample[i-1] < sample[i] and sample[i+1] < sample[i]:
                peaks.append(1)
            else:
                peaks.append(0)
        peaks.append(0)
        peak_matrix.append(peaks)
    return peak_matrix

plt.figure(figsize=[10,5])
    
#t, ecg_signal, position_matrix = ecg_signal_generator(70, 4)
#noisy_ecg = add_noise(ecg_signal, 0.004)

t, signal_matrix, target_matrix = dataset(16)
train_tuple, val_tuple, test_tuple = split_dataset(signal_matrix, target_matrix, [10, 1, 5])

detected_peaks = detect_R(train_tuple[0])

# Calculate accuracy
def accuracy(target_matrix, detected_matrix):
    TP = TN = FP = FN = 0
    for i in range(target_matrix.shape[0]):
        for j in range(target_matrix.shape[1]):
            if target_matrix[i][j] == 1 and detected_matrix[i][j] == 1:
                TP += 1
            elif target_matrix[i][j] != 1 and detected_matrix[i][j] != 1:
                TN += 1
            elif target_matrix[i][j] != 1 and detected_matrix[i][j] == 1:
                FP += 1
            elif target_matrix[i][j] == 1 and detected_matrix[i][j] != 1:
                FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


accuracy = accuracy(train_tuple[1], detected_peaks)
print("Accuracy:", accuracy)


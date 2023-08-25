import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.signal


# Create an array with a certain amount of peaks
def peaks_vector(length, peaks):
    vector = np.zeros(length)
    # Randomly define peak position
    vector[np.random.choice(length, peaks, replace=False)] = 1

    # Randomly define peak height
    for i, peak in enumerate(vector):
        if peak == 1:       
            vector[i] = random.randrange(20, 50)

    return vector


# Add gaussian kernel to create a pseudo signal
def gaussian_kernel(signal, mean, std):
    # Define gaussian kernel
    kernel_size = 15  #len(signal)
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    kernel = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    kernel = kernel / np.sum(kernel)

    # Convolve gaussian kernel with signal
    pseudo_signal = np.convolve(signal, kernel, mode='same')
    
    return pseudo_signal


# Add noise to a signal
def add_noise(signal, scale):
    noise = np.random.normal(0, scale, len(signal))
    noisy_signal = signal + noise
    return noisy_signal


# Test for five peaks
original_signal = peaks_vector(1000, 5)
pseudo_signal = gaussian_kernel(original_signal, 0, 1)
noisy_signal = add_noise(pseudo_signal, 0.1)

#plt.figure()
#plt.plot(range(len(noisy_signal)),noisy_signal)
#plt.show()


# Create a matrix with the previous functions
def matrix(samples, length, peaks, mean, std):
    matrix = []

    for s in range(samples):
        original_signal = peaks_vector(length, peaks)
        pseudo_signal = gaussian_kernel(original_signal, mean, std)
        noisy_signal = add_noise(pseudo_signal, 0.1)
        matrix.append(noisy_signal)

    matrix = np.array(matrix)
    return matrix

matrix = matrix(4, 1000, 5, 0, 5)


# Detect peaks and create a matrix with their indices
def detect_peaks(matrix):
    peak_matrix = []
    for sample in matrix:
        peaks, _ = scipy.signal.find_peaks(sample, prominence=(0.8, None))
        peak_matrix.append(peaks)
    return peak_matrix


# Plot the matrix through subplots and mark the peaks
def plot_matrix(matrix):
    peak_matrix = detect_peaks(matrix)

    plot = plt.figure()
    
    for i, sample in enumerate(matrix):
        x = np.array(range(len(sample)))
        peaks = peak_matrix[i]

        plt.subplot(len(matrix), 1, i+1)
        plt.plot(x, sample)
        plt.scatter(x[peaks], sample[peaks], marker='D', s=30, color='red')

    return plot


plot_matrix(matrix)
plt.show()

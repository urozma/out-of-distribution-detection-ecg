import numpy as np
import random
import matplotlib.pyplot as plt


def peaks_vector(length, peaks):
    vector = np.zeros(length)
    vector[np.random.choice(length, peaks, replace=False)] = 1

    for i, peak in enumerate(vector):
        if peak == 1:       
            vector[i] = random.randrange(20, 50)

    return vector


def gaussian_kernel(signal, mean, std):
    kernel_size = len(signal)

    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)

    kernel = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    kernel = kernel / np.sum(kernel)

    pseudo_signal = np.convolve(signal, kernel, mode='same')
    
    return pseudo_signal

def add_noise(signal, scale):
    noise = np.random.normal(0, scale, len(signal))
    noisy_signal = signal + noise
    return noisy_signal


# test for five peaks
original_signal = peaks_vector(1000, 5)
pseudo_signal = gaussian_kernel(original_signal, 0, 10)
noisy_signal = add_noise(pseudo_signal, 0.1)

#plt.figure()
#plt.plot(range(len(noisy_signal)),noisy_signal)
#plt.show()


def matrix(samples, length, peaks):
    matrix = np.empty((samples, length))
    for s in range(samples):
        original_signal = peaks_vector(length, peaks)
        pseudo_signal = gaussian_kernel(original_signal, 0, 1)
        noisy_signal = add_noise(pseudo_signal, 0.1)

        matrix = matrix + noisy_signal
    return matrix

matrix = matrix(10, 1000, 5)
print(matrix)
import numpy as np
import random
import matplotlib.pyplot as plt


def peaks_vector(length, peaks):
    vector = np.zeros(length)
    vector[np.random.choice(length, peaks, replace=False)] = 1

    for i, peak in enumerate(vector):
        if peak == 1:

            #peaks are randomly chosen between 20 and 50
            vector[i] = random.randrange(20, 50, 1)

    return vector


def gaussian_kernel(signal, mean, std):
    kernel_size = len(signal)

    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)

    kernel = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    kernel = kernel / np.sum(kernel)

    pseudo_signal = np.convolve(signal, kernel, mode='same')
    
    return pseudo_signal

# test for five peaks
original_signal = peaks_vector(1000, 5)
pseudo_signal = gaussian_kernel(original_signal, 0, 10)
print(pseudo_signal)

plt.figure()
plt.plot(range(len(pseudo_signal)),pseudo_signal)
plt.show()

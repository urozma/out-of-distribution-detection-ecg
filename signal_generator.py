import numpy as np
import random
import matplotlib.pyplot as plt


def matrix(samples, length, peaks, mean, std):

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
        kernel_size = 15

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


    # Create a matrix with the previous functions

    matrix = []
    ground_truth_matrix = []

    for s in range(samples):
        original_signal = peaks_vector(length, peaks)
        ground_truth_matrix.append(original_signal)

        pseudo_signal = gaussian_kernel(original_signal, mean, std)
        noisy_signal = add_noise(pseudo_signal, 0.1)
        matrix.append(noisy_signal)

    matrix = np.array(matrix)
    ground_truth_matrix = np.array(ground_truth_matrix)

    return matrix, ground_truth_matrix


# Detect peaks and create a matrix with their indices
def detect_peaks(matrix):
    peak_matrix = []
    for sample in matrix:
        peaks = []
        for i in range(1, len(sample)-1):
            if sample[i] > max(sample)*0.3 and sample[i-1] < sample[i] and sample[i+1] < sample[i]:
                peaks.append(i)
        peak_matrix.append(peaks)
    return peak_matrix


# Turn ground truth matrix with the whole signals in a matrix with just the peak indices
def ground_truth_indices(ground_truth_matrix):
    true_peak_matrix = []
    for sample in ground_truth_matrix:
        true_peaks = [i for i, point in enumerate(sample) if point > 0]
        true_peak_matrix.append(true_peaks)
    return true_peak_matrix


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


# Calculate the accuracy of the peak detection with the ground truth
def accuracy(true_peak_matrix, detected_peak_matrix, tolerance):
    accuracies = []

    for i, peaks in enumerate(detected_peak_matrix):
        true_positives = 0
        for peak in peaks:
            
            # Check if the peak is within tolerance of any ground truth peak
            if any(abs(peak - true_peak) <= tolerance for true_peak in true_peak_matrix[i]):
                true_positives += 1
        print(true_positives)
        false_positives = len(peaks) - true_positives
        false_negatives = len(true_peak_matrix[i]) - true_positives
        accuracy = true_positives / (true_positives + false_positives + false_negatives)
        accuracies.append(accuracy)
        print(accuracy)
    mean_accuracy = sum(accuracies) / len(accuracies)
    
    return mean_accuracy

# Test
#matrix, ground_truth_matrix = matrix(4, 1000, 5, 0, 5)
#detected_peak_matrix = detect_peaks(matrix)
#true_peak_matrix = ground_truth_indices(ground_truth_matrix)
#accuracy = accuracy(true_peak_matrix, detected_peak_matrix, 1)

#plot_matrix(matrix)
#plt.show()

#print("accuracy:",accuracy)

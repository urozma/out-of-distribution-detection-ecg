import numpy as np
import torch

def add_qs_peaks(signals, targets):
    targets = targets[0].squeeze().tolist()
    updated_targets = []
    for signal_tuple, target in zip(signals, targets):
        signal = signal_tuple[0][0].tolist()
        new_target = target
        i = 0
        while i < len(target) - 1:
            # Find P peak (1) followed by any R peak (3), ignoring subsequent adjacent R peaks
            if target[i] == 1:
                for j in range(i + 1, len(target)):
                    if target[j] == 3:
                        if j - i > 1:  # Ensure there's at least one point between P and R
                            q_values = signal[i + 1:j]
                            if len(q_values) > 0:  # Ensure segment is not empty
                                q_index = i + 1 + np.argmin(q_values).item()
                                if target[q_index] == 0:  # Ensure Q peak isn't already marked
                                    new_target[q_index] = 2

            # Find R peak (3) followed by any T peak (5), ignoring subsequent adjacent T peaks
            if target[i] == 3:
                for k in range(i + 1, len(target)):
                    if target[k] == 5:
                        if k - i > 1:  # Ensure there's at least one point between R and T
                            s_values = signal[i + 1:k]  # Extract the segment between R and T
                            if len(s_values) > 0:  # Ensure segment is not empty
                                s_index = i + 1 + np.argmin(s_values).item()
                                if target[s_index] == 0 or 2:
                                    new_target[s_index] = 4
                        break  # Stop after the first T peak is processed

            i += 1

        updated_targets.append(new_target)

    targets_tensor = [torch.tensor(updated_targets).unsqueeze(1)]
    return targets_tensor


def add_qs_peaks_modified(signals, targets):
    updated_targets = []
    for signal, target in zip(signals, targets):
        new_target = target
        i = 0
        while i < len(target) - 1:
            # Find P peak (1) followed by any R peak (3), ignoring subsequent adjacent R peaks
            if target[i] == 1:
                for j in range(i + 1, len(target)):
                    if target[j] == 3:
                        if j - i > 1:  # Ensure there's at least one point between P and R
                            q_values = signal[i + 1:j]
                            if len(q_values) > 0:  # Ensure segment is not empty
                                q_index = i + 1 + np.argmin(q_values).item()
                                if target[q_index] == 0:  # Ensure Q peak isn't already marked
                                    new_target[q_index] = 2

            # Find R peak (3) followed by any T peak (5), ignoring subsequent adjacent T peaks
            if target[i] == 3:
                for k in range(i + 1, len(target)):
                    if target[k] == 5:
                        if k - i > 1:  # Ensure there's at least one point between R and T
                            s_values = signal[i + 1:k]  # Extract the segment between R and T
                            if len(s_values) > 0:  # Ensure segment is not empty
                                s_index = i + 1 + np.argmin(s_values).item()
                                if target[s_index] == 0 or 2:
                                    new_target[s_index] = 4
                        break  # Stop after the first T peak is processed

            i += 1

        updated_targets.append(new_target)

    targets_array = np.array(updated_targets)
    return targets_array
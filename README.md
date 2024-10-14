# Out-of-Distribution Detection on ECG signals via Knowledge Transfer from Synthetic Data

This project uses machine learning to explore how well it's possible to automate the analysis of ECG signals through out-of-distribution detection and transfer learning with limited data availability.

## Features

- **U-Net**: The network used is an adapted U-Net architecture to improve ECG pattern interpretation.
![The exact U-Net architecture used](images/unet.png)

- **Out-of-Distribution Detection**: The detection of ECG anomalies is attempted using models trained exclusively on healthy data and
  out-of-distribution detection to determine whether the looked-at signal shows a significant difference from a healthy one.
- **Synthetic Data Pretraining**: Pretrain on synthetically generated data to enhance performance when finetuned on the few real ECG signals available.

## Getting Started

### Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8+
- PyTorch or TensorFlow (depending on the model framework used)
- Pipenv (for managing project dependencies)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/urozma/out-of-distribution-detection-ecg.git
   cd out-of-distribution-detection-ecg
   ```

2. Set up the project environment using Pipenv:

   ```bash
   pipenv install
   ```

   This will create a virtual environment and install all the required dependencies for the project.

# ANN for Logic Gates: XOR, AND, OR

This sub-directory contains a Jupyter notebook implementing a simple Artificial Neural Network (ANN) to learn binary logic operations (XOR, AND, OR) using Gradient Descent (GD) and Stochastic Gradient Descent (SGD) optimization. The network is a single-hidden-layer feedforward neural network with sigmoid activation, trained on datasets with added Gaussian noise.

## Overview

The code defines an ANN class with forward and backward propagation, using squared error loss. It generates synthetic datasets for XOR, AND, and OR operations, splits them into training (80%) and testing (20%) sets, and trains the model under varying conditions:
- Dataset sizes: 100, 500, 1000 samples.
- Optimization methods: Full-batch GD and mini-batch SGD with batch sizes 8, 16, 32.
- Epochs: 5000 (with logging every 1000 epochs).

Key components include:
- Sigmoid activation and its derivative.
- Weight and bias updates via backpropagation.
- Dataset generation with optional Gaussian noise (standard deviation 0.1).
- Evaluation metrics: Mean squared error loss and accuracy (threshold at 0.5).

The experiments demonstrate how SGD often converges faster and achieves perfect accuracy (1.0000) on both training and test sets for larger datasets, while GD sometimes plateaus at lower accuracies (e.g., ~0.5 for XOR with 500+ samples).

## Requirements

- Python 3.x
- Libraries: NumPy, Matplotlib

Install dependencies via:
```
bash
pip install numpy matplotlib
```

## Usage

1. Open the notebook in JupyterLab or Jupyter Notebook.
2. Run all cells to execute the experiments. The code generates datasets, trains models, and prints loss/accuracy for each configuration.
3. Modify parameters in the last cell (e.g., `operations`, `n_samples_list`, `batch_size_list`, `epochs`) to customize runs.

Example output snippet for XOR with 100 samples (GD):
- Epoch 0 - Loss: 0.1255, Accuracy: 0.5500
- Final GD Training Loss: 0.0000, Training Accuracy: 1.0000
- Test Loss: 0.0003, Test Accuracy: 1.0000

## Key Observations from Experiments

- **XOR (Non-linearly separable)**: GD struggles with larger datasets (accuracy ~0.5), but SGD achieves 1.0000 accuracy across all sizes and batch sizes.
- **AND/OR (Linearly separable)**: Both GD and SGD perform well, but GD often stabilizes at ~0.7-0.8 accuracy, while SGD reaches 1.0000 quickly.
- Smaller batch sizes (e.g., 8) in SGD lead to faster convergence in noisy datasets.
- Increasing sample size improves generalization, with test accuracies reaching 1.0000 in most SGD cases.

For full code and outputs, refer to the attached notebook.

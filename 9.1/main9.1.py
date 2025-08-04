import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Tutorial 9.1 Full Script: Parts (a)–(i)

# Parameters
fs = 1000                    # sampling frequency (Hz)
duration = 10                # duration (seconds)
t = np.arange(0, duration, 1/fs)
f = 0.5                      # oscillation frequency (Hz)
A, B = 20, 10                # amplitudes for sine and cosine

# Part 2
f_A = 1 # Hz
f_B = 0.5 # Hz

# Part (a): Generate oscillating inputs
I_minus = A * np.sin(2 * np.pi * f_A * t)
I_cos   = B * np.cos(2 * np.pi * f_B * t)

# Part (b): Simulate population firing rates
n_neurons = 50
I_static = 50
W_static  = np.random.randn(n_neurons)
W_minus   = np.random.randn(n_neurons)
W_cos     = np.random.randn(n_neurons)
sigma     = 10             # noise scale

# Baseline plus modulations
base_rate = 100 + W_static * I_static
rate = (base_rate
        + np.outer(I_minus, W_minus)
        + np.outer(I_cos, W_cos)
        + sigma * np.random.randn(len(t), n_neurons))

# Part (c): Perform PCA
pca   = PCA()
SCORE = pca.fit_transform(rate)    # time × components
COEFF = pca.components_            # components × neurons
EXPLAINED = pca.explained_variance_ratio_
MU       = pca.mean_               # neuron means

# Part (d): Plot two columnns of COEFF against input weights
plt.figure()
plt.scatter(W_minus, COEFF[0, :])
plt.xlabel('Wi(A) weights')
plt.ylabel('PC1 loadings')
plt.title('PC1 Loadings vs Wi(A)')
plt.show()

plt.figure()
plt.scatter(W_cos, COEFF[1, :])
plt.xlabel('Wi(B) weights')
plt.ylabel('PC2 loadings')
plt.title('PC2 Loadings vs Wi(B)')
plt.show()

# Part (e): Explained variance ratio
plt.figure()
plt.plot(np.arange(1, len(EXPLAINED)+1), EXPLAINED)
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('Variance Explained by Each PC')
plt.show()

# Part (f): Reconstruct using first two PCs (denoising)
S2 = SCORE[:, :2]            # time × 2
C2 = COEFF[:2, :]            # 2 × neurons
reconstructed = S2.dot(C2) + MU  # time × neurons

# Part (g): Plot original vs denoised for example neurons 0 and 1
example_neurons = [0, 1]
for idx in example_neurons:
    plt.figure()
    plt.plot(t, rate[:, idx], label='Original')
    plt.plot(t, reconstructed[:, idx], label='Denoised')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate')
    plt.title(f'Neuron {idx}: Original vs Denoised')
    plt.legend()
    plt.show()

# Part (h): Rate–rate scatter before and after denoising
n0, n1 = example_neurons
plt.figure()
plt.scatter(rate[:, n0], rate[:, n1])
plt.xlabel(f'Neuron {n0} rate')
plt.ylabel(f'Neuron {n1} rate')
plt.title('Rate–Rate Scatter: Original')
plt.show()

plt.figure()
plt.scatter(reconstructed[:, n0], reconstructed[:, n1])
plt.xlabel(f'Neuron {n0} rate')
plt.ylabel(f'Neuron {n1} rate')
plt.title('Rate–Rate Scatter: Denoised')
plt.show()

# Part (i): Time courses of PC1 and PC2
plt.figure()
plt.plot(t, SCORE[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('PC1 score')
plt.title('Time Course of PC1')
plt.show()

plt.figure()
plt.plot(t, SCORE[:, 1])
plt.xlabel('Time (s)')
plt.ylabel('PC2 score')
plt.title('Time Course of PC2')
plt.show()

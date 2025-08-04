import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Tutorial 9.1 Part 3 Full Script: Repeating parts (a)-(i) with new inputs

# Parameters
fs = 1000                    # sampling frequency (Hz)
duration = 10                # duration (seconds)
t = np.arange(0, duration, 1/fs)
f = 0.5                      # oscillation frequency for transient input
A, B = 20, 10                # amplitudes

# Part 3a: Define new inputs
I1 = A * t                               # slowly ramping current
I2 = B * np.sin(2 * np.pi * f * t)      # transient sine
I2[(t <= 4) | (t >= 5)] = 0              # active only between 4s and 5s

# Part 3b: Simulate population firing rates
n_neurons = 50
I_static = 50
W_static = np.random.randn(n_neurons)
W1 = np.random.randn(n_neurons)
W2 = np.random.randn(n_neurons)
sigma = 10

base_rate = 100 + W_static * I_static
rate = (base_rate
        + np.outer(I1, W1)
        + np.outer(I2, W2)
        + sigma * np.random.randn(len(t), n_neurons))

# Part 3c: PCA
pca = PCA()
SCORE = pca.fit_transform(rate)    # time × components
COEFF = pca.components_            # components × neurons
EXPLAINED = pca.explained_variance_ratio_
MU = pca.mean_                     # neuron means

# Part 3d: PC1 vs W1 and PC2 vs W2
plt.figure()
plt.scatter(W1, COEFF[0, :])
plt.xlabel('W1 weights')
plt.ylabel('PC1 loadings')
plt.title('PC1 Loadings vs W1')
plt.show()

plt.figure()
plt.scatter(W2, COEFF[1, :])
plt.xlabel('W2 weights')
plt.ylabel('PC2 loadings')
plt.title('PC2 Loadings vs W2')
plt.show()

# Part 3e: Explained variance
plt.figure()
plt.plot(np.arange(1, len(EXPLAINED) + 1), EXPLAINED, marker='o')
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('Variance Explained by Each PC')
plt.show()

# Part 3i: Time courses of PC1 and PC2
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

# Part 3f: Denoising using first two PCs
S2 = SCORE[:, :2]            # time × 2
C2 = COEFF[:2, :]            # 2 × neurons
reconstructed = S2.dot(C2) + MU  # time × neurons

# Part 3g: Original vs. denoised for two example neurons
for idx in [0, 1]:
    plt.figure()
    plt.plot(t, rate[:, idx], label='Original')
    plt.plot(t, reconstructed[:, idx], label='Denoised')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate')
    plt.title(f'Neuron {idx}: Original vs Denoised')
    plt.legend()
    plt.show()

# Part 3h: Rate–rate scatter before and after denoising
n0, n1 = 0, 1
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

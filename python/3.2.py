import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Global parameters for the AELIF model
G_L = 10e-9               # Leak conductance (S)
C = 100e-12               # Capacitance (F)
E_L = -70e-3              # Leak potential (V)
V_Thresh = -50e-3         # Threshold potential (V)
V_Reset = -80e-3          # Reset potential (V)
delta_th = 2e-3           # Threshold shift factor (V)
tauw = 150e-3             # Adaptation time constant (s)
a = 2e-9                  # Adaptation recovery (S)
b = 0e-9                  # Adaptation strength (A)

# Simulation parameters
dt = 1e-2                 # Time step (s)
tmax = 100                # Maximum simulation time (s)
tvector = np.arange(0, tmax, dt)

# Generate stochastic input current
sigma = 20e-12  # Noise level (A·s^0.5)
Iapp = np.random.normal(0.2e-9, sigma / np.sqrt(dt), size=len(tvector))

@jit(nopython=True)
def run_adex_model(tvector, I, b):
    """Simulate the Adaptive Exponential Leaky Integrate-and-Fire model."""
    v = np.full_like(tvector, E_L)  # Membrane potential
    w = np.zeros_like(tvector)      # Adaptation variable
    spikes = np.zeros_like(tvector)  # Spike times
    
    for j in range(len(tvector) - 1):
        if v[j] > V_Thresh:        # Spike condition
            v[j] = V_Reset         # Reset membrane potential
            w[j] += b               # Increment adaptation variable
            spikes[j] = 1           # Record spike
        
        # Update membrane potential
        dv = (G_L * (E_L - v[j] + delta_th * np.exp((v[j] - V_Thresh) / delta_th))
              - w[j] + I[j]) / C
        v[j + 1] = v[j] + dt * dv
        
        # Update adaptation variable
        dw = (a * (v[j] - E_L) - w[j]) / tauw
        w[j + 1] = w[j] + dt * dw
    
    return v, w, spikes

# Run simulation
v, w, spikes = run_adex_model(tvector, Iapp, b)

# Compute Inter-Spike Intervals (ISIs)
spike_times = tvector[spikes == 1]
ISIs = np.diff(spike_times)

# Plot ISI histogram
plt.figure(figsize=(8, 5))
plt.hist(ISIs, bins=25, edgecolor='black', alpha=0.7)
plt.xlabel('Inter-Spike Interval (s)')
plt.ylabel('Count')
plt.title('Histogram of ISIs')
plt.show()

# Compute CV of ISIs
CV_ISI = np.std(ISIs) / np.mean(ISIs)
print(f'Coefficient of Variation (CV) of ISIs: {CV_ISI:.3f}')

# Compute Fano factor in consecutive 100ms windows
window_size = 0.1  # 100ms
num_windows = int(tmax / window_size)
spike_counts_100ms = np.array([np.sum(spikes[int(i * window_size / dt): int((i + 1) * window_size / dt)]) for i in range(num_windows)])
Fano_factor_100ms = np.var(spike_counts_100ms) / np.mean(spike_counts_100ms)
print(f'Fano Factor in 100ms windows: {Fano_factor_100ms:.3f}')

# Compute Fano factor over different time windows
window_sizes = np.logspace(-2, 0, 10)  # From 10ms to 1s
fano_factors = []

for win in window_sizes:
    num_windows = int(tmax / win)
    spike_counts = np.array([np.sum(spikes[int(i * win / dt): int((i + 1) * win / dt)]) for i in range(num_windows)])
    fano_factors.append(np.var(spike_counts) / np.mean(spike_counts))

# Plot Fano factor vs window size
plt.figure(figsize=(8, 5))
plt.plot(window_sizes, fano_factors, marker='o', linestyle='-', label='Fano Factor')
plt.xscale('log')
plt.xlabel('Window size (s)')
plt.ylabel('Fano Factor')
plt.title('Fano Factor vs Window Size')
plt.legend()
plt.show()


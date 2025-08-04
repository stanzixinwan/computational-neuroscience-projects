import numpy as np
import matplotlib.pyplot as plt

# Parameters (in base units)
V_th = -50e-3      # V
V_reset = -80e-3   # V
sigma_V = 1e-3   # V
tau = 3e-3       # s

E_L = -70e-3       # V
E_I = -65e-3      # V
E_E = 0e-3         # V
G_L = 50e-12      # S

# Connection weights
W_EE = 25e-9      # S
W_EI = 4e-9       # S
W_IE = 800e-9     # S

# Time constants
tau_E = 2e-3     # s
tau_I = 5e-3     # s
alpha = 0.2

# Simulation settings
T = 2.5           # total simulation time (s)
dt = 0.1e-3       # time step (s)
steps = int(T / dt)

# Time vector
t = np.arange(0, T, dt)

# Initialize variables
r1 = np.zeros(steps)
r2 = np.zeros(steps)
s_E = np.zeros(steps)
s_I = np.zeros(steps)

# Input conductances (G_in)
G_Ein1 = 1e-9     # S
G_Ein2 = 0e-9        # S

# Helper function: steady-state voltage
def V_ss(G_L, G_I, G_E, E_L, E_I, E_E):
    G_total = G_L + G_I + G_E
    return (G_L * E_L + G_I * E_I + G_E * E_E) / G_total

# Firing rate function
def f(V_ss):
    if np.isclose(V_ss, V_th):
        return sigma_V / (tau * (V_th - V_reset))
    else:
        numerator = V_ss - V_th
        denominator = tau * (V_th - V_reset) * (1 - np.exp(-(V_ss - V_th) / sigma_V))
        return numerator / denominator

# Main simulation loop
for i in range(1, steps):
    # Synaptic conductances
    G_E1 = W_EE * s_E[i-1] + G_Ein1
    G_I1 = W_IE * s_I[i-1]

    G_E2 = W_EI * s_E[i-1] + G_Ein2
    G_I2 = 0

    # V_ss for both units
    V_ss1 = V_ss(G_L, G_I1, G_E1, E_L, E_I, E_E)
    V_ss2 = V_ss(G_L, G_I2, G_E2, E_L, E_I, E_E)

    # Rate updates
    dr1 = (-r1[i-1] + f(V_ss1)) / tau
    dr2 = (-r2[i-1] + f(V_ss2)) / tau
    r1[i] = r1[i-1] + dt * dr1
    r2[i] = r2[i-1] + dt * dr2

    # Synaptic variable updates
    ds_E = (-s_E[i-1] + alpha * r1[i-1] * (1 - s_E[i-1])) / tau_E
    ds_I = (-s_I[i-1] + alpha * r2[i-1] * (1 - s_I[i-1])) / tau_I
    s_E[i] = s_E[i-1] + dt * ds_E
    s_I[i] = s_I[i-1] + dt * ds_I

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, r1, label='Excitatory Unit Firing Rate (r1)')
plt.plot(t, r2, label='Inhibitory Unit Firing Rate (r2)')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('E-I Coupled Oscillator')
plt.legend()
plt.tight_layout()
plt.show()

# Step 2: Automatic frequency detection for r2 (inhibitory unit)
transient_index = int(0.5 / dt)   # index corresponding to 0.5s
r2_post = r2[transient_index:]    # exclude transient
t_post = t[transient_index:]

# Define thresholds
r2_max = np.max(r2_post)
r2_min = np.min(r2_post)
upper_thresh = r2_min + 0.8 * (r2_max - r2_min)
lower_thresh = r2_min + 0.2 * (r2_max - r2_min)

# Detect threshold crossings
crossings = []
above = r2_post[0] > upper_thresh

for i in range(1, len(r2_post)):
    if not above and r2_post[i] > upper_thresh:
        crossings.append(t_post[i])
        above = True
    elif r2_post[i] < lower_thresh:
        above = False

# Compute period and frequency from first and last crossings
if len(crossings) >= 2:
    first_cross = crossings[0]
    last_cross = crossings[-1]
    num_cycles = len(crossings) - 1  # number of full oscillations between first and last

    total_time = last_cross - first_cross
    avg_period = total_time / num_cycles
    frequency = 1 / avg_period

    print(f"Number of Oscillations: {num_cycles}")
    print(f"Time between 1st and last crossing: {total_time:.4f} s")
    print(f"Average Period: {avg_period:.4f} s")
    print(f"Estimated Frequency: {frequency:.2f} Hz")
else:
    print("Not enough crossings to calculate period and frequency.")


# Step 3: Compute power spectrum P(f) using cosine and sine overlaps
# Use same truncated portion as Step 2
r1_post = r1[transient_index:]  # truncate to periodic portion
t_post = t[transient_index:]    # time vector after transient

# Frequency vector from 0 Hz to 100 Hz in 0.5 Hz steps
frequencies = np.arange(0, 100.5, 0.5)
A = []
B = []

for f in frequencies:
    sin_wave = np.sin(2 * np.pi * f * t_post)
    cos_wave = np.cos(2 * np.pi * f * t_post)
    A_f = np.mean(sin_wave * r1_post)
    B_f = np.mean(cos_wave * r1_post)
    A.append(A_f)
    B.append(B_f)

# Power spectrum: P(f) = A^2 + B^2
P = np.square(A) + np.square(B)

# Plot the power spectrum
plt.figure(figsize=(8, 4))
plt.plot(frequencies, P)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power P(f)")
plt.title("Power Spectrum of r1(t)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Ignore f = 0, then find dominant frequency
f_max_index = np.argmax(P[1:]) + 1  # skip f = 0
dominant_freq = frequencies[f_max_index]
print(f"Dominant Frequency from P(f): {dominant_freq:.2f} Hz")

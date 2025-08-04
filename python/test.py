import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
dt = 0.0001  # Time step in ms
total_time = 6  # Total simulation time in ms
time = np.arange(0, total_time, dt)

# Neuron parameters
C = 1e-9  # Capacitance in F
R = 10e6  # Membrane resistance in Ohm
E = -70e-3  # Resting potential in V
V_th = -54e-3  # Threshold voltage in V
V_reset = -80e-3  # Reset voltage in V

# Synaptic parameters
E_rev = -70e-3  # Inhibitory reversal potential in V
G_syn = 1e-6  # Synaptic conductance in S
T_syn = 10e-3  # Synaptic time constant in ms
p_R = 1  # No synaptic depression (Part A)
D1, D2 = 1, 1  # Depression variables (no depression in Part A)
I_b = 2e-9

# Noise parameters
sigma = 0  # Set later for noise cases
np.random.seed(42)

# Initialize variables
V1, V2 = np.full(len(time), E), np.full(len(time), E)
s1, s2 = np.zeros(len(time)), np.zeros(len(time))
spike_times = []

# External current
I_app = np.full(len(time), I_b)
I_app[:1000] += 3e-9  # Apply extra current to neuron 1 for first 100 ms
I_app[30000:31000] += 3e-9  # Apply extra current to neuron 2 at midpoint

# Simulation loop
for t in range(1, len(time)):
    # Update synaptic gating variables
    ds1 = (-s1[t-1] / T_syn) * dt
    ds2 = (-s2[t-1] / T_syn) * dt
    s1[t] = s1[t-1] + ds1
    s2[t] = s2[t-1] + ds2

    # Update membrane potentials
    dV1 = ((E - V1[t-1]) / (R*C) + G_syn * s2[t-1] * (E_rev - V1[t-1]) + I_app[t]/C) * dt
    dV2 = ((E - V2[t-1]) / (R*C) + G_syn * s1[t-1] * (E_rev - V2[t-1]) + I_app[t]/C) * dt
    
    if sigma > 0:
        dV1 += sigma * np.random.randn() * np.sqrt(dt)
        dV2 += sigma * np.random.randn() * np.sqrt(dt)
    
    V1[t] = V1[t-1] + dV1
    V2[t] = V2[t-1] + dV2
    
    # Check for spikes
    if V1[t] >= V_th:
        V1[t] = V_reset
        s1[t] = 1  # Trigger synaptic response
        spike_times.append(time[t])
    if V2[t] >= V_th:
        V2[t] = V_reset
        s2[t] = 1  # Trigger synaptic response
        spike_times.append(time[t])

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, V1 * 1e3, label="Neuron 1", color='blue')
plt.plot(time, V2 * 1e3, label="Neuron 2", color='red')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Bistability of Two Coupled LIF Neurons")
plt.show()

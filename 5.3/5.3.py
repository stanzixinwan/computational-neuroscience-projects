import numpy as np
import matplotlib.pyplot as plt

# Neuron parameters
C = 1e-9  # Capacitance in Farads
R = 10e6  # Membrane resistance in Ohms
E = -70e-3  # Resting potential in Volts
V_th = -54e-3  # Threshold voltage in Volts
V_reset = -80e-3  # Reset voltage in Volts

# Synaptic parameters
E12_rev = -70e-3  
E21_rev = -70e-3  
G12 = 1e-6  
G21 = 1e-6
T_syn = 10e-3  # Synaptic time constant in seconds
T_D = 0.2  # Depression recovery time constant
I_b = 2e-9  # Baseline current in Amperes
p_R = 1  # No synaptic depression (Part A)

# Noise parameters
sigma = 0
np.random.seed(42)

def double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression):
    # Initialize variables
    V1, V2 = np.zeros(len(time)), np.zeros(len(time))
    V1[0] = E
    V2[0] = E
    s1, s2 = np.zeros(len(time)), np.zeros(len(time))
    D1, D2 = np.ones(len(time)), np.ones(len(time))  # Depression variables start at 1
    spike_times = []

    for t in range(1, len(time)):
        # Update synaptic gating variables (exponential decay)
        s1[t] = s1[t-1] + (-s1[t-1] / T_syn) * dt
        s2[t] = s2[t-1] + (-s2[t-1] / T_syn) * dt

        # Correct synaptic depression recovery equation
        D1[t] = D1[t-1] + ((1 - D1[t-1]) / T_D) * dt
        D2[t] = D2[t-1] + ((1 - D2[t-1]) / T_D) * dt

        # Update membrane potentials with noise
        noise1 = sigma * np.random.randn() / np.sqrt(dt)
        noise2 = sigma * np.random.randn() / np.sqrt(dt)
        dV1 = ((E - V1[t-1]) / R + G21 * s2[t-1] * (E21_rev - V1[t-1]) + I1_app[t-1] + noise1) * dt / C
        dV2 = ((E - V2[t-1]) / R + G12 * s1[t-1] * (E12_rev - V2[t-1]) + I2_app[t-1] + noise2) * dt / C

        V1[t] = V1[t-1] + dV1
        V2[t] = V2[t-1] + dV2

        # Check for spikes
        if V1[t] >= V_th:
            V1[t] = V_reset
            s1[t] = s1[t] + p_R * D1[t] * (1 - s1[t])
            if depression:
                D1[t] = D1[t] * (1 - p_R)
            spike_times.append(time[t])

        if V2[t] >= V_th:
            V2[t] = V_reset
            s2[t] = s2[t] + p_R * D2[t] * (1 - s2[t])
            if depression:
                D2[t] = D2[t] * (1 - p_R)
            spike_times.append(time[t])

    return s1, s2, V1, V2, spike_times


dt = 0.1e-3  # Time step in seconds
total_time = 6  # Total simulation time in seconds
time = np.arange(0, total_time, dt)

# External current
I1_app, I2_app = np.full(len(time), I_b), np.full(len(time), I_b)
I1_app[:1000] = 3e-9 + I_b  
I2_app[30000:31000] = 3e-9 + I_b 

# Stimulate the model
s1, s2, V1, V2, _ = double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression = False)

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(time, s1, label="s1 (Neuron 1)")
plt.plot(time, s2, label="s2 (Neuron 2)")
plt.xlabel("Time (s)")
plt.ylabel("Synaptic gating")
plt.legend()
plt.title("Synaptic Gating Variables")
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(time, V1 * 1e3, label="Neuron 1")
plt.plot(time, V2 * 1e3, label="Neuron 2")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Bistability of Two Coupled LIF Neurons")
plt.show()


# Define noise level
sigma = 5e-11  # 50 pA * s^(-1/2)

# Only baseline current, no additional transient currents
I1_app, I2_app = np.full(len(time), I_b), np.full(len(time), I_b)

# Simulation with noise
s1, s2, V1, V2, _ = double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression = False)

plt.figure(figsize=(10, 4))
plt.plot(time, s1, label="s1 (Neuron 1)")
plt.plot(time, s2, label="s2 (Neuron 2)")
plt.xlabel("Time (s)")
plt.ylabel("Synaptic gating")
plt.legend()
plt.title("Synaptic Gating Variables")
plt.show()

# Plot results with noise
plt.figure(figsize=(16, 8))
plt.plot(time, V1 * 1e3, label="Neuron 1 (with noise)")
plt.plot(time, V2 * 1e3, label="Neuron 2 (with noise)")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Bistability with Noise")
plt.show()


def switch(s1, s2, t):
    switch_times = []
    state = -1
    for i in range(1, len(t)):
        if s1[i] > s1[i-1] and state != 1:
            switch_times.append(t[i])
            state = 1
        if s2[i] > s2[i-1] and state != 2:
            switch_times.append(t[i])
            state = 2
    switch_times = np.array(switch_times)
    durations = np.diff(switch_times)

    dur1 = durations[::2]
    dur2 = durations[1::2]

    return dur1, dur2

# Extend simulation time for more state switches
total_time = 600  # Increase total simulation time to get more than 1000 switches
time = np.arange(0, total_time, dt)
I1_app, I2_app = np.full(len(time), I_b), np.full(len(time), I_b)

# Simulation loop with noise
s1, s2, V1, V2, _ = double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression = False)
dur1, dur2 = switch(s1, s2, time)

# Plot histogram of state durations
plt.figure(figsize=(8, 6))
plt.hist(dur1, bins=40, alpha=0.8, label='Neuron 1 switches')
plt.hist(dur2, bins=40, alpha=0.4, label='Neuron 2 switches')
plt.xlabel("State Duration (s)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


dt = 0.1e-3  # Time step in seconds
total_time = 6  # Total simulation time in seconds
time = np.arange(0, total_time, dt)

# External current
I1_app, I2_app = np.full(len(time), I_b), np.full(len(time), I_b)
I1_app[:1000] = 3e-9 + I_b  
I2_app[30000:31000] = 3e-9 + I_b 

# Stimulate the model
p_R = 0.2
sigma = 0
s1, s2, V1, V2, _ = double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression = True)
# Plot results
plt.figure(figsize=(10, 4))
plt.plot(time, s1, label="s1 (Neuron 1)")
plt.plot(time, s2, label="s2 (Neuron 2)")
plt.xlabel("Time (s)")
plt.ylabel("Synaptic gating")
plt.legend()
plt.title("Synaptic Gating Variables (depression)")
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(time, V1 * 1e3, label="Neuron 1")
plt.plot(time, V2 * 1e3, label="Neuron 2")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("Bistability of Two Coupled LIF Neurons (depression)")
plt.show()


total_time = 600  # Increase total simulation time to get more than 1000 switches
time = np.arange(0, total_time, dt)
I1_app, I2_app = np.full(len(time), I_b), np.full(len(time), I_b)
sigma = 5e-12

# Simulation loop with noise
s1, s2, V1, V2, _ = double_LIF(time, dt, I1_app, I2_app, p_R, sigma, depression = True)
dur1, dur2 = switch(s1, s2, time)

# Plot histogram of state durations
plt.figure(figsize=(8, 6))
plt.hist(dur1, bins=40, alpha=0.8, label='Neuron 1 switches')
plt.hist(dur2, bins=40, alpha=0.4, label='Neuron 2 switches')
plt.xlabel("State Duration (s)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
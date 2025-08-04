import numpy as np
import matplotlib.pyplot as plt

E_L = -70e-3 # Leak potential (V)
R_m = 5e6 # Membrane resistance (omega)
C_m = 2e-9 # Membrane capacitance (F)
V_th = -50e-3 # Spike threshold (V)
V_reset = -65e-3 # Reset potential (V)

# Global Parameters
dt = 0.0001  # Time-step
t_max = 2
t = np.arange(0, t_max, dt)  # Time vector

def LIF_neuron(I_app, add_noise=False, sigma_V=0, dt=0.1e-3):
    t = np.arange(0, t_max, dt)
    V = np.full_like(t, E_L)  # Initialize membrane potential
    spike_times = []  # Store spike times
    
    # Generate noise vector
    noise_vec = np.random.randn(len(t)) * sigma_V * np.sqrt(dt) if add_noise else np.zeros(len(t))

    for i in range(1, len(t)):
        dVdt = (E_L - V[i-1]) / R_m + I_app
        V[i] = V[i-1] + (dVdt / C_m) * dt + noise_vec[i]  # <--- Now applying noise!

        # Check for spike
        if V[i] >= V_th:
            V[i] = V_reset  # Reset after spike
            spike_times.append(t[i])

    return t, V, np.array(spike_times)



I_th = (V_th - E_L) / R_m

# Simulate LIF neuron for I slightly lower and higher than threshold
I_low = I_th * 0.9  # Below threshold
I_high = I_th * 1.1  # Above threshold

t, V_low, _ = LIF_neuron(I_low)
t, V_high, spikes_high = LIF_neuron(I_high)

# Plot membrane potential over time
plt.figure(figsize=(10, 4))
plt.plot(t[:2000], V_low[:2000], label=f'I_app = {I_low:.2e} A (No Spikes)', color='r')
plt.plot(t[:2000], V_high[:2000], label=f'I_app = {I_high:.2e} A (Spiking)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('LIF Neuron Response to Different Currents')
plt.legend()
plt.grid()
plt.show()

# Generate f-I curve
I_values = np.linspace(0, 2 * I_th, 10)
firing_rates = []

for I in I_values:
    _, _, spike_times = LIF_neuron(I)
    firing_rate = len(spike_times) / t_max  # Spikes per second (Hz)
    firing_rates.append(firing_rate)

# Plot f-I curve
plt.figure(figsize=(8, 5))
plt.plot(I_values * 1e9, firing_rates, 'o-', label='Simulated f-I Curve')
plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate vs. Injected Current (f-I Curve)')
plt.legend()
plt.grid()
plt.show()

def firing_rate_equation(I_app):
    if I_app <= I_th:
        return 0  # No firing below threshold
    return 1 / (R_m * C_m * np.log((I_app * R_m + E_L - V_reset) / (I_app * R_m + E_L - V_th)))

calculated_firing_rates = [firing_rate_equation(I) for I in I_values]

# Plot both simulated and theoretical f-I curves
plt.figure(figsize=(8, 5))
plt.plot(I_values * 1e9, firing_rates, 'o-', label='Simulated f-I Curve')
plt.plot(I_values * 1e9, calculated_firing_rates, 's--', label='Theoretical f-I Curve', color='g')
plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Comparison of Simulated vs. Theoretical f-I Curve')
plt.legend()
plt.grid()
plt.show()

# Test with larger noise levels to see a stronger effect
sigma_values = [0, 0.05, 0.1]  # Noise levels in V

plt.figure(figsize=(8, 5))

for sigma in sigma_values:
    firing_rates_noise = []
    for I in I_values:
        _, _, spike_times = LIF_neuron(I, add_noise=True, sigma_V=sigma)
        firing_rates_noise.append(len(spike_times) / t_max)
    
    plt.plot(I_values * 1e9, firing_rates_noise, 'o-', label=f'Noise: {sigma * 1e3:.1f} mV')

plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Effect of Noise on f-I Curve')
plt.legend()
plt.grid()
plt.show()

# Define different time steps for numerical stability test
dt_values = [0.1e-3, 0.05e-3, 0.01e-3]  # Original, half, and 1/10th of original dt

plt.figure(figsize=(8, 5))

for dt_test in dt_values:
    firing_rates_dt = []
    
    for I in I_values:
        _, _, spike_times = LIF_neuron(I, dt=dt_test)  # Run simulation with different dt
        firing_rate = len(spike_times) / t_max  # Compute firing rate
        firing_rates_dt.append(firing_rate)
    
    plt.plot(I_values * 1e9, firing_rates_dt, 'o-', label=f'dt = {dt_test * 1e3:.2f} ms')

plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Effect of Time Step on f-I Curve (Numerical Stability)')
plt.legend()
plt.grid()
plt.show()
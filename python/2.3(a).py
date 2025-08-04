import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C_m = 100e-12  # Capacitance (F)
E_L = -75e-3   # Leak potential (V)
V_th = -50e-3  # Threshold potential (V)
V_reset = -80e-3  # Reset potential (V)
R_m = 100e6    # Membrane resistance (Ohm)
E_K = -80e-3   # Potassium reversal potential (V)

tau_SRA = 200e-3  # Adaptation time constant (s)
deltaG_SRA = 1e-9  # Adaptation increment (S)

# Part (a) Simulation parameters
I_app = 500e-12  # Applied current (A)
t_max = 1.5  # Maximum simulation time (s)
dt = 1e-6  # Time step (s)
time = np.arange(0, t_max, dt)  # Time vector

# Define applied current for Part (a)
I = np.zeros_like(time)  # Baseline current (A)
I[(time >= 0.5) & (time <= 1.0)] = I_app  # Apply current from 0.5s to 1.0s

def LIF_neuron(I, time):
    """Simulates the LIF neuron with an adaptation current"""
    V = np.full_like(time, E_L)  # Membrane potential (V)
    G_SRA = np.zeros_like(time)  # Adaptation conductance (S)
    spike_times = []

    for i in range(1, len(time)):
        # Update adaptation conductance using exponential decay
        G_SRA[i] = G_SRA[i-1] * np.exp(-dt / tau_SRA)

        # Compute membrane potential update (with adaptation current)
        dV = ((E_L - V[i-1]) / R_m + G_SRA[i] * (E_K - V[i-1]) + I[i]) / C_m
        V[i] = V[i-1] + dt * dV

        # Check for spike condition
        if V[i] >= V_th:
            spike_times.append(time[i])  # Record spike time
            V[i-1] = 0.04  # Artificial spike before reset
            V[i] = V_reset  # Reset membrane potential
            G_SRA[i] += deltaG_SRA  # Increase adaptation conductance

    return I, V, G_SRA, spike_times

# Run Part (a) simulation
I, V, G_SRA, spike_times = LIF_neuron(I, time)

# Plot results for Part (a)
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, I * 1e9, 'k')
plt.ylabel('I$_{app}$ (nA)')
plt.xlim([0, t_max])
plt.ylim([-0.1, 1.1 * np.max(I) * 1e9])
plt.title('Applied Current')

plt.subplot(3, 1, 2)
plt.plot(time, V * 1e3, 'k')
plt.ylabel('V$_m$ (mV)')
plt.xlim([0, t_max])
plt.ylim([-100, 50])
plt.title('Membrane Potential')

plt.subplot(3, 1, 3)
plt.plot(time, G_SRA * 1e9, 'k')
plt.xlabel('Time (s)')
plt.ylabel('G$_{SRA}$ (nS)')
plt.xlim([0, t_max])
plt.title('Adaptation Conductance')

plt.tight_layout()
plt.show()


# Part (b) Simulation Parameters
t_max_b = 5.0  # Maximum simulation time (s)
dt_b = 1e-4  # Time step (s)
time_b = np.arange(0, t_max_b, dt_b)  # Time vector

def LIF_neuron_constant_I(I_app):
    """Simulates the LIF neuron for a constant applied current"""
    V = np.full_like(time_b, E_L)  # Membrane potential (V)
    G_SRA = np.zeros_like(time_b)  # Adaptation conductance (S)
    spike_times = []

    for i in range(1, len(time_b)):
        # Update adaptation conductance using exponential decay
        G_SRA[i] = G_SRA[i-1] * np.exp(-dt_b / tau_SRA)

        # Compute membrane potential update (with adaptation current)
        dV = ((E_L - V[i-1]) / R_m + G_SRA[i] * (E_K - V[i-1]) + I_app) / C_m
        V[i] = V[i-1] + dt_b * dV

        # Check for spike condition
        if V[i] >= V_th:
            spike_times.append(time_b[i])  # Record spike time
            V[i-1] = 0.04  # Artificial spike before reset
            V[i] = V_reset  # Reset membrane potential
            G_SRA[i] += deltaG_SRA  # Increase adaptation conductance

    return spike_times

# Define applied currents to test
I_values = np.linspace(0.2e-9, 0.350e-9, 20)  # From 0.2nA to 0.3nA (20 steps)
firing_rates = []
inverse_initial_ISI = []
single_spike_cases = []

# Run simulations for each applied current
for I_app in I_values:
    spike_times = LIF_neuron_constant_I(I_app)
    
    if len(spike_times) > 1:
        ISIs = np.diff(spike_times)  # Compute inter-spike intervals
        initial_ISI = ISIs[0]  # First inter-spike interval
        steady_state_ISI = np.mean(ISIs[-10:])  # Average of last 10 ISIs
        firing_rate = 1 / steady_state_ISI  # Steady-state firing rate (Hz)
        inverse_initial_ISI.append(1 / initial_ISI)  # 1/ISI(1)
    else:
        firing_rate = 0  # No sustained firing
        inverse_initial_ISI.append(0)  # No second spike
    
    firing_rates.append(firing_rate)

# Convert currents to nA for plotting
I_values_nA = I_values * 1e9

# Plot f-I curve
plt.figure(figsize=(10, 6))
plt.plot(I_values_nA, firing_rates, 'k-', label="Final Rate")  # Solid line for steady-state rate
plt.scatter(I_values_nA, inverse_initial_ISI, marker='x', color='k', label="1/ISI(1)")  # Crosses for 1/ISI(1)

plt.xlabel("I$_{app}$ (nA)")
plt.ylabel("Spike Rate (Hz)")
plt.title("f-I Curve of the LIF Neuron with Adaptation")
plt.legend()
plt.show()

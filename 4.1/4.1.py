import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define Hodgkin-Huxley parameters
Cm = 100e-12  # Membrane capacitance (F)
G_L = 30e-9   # Leak conductance (S)
G_Na_max = 12e-6  # Max sodium conductance (S)
G_K_max = 3.6e-6  # Max potassium conductance (S)
E_Na = 45e-3  # Sodium reversal potential (V)
E_K = -82e-3  # Potassium reversal potential (V)
E_L = -60e-3  # Leak reversal potential (V)

# Define rate equations for gating variables
def alpha_m(V):
    return (1e5 * (-V - 45e-3)) / (np.exp(100 * (-V - 45e-3)) - 1)

def beta_m(V):
    return 4e3 * np.exp((-V - 70e-3) / 18e-3)

def alpha_h(V):
    return 70 * np.exp(50 * (-V - 70e-3))

def beta_h(V):
    return 1e3 / (1 + np.exp(100 * (-V - 40e-3)))

def alpha_n(V):
    return (1e4 * (-V - 60e-3)) / (np.exp(100 * (-V - 60e-3)) - 1)

def beta_n(V):
    return 125 * np.exp((-V - 70e-3) / 80e-3)

# Differential equations for Hodgkin-Huxley model
def hodgkin_huxley(y, t, I_app):
    V, m, h, n = y
    dVdt = (G_L * (E_L - V) + G_Na_max * (m ** 3) * h * (E_Na - V) + G_K_max * (n ** 4) * (E_K - V) + I_app(t)) / Cm
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

# Time vector
t = np.linspace(0, 0.35, int(1.75e6))  # 0.35s duration, 0.02𝜇s time step

# Define applied current function for step input
def I_app(t):
    return 0.22e-9 if 0.1 <= t <= 0.2 else 0

# Initial conditions
V0 = -70.2e-3  # Membrane potential in Volts
m0, h0, n0 = 0, 0, 0  # Gating variables
initial_conditions = [V0, m0, h0, n0]

# Solve ODE
y = odeint(hodgkin_huxley, initial_conditions, t, args=(I_app,))
V, m, h, n = y.T

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t * 1e3, [I_app(time) * 1e9 for time in t], label="Applied Current")
ax[0].set_ylabel("Current (nA)")
ax[0].set_title("Applied Current")
ax[0].legend()

ax[1].plot(t * 1e3, V * 1e3, label="Membrane Potential")
ax[1].set_xlabel("Time (ms)")
ax[1].set_ylabel("Membrane Potential (mV)")
ax[1].set_title("Membrane Response to Step Current")
ax[1].legend()

plt.tight_layout()
plt.show()

# Define applied current function for pulse train
def I_app(t):
    baseline_current = 0.65e-9
    excitatory_current = 1e-9
    pulse_start = 0.1
    pulse_duration = 5e-3
    pulse_interval = 20e-3
    num_pulses = 10
    for i in range(num_pulses):
       if pulse_start <= t <= pulse_start + pulse_duration:
            return excitatory_current #Increase the excitatory current to 1nA for a 5ms pulse at the time point of 100ms.
       if pulse_start + i * (pulse_interval + pulse_duration) <= t <= pulse_start + i * (pulse_interval) + (i + 1) * pulse_duration:
            return 0 #inhibitory pulses to bring the applied current to zero
    return baseline_current

V0 = -0.065e-3  # Membrane potential in Volts
m0, h0, n0 = 0, 0, 0  # Gating variables
initial_conditions = [V0, m0, h0, n0]

# Solve ODE
y = odeint(hodgkin_huxley, initial_conditions, t, args=(I_app,))
V, m, h, n = y.T

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t * 1e3, [I_app(time) * 1e9 for time in t], label="Applied Current")
ax[0].set_ylabel("Current (nA)")
ax[0].set_title("Applied Current (Pulse Train)")
ax[0].legend()

ax[1].plot(t * 1e3, V * 1e3, label="Membrane Potential")
ax[1].set_xlabel("Time (ms)")
ax[1].set_ylabel("Membrane Potential (mV)")
ax[1].set_title("Membrane Response to Pulse Train")
ax[1].legend()

plt.tight_layout()
plt.show()

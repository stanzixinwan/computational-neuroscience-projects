import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Global parameters for the AELIF model
G_L = 10e-9               # Leak conductance (S)
C = 100e-12               # Capacitance (F)
E_L = -75e-3              # Leak potential (V)
V_Thresh = -50e-3         # Threshold potential (V)
V_Reset = -80e-3          # Reset potential (V)
delta_th = 2e-3           # Threshold shift factor (V)
tauw = 200e-3             # Adaptation time constant (s)
a = 2e-9                  # Adaptation recovery (S)
b = 0.02e-9               # Adaptation strength (A)

@jit(nopython=True)
def run_adex_model(tvector, I):
    """Simulate the Adaptive Exponential Leaky Integrate-and-Fire model."""
    v = np.full_like(tvector, E_L)  # Membrane potential
    w = np.zeros_like(tvector)      # Adaptation variable
    spikes = np.zeros_like(tvector)  # Spike times

    for j in range(len(tvector) - 1):
        if v[j] > Vmax:             # Spike condition
            v[j] = V_Reset          # Reset membrane potential
            w[j] += b               # Increment adaptation variable
            spikes[j] = 1           # Record spike
        
        # Update membrane potential using Forward Euler method
        dv = (G_L * (E_L - v[j] + delta_th * np.exp((v[j] - V_Thresh) / delta_th))
              - w[j] + I[j]) / C
        v[j + 1] = v[j] + dt * dv
        
        # Update adaptation variable
        dw = (a * (v[j] - E_L) - w[j]) / tauw
        w[j + 1] = w[j] + dt * dw
    
    return v, w, spikes

# Time vector
dt = 1e-6                 # Time step (s)
tmax = 1.5                # Maximum simulation time (s)
ton = 0.5                 # Start time of applied current (s)
toff = 1.0                # End time of applied current (s)

# Current setup
I0 = 0e-9                 # Baseline current (A)
Iapp = 500e-12           # Applied current step (A)
Vmax = 100e-3              # Voltage threshold for spike clipping (V)

# Simulation setup
tvector = np.arange(0, tmax + dt, dt)  # Time vector
I = np.full_like(tvector, I0)         # Initialize current
non = int(ton / dt)                   # Index for current onset
noff = int(toff / dt)                 # Index for current offset
I[non:noff] = Iapp                    # Apply step current

# Run the simulation
v, w, spikes = run_adex_model(tvector, I)

# Plot the results
plt.figure(figsize=(10, 8))

# Plot input current
plt.subplot(2, 1, 1)
plt.plot(tvector, I * 1e9, 'k')
plt.ylabel('I$_{app}$ (nA)')
plt.xlim([0, tmax])
plt.ylim([0, 1.25 * np.max(I) * 1e9])

# Plot membrane potential
plt.subplot(2, 1, 2)
plt.plot(tvector, v * 1e3, 'k')
plt.ylabel('V$_m$ (mV)')
plt.xlim([0, tmax])
plt.ylim([-95, 35])
plt.yticks([-50, 0])


# Simulation parameters
dt = 1e-6                 # Time step (s)
tmax = 5                  # Maximum simulation time (s)
tvector = np.arange(0, tmax, dt)  # Time vector
tau_sra = 200e-3          # Adaptation time constant (s)

# Current step parameters
ton = 0                   # Time to switch on the current step
toff = tmax               # Time to switch off the current step
non = int(ton / dt)       # Index for current onset
noff = int(toff / dt)     # Index for current offset
Iappvec = np.arange(0.20, 0.375, 0.005) * 1e-9  # Applied current vector

# Pre-allocate results arrays
initialrate = np.zeros(len(Iappvec))
finalrate = np.zeros(len(Iappvec))
singlespike = np.zeros(len(Iappvec))
meanV = np.zeros(len(Iappvec))

@jit(nopython=True)
def simulate_adex(Iappvec, tvector, non, noff):
    """Simulate the AdEx model for different applied currents."""
    dt = tvector[1] - tvector[0]
    initialrate = np.zeros(len(Iappvec))
    finalrate = np.zeros(len(Iappvec))
    singlespike = np.zeros(len(Iappvec))
    meanV = np.zeros(len(Iappvec))
    
    for trial, Iapp in enumerate(Iappvec):
        I = np.full_like(tvector, I0)
        I[non:noff] = Iapp

        v = np.zeros_like(tvector)
        v[0] = E_L
        I_sra = np.zeros_like(tvector)
        spikes = np.zeros_like(tvector)

        for j in range(len(tvector) - 1):
            if v[j] > Vmax:
                v[j] = V_Reset
                I_sra[j] += b
                spikes[j] = 1

            dv = (G_L * (E_L - v[j] + delta_th * np.exp((v[j] - V_Thresh) / delta_th)) 
                  - I_sra[j] + I[j]) / C
            v[j + 1] = v[j] + dt * dv

            dI_sra = (a * (v[j] - E_L) - I_sra[j]) / tau_sra
            I_sra[j + 1] = I_sra[j] + dt * dI_sra

        spiketimes = dt * np.where(spikes > 0)[0]

        if len(spiketimes) > 1:
            ISIs = np.diff(spiketimes)
            initialrate[trial] = 1 / ISIs[0]
            if len(ISIs) > 1:
                finalrate[trial] = 1 / ISIs[-1]
        elif len(spiketimes) == 1:
            singlespike[trial] = 1
        
        meanV[trial] = np.mean(v)
    
    return initialrate, finalrate, singlespike, meanV

initialrate, finalrate, singlespike, meanV = simulate_adex(Iappvec, tvector, non, noff) # Run the simulation

plt.figure(figsize=(10, 6)) # Plot the results

plt.plot(1e9 * Iappvec, finalrate, 'k', label='Final Rate') # Plot final rate

ISIindices = np.where(initialrate > 0)[0] # Plot initial rate
plt.plot(1e9 * Iappvec[ISIindices], initialrate[ISIindices], 'ok', markerfacecolor = 'none', label='1/ISI(1)')

ISIindices = np.where(singlespike > 0)[0] # Plot single spike cases
plt.plot(1e9 * Iappvec[ISIindices], singlespike[ISIindices] * 0, '*k', label='Single Spike')

# Labels and legend
plt.xlabel('Iapp (nA)')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

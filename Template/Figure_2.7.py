#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:36:30 2024

@author: pmiller
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Global parameters for the AdEx model
G_L = 10e-9               # Leak conductance (S)
C = 100e-12               # Capacitance (F)
E_L = -70e-3              # Leak potential (V)
V_Thresh = -50e-3         # Threshold potential (V)
V_Reset = -80e-3          # Reset potential (V)
deltaT = 2e-3             # Threshold shift factor (V)
tauw = 200e-3             # Adaptation time constant (s)
a = 2e-9                  # Adaptation recovery (S)
b = 0.02e-9               # Adaptation strength (A)

I0 = 0e-9                 # Baseline current (A)
Iapp = 0.221e-9           # Applied current step (A)
Vmax = 50e-3              # Voltage threshold for spike clipping (V)

dt = 1e-6                 # Time step (s)
tmax = 3                  # Maximum simulation time (s)
ton = 0.5                 # Start time of applied current (s)
toff = 2.5                # End time of applied current (s)

@jit(nopython=True)  # JIT-compiled function for efficiency
def run_adex_model(tvector, I):
    """
    Simulate the Adaptive Exponential Leaky Integrate-and-Fire (AdEx) model.

    Parameters:
        tvector (ndarray): Array of time points.
        I (ndarray): Applied current at each time point.

    Returns:
        v (ndarray): Membrane potential over time.
        w (ndarray): Adaptation variable over time.
        spikes (ndarray): Spike times.
    """
    v = np.full_like(tvector, E_L)  # Initialize membrane potential
    w = np.zeros_like(tvector)      # Initialize adaptation variable
    spikes = np.zeros_like(tvector)  # Initialize spike record

    for j in range(len(tvector) - 1):
        if v[j] > Vmax:             # Spike condition
            v[j] = V_Reset          # Reset membrane potential
            w[j] += b               # Increment adaptation variable
            spikes[j] = 1           # Record spike
        
        # Update membrane potential using Forward Euler method
        dv = (G_L * (E_L - v[j] + deltaT * np.exp((v[j] - V_Thresh) / deltaT))
              - w[j] + I[j]) / C
        v[j + 1] = v[j] + dt * dv
        
        # Update adaptation variable
        dw = (a * (v[j] - E_L) - w[j]) / tauw
        w[j + 1] = w[j] + dt * dw
    
    return v, w, spikes

# Main script
# if __name__ == "__main__":
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
plt.subplot(3, 1, 1)
plt.plot(tvector, I * 1e9, 'k')
plt.ylabel('I$_{app}$ (nA)')
plt.xlim([0, tmax])
plt.ylim([0, 1.25 * np.max(I) * 1e9])

plt.annotate('A',xy=(-0.15,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
# plt.text(-0.15, 1.05, 'A', transform=plt.gca().transAxes,
#          fontsize=16, fontweight='bold', va='top')

# Plot membrane potential
plt.subplot(3, 1, 2)
plt.plot(tvector, v * 1e3, 'k')
plt.ylabel('V$_m$ (mV)')
plt.xlim([0, tmax])
plt.ylim([-95, 35])
plt.yticks([-50, 0])
plt.annotate('B',xy=(-0.15,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')

# Plot adaptation variable
plt.subplot(3, 1, 3)
plt.plot(tvector, w * 1e9, 'k')
plt.xlabel('Time (s)')
plt.ylabel('I$_{SRA}$ (nA)')
plt.xlim([0, tmax])

# Add labels for subplots
plt.annotate('C',xy=(-0.15,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')

plt.tight_layout()
plt.show()

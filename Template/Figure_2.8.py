#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:48:46 2024

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
tau_sra = 200e-3          # Adaptation time constant (s)
a = 2e-9                  # Adaptation recovery (S)
b = 0.02e-9               # Adaptation strength (A)

I0 = 0e-9                 # Baseline current (A)
Vmax = 50e-3              # Spike voltage threshold (V)

# Simulation parameters
dt = 2e-6                 # Time step (s)
tmax = 5                  # Maximum simulation time (s)
tvector = np.arange(0, tmax, dt)  # Time vector

# Current step parameters
ton = 0                   # Time to switch on the current step
toff = tmax               # Time to switch off the current step
non = int(ton / dt)       # Index for current onset
noff = int(toff / dt)     # Index for current offset
Iappvec = np.arange(0.15, 0.305, 0.005) * 1e-9  # Applied current vector

# Pre-allocate results arrays
initialrate = np.zeros(len(Iappvec))
finalrate = np.zeros(len(Iappvec))
singlespike = np.zeros(len(Iappvec))
meanV = np.zeros(len(Iappvec))

@jit(nopython=True)  # JIT-compilation for performance
def simulate_adex(Iappvec, tvector, non, noff):
    """
    Simulate the AdEx model for different applied currents.

    Parameters:
        Iappvec (array): Applied currents to test.
        tvector (array): Time vector.
        non (int): Onset index for current.
        noff (int): Offset index for current.

    Returns:
        initialrate (array): Initial spike rates (1/ISI(1)).
        finalrate (array): Final spike rates (1/ISI(last)).
        singlespike (array): Indicators of single spikes.
        meanV (array): Mean voltage across trials.
    """
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

            dv = (G_L * (E_L - v[j] + deltaT * np.exp((v[j] - V_Thresh) / deltaT)) 
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

# Run the simulation
initialrate, finalrate, singlespike, meanV = simulate_adex(Iappvec, tvector, non, noff)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot final rate
plt.plot(1e9 * Iappvec, finalrate, 'k', label='Final Rate')

# Plot initial rate
ISIindices = np.where(initialrate > 0)[0]
plt.plot(1e9 * Iappvec[ISIindices], initialrate[ISIindices], 'ok', markerfacecolor = 'none', label='1/ISI(1)')

# Plot single spike cases
ISIindices = np.where(singlespike > 0)[0]
plt.plot(1e9 * Iappvec[ISIindices], singlespike[ISIindices] * 0, '*k', label='Single Spike')

# Labels and legend
plt.xlabel('Iapp (nA)')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

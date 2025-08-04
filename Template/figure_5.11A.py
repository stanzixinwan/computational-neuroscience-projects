#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:49:25 2024

@author: pmiller
"""

# Figure_5_11A.m
# This code runs through multiple trials of applied current, commencing
# each trial from the final state of the prior trial.
#
# This model is the Hodgkin-Huxley model in new units.
#
# This code is used to produce Figure 5.11A of the textbook
# An Introductory Course in Computational Neuroscience
# by Paul Miller
#
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from pm_integrate_models import integrate_HH

dt = 2e-8          # time-step for integration (sec)

## Neuron parameters
V_L = -0.060       # leak reversal potential (V)
E_Na = 0.045       # reversal for sodium channels (V)
E_K = -0.082       # reversal for potassium channels (V)
V0 = -0.065

G_L = 30e-9        # specific leak conductance (S)
G_Na = 12e-6       # specific sodium conductance (S)
G_K = 3.6e-6         # specific potassium conductance (S)

Cm = 100e-12       # specific membrane capacitance (F)

Ibase = 0.7e-9
tmax=0.25             # maximum time of simulation (s)
t=np.arange(0,tmax,dt)        # time vector

## Now add current pulses at different points on the cycle and analyze the
#  change in response due to the pulse.

Ipulse_amp = 10e-12

Npulses = 200
Ipulse = 0

@jit(nopython=True)
def trial_loop():
    shift = np.zeros(Npulses)
    i_startphase = 0
    i_stopphase = 0
    for trial in range(0,Npulses+1):  # INitially trial 0 is used to get the baseline
           
        print(trial)
        
        Iapp=Ibase*np.ones(len(t)) # Applied current, relevant in current-clamp mode
        if ( trial > 0 ):
            i_tpulse = i_startphase +  \
                int(np.floor((i_stopphase-i_startphase)*trial/Npulses))
            Iapp[i_tpulse:i_tpulse+int(np.round(0.005/dt))] = \
                Iapp[i_tpulse:i_tpulse+int(np.round(0.005/dt))] + Ipulse_amp
    
        
        V = integrate_HH(Iapp,t,dt,V_L,E_Na,E_K,G_Na,G_K,G_L,Cm)
    
        ## Detect the initiation time of individual bursts
        inspike = 0
        tspike = []
        Nspikes = 0
        for i in range(0,len(t)):
            if ( inspike == 0 ) & ( V[i] > -0.010 ):
                inspike = 1
                tspike.append(t[i])
                Nspikes = Nspikes + 1
            
            if (inspike == 1 ) & ( V[i] < -0.05 ):
                inspike = 0
            
        
        ## Now decide where a phase of "zero" corresponds to in the oscillation
        #  and calculate the time where this occurs -- time should be well after any
        #  initial transients.
        if ( trial == 0 ):
            i_startphase = int(np.round(tspike[3]/dt))
            i_stopphase = int(np.round(tspike[4]/dt))
            period = (tspike[4] - tspike[3])
        else:
            shift[trial] = 2*np.pi*(1 - (tspike[4] - tspike[3])/period )
            if ( shift[trial] < -np.pi ):
                shift[trial] = shift[trial] + 2*np.pi
            
            if ( shift[trial] > np.pi ):
                shift[trial] = shift[trial] - 2*np.pi
    return shift

shift = trial_loop()
 
## Set up the plotting parameters and plot the distibutions and ROC curves
## Set default styles for the plot
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8.0
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.labelsize'] = 12.0
plt.rcParams['xtick.labelsize'] = 12.0
plt.rcParams['ytick.labelsize'] = 12.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.labelsize'] = 16.0
size_of_labels =16
 

plt.plot(np.arange(1,Npulses+1)/Npulses,shift,'k')

plt.plot(np.arange(1,Npulses+1)/Npulses,np.zeros(Npulses),'k:')
plt.xlabel('Phase of pulse',fontsize=size_of_labels)
plt.ylabel('Phase shift',fontsize=size_of_labels)
plt.xlim(0,1)
plt.ylim(-0.15,0.18)
plt.yticks([-0.1,0, 0.1,0.2])
plt.xticks([0,0.5,1], ['0',r'$\pi$',r'2$\pi$'] )


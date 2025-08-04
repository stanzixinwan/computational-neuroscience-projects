#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:39:14 2024

@author: pmiller
"""

# Figure_4_12.m
# This model contains a T-type Calcium current to generate a
# post-inhibitory rebound as a model of thalamic relay cells.
# The code will step from a hyperpolarizing applied current in 5 increments
# of decreasing hyperpolarization to depolarization.
#
# This code is used to produce Figure 4.12 in the textbook 
# An Introductory Course in Computational Neuroscience
# by Paul Miller, Brandeis University (2017)
#
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

## Simulation parameters for Fig. 4.12
dt = 0.00001       # small time-step when action potentials are simulated
istart = 0.5       # time applied current starts
I0=-0.1e-9         # Initial hyperpolarizing current
Istep= 0.05e-9     # magnitude of each step in the current
Nsteps = 6         # Number of values of applied current
steplength = 0.25  # Duration of each step of constant current
tmax = Nsteps*steplength   # Total simulation time
nsteplength = int(np.ceil(steplength/dt)) # No. of time points per current step
t=np.arange(0,tmax,dt)        # time vector

## Parameters for the Thalamic rebound model used here
E_L = -0.070       # leak reversal potential
E_Na = 0.055       # reversal for sodium channels
E_K = -0.090       # reversal for potassium channels
E_Ca = 0.120        # reversal potential for Ca current

G_L = 11e-9        # leak conductance in Siemens 
G_Na = 2.5e-6      # sodium conductance
G_K = 1.5e-6       # potassium conductance
G_CaT = 0.19e-6    # T-type calcium conductance

Cm = 0.1e-9        # membrane capacitance in Farads 

Iapp=np.zeros(len(t)) # Applied current vector
for step in range(1,Nsteps+1):                    # Loop through current steps
    istart = (step-1)*nsteplength+1    # Index of current onset
    istop = step*nsteplength           # Index of current offset
    Iapp[istart:istop+1] = I0+(step-1)*Istep # Set applied current



## Commence the simulation through time
@jit(nopython=True)
def integrate_CaT(Iapp):
    ## Initialize variables used in the simulation
    I_L= np.zeros(len(t))    # to store leak current
    I_Na= np.zeros(len(t))   # to store sodium current
    I_K= np.zeros(len(t))    # to store potassium current
    I_CaT = np.zeros(len(t)) # to store T-type calcium current

    V=np.zeros(len(t))   # membrane potential vector
    V[0] = -0.078      # initialize membrane potential
    n=np.zeros(len(t))   # n: potassium activation gating variable
    n[0] = 0.025       # initialize near steady state
    m=np.zeros(len(t))   # m: sodium activation gating variable
    m[0] = 0.005       # initialize near steady state
    h=np.zeros(len(t))   # h: sodim inactivation gating variplot(t,V)able
    h[0] = 0.6         # initialize near steady state

    mca=np.zeros(len(t)) # CaT current activation gating variable
    mca[0] = 0.025     # initialize near steady state   
    hca=np.zeros(len(t)) # CaT current inactivation gating variable
    hca[0] = 0.6       # initialize near steady state
    
    Itot=np.zeros(len(t)) # in case we want to plot and look at the total current

    for i in range(1,len(t)): # now see how things change through time
        Vm = V[i-1] 
        
        # Sodium and potassium gating variables are defined by the
        # voltage-dependent transition rates between states, labeled alpha and
        # beta. Written out from Dayan/Abbott, units are 1/sec.
        if ( Vm == -35 ): 
            alpha_m = 1e3
        else: 
            alpha_m = 1e5*(Vm+0.035)/(1-np.exp(-100*(Vm+0.035)))
        
        beta_m = 4000*np.exp(-(Vm+0.060)/0.018)
    
        # Now sodium inactivation rate constants
        alpha_h = 350*np.exp(-50*(Vm+0.058))
        beta_h = 5000/(1+np.exp(-100*(Vm+0.028)))
        
        # Now potassium activation rate constants (the "if" prevents a divide
        # by zero)
        if ( Vm == -0.034 ): 
           alpha_n = 500
        else:
            alpha_n = 5e4*(Vm+0.034)/(1-np.exp(-100*(Vm+0.034)))
        
        beta_n = 625*np.exp(-12.5*(Vm+0.044))
         
        # From the alpha and beta for each gating variable we find the steady
        # state values (_inf) and the time constants (tau_) for each m,h and n.   
        m_inf = alpha_m/(alpha_m+beta_m)
        
        tau_h = 1/(alpha_h+beta_h)      # time constant converted from ms to sec
        h_inf = alpha_h/(alpha_h+beta_h)
        
        tau_n = 1/(alpha_n+beta_n)      # time constant converted from ms to sec
        n_inf = alpha_n/(alpha_n+beta_n)   
        
        # for the Ca_T current gating variables are given by formulae for the 
        # steady states and time constants:    
        mca_inf = 1/(1+np.exp(-(Vm+0.052)/0.0074))    # Ca_T activation
        hca_inf = 1/(1+np.exp(500*(Vm+0.076)))        # Ca_T inactivation
        if ( Vm < -80 ): 
            tau_hca = 1e-3*np.exp(15*(Vm+0.467))
        else:
            tau_hca = 1e-3*(28+np.exp(-(Vm+0.022)/0.0105))
    
        m[i] = m_inf    # Update m, assuming time constant is neglible.
        
        h[i] = h_inf - (h_inf-h[i-1])*np.exp(-dt/tau_h)    # Update h
        
        n[i] = n_inf - (n_inf-n[i-1])*np.exp(-dt/tau_n)    # Update n
            
        mca[i] = mca_inf                           # Update mca instantaneously
        hca[i] = hca_inf - (hca_inf-hca[i-1])*np.exp(-dt/tau_hca) # update hca
        
        G_Na_now = G_Na*m[i]*m[i]*m[i]*h[i]    # sodium conductance
        I_Na[i-1] = G_Na_now*(E_Na-V[i-1])     # sodium current
        
        G_K_now = G_K*n[i]*n[i]*n[i]*n[i]      # potassium conductance
        I_K[i-1] = G_K_now*(E_K-V[i-1])        # potassium current
        
        G_CaT_now = G_CaT*mca[i]*mca[i]*hca[i] # T-type calcium conductance
        I_CaT[i-1] = G_CaT_now*(E_Ca-V[i-1])   # Calcium T-type current
            
        I_L[i-1] = G_L*(E_L-V[i-1])            # Leak current
    
        Itot[i-1] = I_L[i-1]+I_Na[i-1]+I_K[i-1] \
                    +I_CaT[i-1] +Iapp[i-1] # total current is sum of leak + active channels + applied current
         
        G_Tot = G_L+G_Na_now+G_K_now+G_CaT_now # Total conductance
    
        # V_inf is steady state voltage given all conductances and reversals
        V_inf = (G_L*E_L + G_Na_now*E_Na + G_K_now*E_K  + G_CaT_now*E_Ca+Iapp[i-1])/G_Tot
            
        # Membrane potential update is via the np.exponential Euler method
        V[i] = V_inf - (V_inf-V[i-1])*np.exp(-dt*G_Tot/Cm)  

    return V,I_Na,I_K,I_CaT,I_L,m,h,n,mca,hca

[V,I_Na,I_K,I_CaT,I_L,m,h,n,mca,hca] = integrate_CaT(Iapp)

fig,axs=plt.subplots(3,1,figsize=(5,5))
plt.tight_layout(pad = 0, h_pad = 2)
## Now set up the plotting parameters
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8.0
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.labelsize'] = 12.0
plt.rcParams['xtick.labelsize'] = 12.0
plt.rcParams['ytick.labelsize'] = 12.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.labelsize'] = 12.0
size_of_labels =14

# First plot the applied current vs time
axs[0].plot(t,Iapp*1e9,'k')
axs[0].set_ylabel('I$_{app}$ (nA)',fontsize=size_of_labels)
axs[0].set_xlim(0,tmax)
axs[0].set_ylim(-0.12,0.17)
axs[0].set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])

# Next plot the membrane potential vs time
axs[1].plot(t,V*1000,'k')
axs[1].set_ylabel('V$_{m}$ (mV)',fontsize=size_of_labels)
axs[1].set_xlim(0,tmax)
axs[1].set_ylim(-85,50)
axs[1].set_xticks([0 ,0.25, 0.5, 0.75, 1, 1.25, 1.5])

# Finally plot the T-type calcium inactivation variable
axs[2].plot(t,hca,'k')
axs[2].set_xlabel('Time (sec)',fontsize=size_of_labels)
axs[2].set_ylabel('h$_{CaT}$ ',fontsize=size_of_labels)
axs[2].set_xlim(0,tmax)
axs[2].set_ylim(0,1)
axs[2].set_xticks([0 ,0.25, 0.5, 0.75, 1, 1.25, 1.5])

# Label the panels A, B, and C
axs[0].annotate('A',xy=(-0.2,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[1].annotate('B',xy=(-0.2,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[2].annotate('C',xy=(-0.2,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')


    
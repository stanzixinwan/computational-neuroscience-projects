#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure_4_15.m
# This model is based on the two-compartment model of Pinsky and Rinzel (1994)
# The dendritic compartment produces calcium spikes, which couple to the
# somatic compartment to produce regular bursts of action potentials.
#
# The code requires the two functions,
# PR_soma_gating and PR_dend_gating
# to be in the path.
# 
# This code is used to generate figures 4.15 and 4.16 of Chapter 4 in the
# textbook:
# "An Introductory Course in Computational Neuroscience"
#
# by Paul Miller, Brandeis University (2017).
#
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit
sys.path.insert(0,'/Users/stanw/OneDrive/document/Brandeis/NBIO 136/Packages')
from pm_functions import PR_soma_gating, PR_dend_gating

dt = 10e-6
tmax=2

E_L = -0.060   # leak reversal potential
E_Na = 0.060   # reversal for sodium channels
E_K = -0.075   # reversal for potassium channels
E_Ca = 0.080   # reversal for calcium channels

S_frac = 1/3  # fraction of total membrane area that is soma
D_frac = 1-S_frac # rest of area is dendritic

# Conductance values for somatic channels follow
G_LS = 5e-9*S_frac     # somatic leak conductance in Siemens 
G_Na = 3e-6*S_frac     # sodium conductance (Soma)
G_K = 2e-6*S_frac      # potassium conductance (Soma)

# Conductance values for dendritic channels follow
G_LD = 5e-9*D_frac         # dendritic leak conductance in Siemens 
G_Ca = 2e-6*D_frac         # calcium conductance (Dendrite)
G_KAHP = 0.04e-6*D_frac    # Potassium conductance to generate after-hyperpolarization
G_KCa = 2.5e-6*D_frac      # calcium-dependent Potassium conductance

G_Link = 0e-9 # conductance linking dendrite and soma 

tau_Ca = 50e-3             # time constant for buffering of calcium 
convert_Ca = 0.25e7/D_frac  # conversion changing calcium charge entry per unit area into concentration

CmS = 100e-12*S_frac     # somatic membrane capacitance in Farads 
CmD = 100e-12*D_frac     # dendritic membrane capacitance in Farads

t = np.arange(0,tmax,dt)       # time vector

@jit(nopython=True)
def integrate_PR(Iapp):
    
    VS=np.zeros(len(t))  # somatic voltage vector
    VD=np.zeros(len(t))  # dendritic voltage vector
    VS[0] = E_L    # set the inititial value of somatic voltage
    VD[0] = E_L    # set the inititial value of dendritic voltage
    
    
    Ca=np.zeros(len(t))  # dendritic calcium level (extra Ca above base level)
    Ca[0] = 0          # initialize with no (extra) Ca in cell.
    
    I_LD= np.zeros(len(t))      # leak current in dendrite
    I_LS= np.zeros(len(t))      # leak current in soma
    I_Na = np.zeros(len(t))     # sodium current (soma)
    I_K = np.zeros(len(t))      # potassium current (soma)
    I_Ca = np.zeros(len(t))     # calcium current (dendrite)
    I_KAHP = np.zeros(len(t))   # after-hyperpolarization current (dendrite)    
    I_KCa = np.zeros(len(t))    # calcium-dependent potassium current (dendrite)
    I_Link=np.zeros(len(t))     # current between compartments
    IS=np.zeros(len(t))         # total current to soma
    ID=np.zeros(len(t))        # total current to dendrite
    n=np.zeros(len(t))   # n: potassium activation gating variable
    m=np.zeros(len(t))   # m: sodium activation gating variable
    h=np.zeros(len(t))   # h: sodim inactivation gating variplot(t,V)able
    n[0] = 0.4         # initialize near steady state at resting potential
    h[0] = 0.5         # initialize near steady state at resting potential
    
    mca=np.zeros(len(t))     # Ca current activation gating variable
    mkca=np.zeros(len(t))    # K_Ca current activation gating variable
    mkahp = np.zeros(len(t)) # K_AHP current activation gating variable
    mkahp[0] = 0.2         # initialize near steady state at resting potential
    mkca[0] = 0.2          # initialize near steady state at resting potential
    Ca[0] = 1e-6           # initialize near steady state at resting potential
    
    for i in range(1,len(t)): # now see how things change through time
        I_LS[i] = G_LS*(E_L-VS[i-1])
        I_LD[i] = G_LD*(E_L-VD[i-1])
       
        Vm = VS[i-1] 
        VmD = VD[i-1] 
        Catmp = Ca[i-1]
        mtmp = m[i-1]
        htmp = h[i-1]
        ntmp = n[i-1]
        mcatmp = mca[i-1]
        mkcatmp = mkca[i-1]
        mkahptmp = mkahp[i-1]
        
        # From the alpha and beta for each gating variable we find the steady
        # state values (_inf) and the time constants (tau_) for each m,h and n.
        [ alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n ] = PR_soma_gating(Vm)

        [ alpha_mca, beta_mca, alpha_mkca, beta_mkca, alpha_mkahp, beta_mkahp ] = PR_dend_gating(VmD, Catmp)

        m[i] = mtmp + dt*( alpha_m*(1-mtmp) - beta_m*mtmp )
        h[i] = htmp + dt*( alpha_h*(1-htmp) - beta_h*htmp )
        n[i] = ntmp + dt*( alpha_n*(1-ntmp) - beta_n*ntmp )
        
        mca[i] = mcatmp + dt*( alpha_mca*(1-mcatmp) - beta_mca*mcatmp )
        mkca[i] = mkcatmp + dt*( alpha_mkca*(1-mkcatmp) - beta_mkca*mkcatmp )
        mkahp[i] = mkahptmp + dt*( alpha_mkahp*(1-mkahptmp) - beta_mkahp*mkahptmp )
         
        G_Na_now = G_Na*m[i]*m[i]*h[i]
        I_Na[i] = G_Na_now*(E_Na-VS[i-1]) # sodium current in soma
        
        G_K_now = G_K*n[i]*n[i]
        I_K[i] = G_K_now*(E_K-VS[i-1]) # potassium delayed rectifier current, soma
        
        G_Ca_now = G_Ca*mca[i]*mca[i]
        I_Ca[i] = G_Ca_now*(E_Ca-VD[i-1]) # persistent sodium current in dendrite
        
        if ( Ca[i-1] > 250e-6 ): 
            G_KCa_now = G_KCa*mkca[i]
        else:
            G_KCa_now = G_KCa*mkca[i]*Ca[i-1]/250e-6
        
        I_KCa[i] = G_KCa_now*(E_K-VD[i-1]) # calcium-dependent potassium current in dendrite
        
        G_KAHP_now = G_KAHP*mkahp[i]
        I_KAHP[i] = G_KAHP_now*(E_K-VD[i-1]) # calcium-dependent potassium current in dendrite
        I_Link[i] = G_Link*(VD[i-1]-VS[i-1])
            
        IS[i] = I_LS[i]+I_Na[i]+I_K[i]+I_Link[i] + Iapp[i]   # total current in soma
        ID[i] = I_LD[i]+I_Ca[i]+I_KCa[i]+I_KAHP[i]-I_Link[i] # total current in dendrite
        
        gS_Tot = G_LS+G_Na_now+G_K_now+G_Link
        VS_inf = (G_LS*E_L + G_Na_now*E_Na + G_K_now*E_K \
                + VD[i-1]*G_Link )/gS_Tot
                       
        gD_Tot = G_LD+G_Ca_now+G_KCa_now+G_KAHP_now+G_Link
        VD_inf = (G_LD*E_L + G_Ca_now*E_Ca + G_KCa_now*E_K + G_KAHP_now*E_K \
                + VS[i-1]*G_Link )/gD_Tot
                       
        VS[i] = VS_inf - (VS_inf-VS[i-1])*np.exp(-dt*gS_Tot/CmS)  # Update the membrane potential, V.
        VD[i] = VD_inf - (VD_inf-VD[i-1])*np.exp(-dt*gD_Tot/CmD)  # Update the membrane potential, V.
        Ca_inf = tau_Ca*convert_Ca*I_Ca[i]
        Ca[i] = Ca_inf - (Ca_inf-Ca[i-1])*np.exp(-dt/tau_Ca)  # update Ca level
            
    return VS, VD, Ca, I_Na, I_K, I_KCa, I_KAHP, I_Link, IS, ID

Iapp = np.zeros(len(t))
[VS, VD, Ca, I_Na, I_K, I_KCa, I_KAHP, I_Link, IS, ID] = integrate_PR(Iapp)


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

## Now plot the graphs
# First plot somatic membrane potential and dendritic membrane potential
# one above the other for entire time window of 2 sec.
fig, axs = plt.subplots(2,1,figsize=(5,6))
plt.tight_layout(pad = 0, h_pad = 2, w_pad = 2)
axs[0].plot(t,VS*1000,'k')
axs[0].set_xlim(0,2)
axs[0].set_ylim(-85,50)
axs[0].set_ylabel('V$_S$ (mV)')

axs[1].plot(t,VD*1000,'k')
axs[1].set_xlim(0,2)
axs[1].set_ylim(-85,50)
axs[1].set_ylabel('V$_D$ (mV)')
axs[1].set_xlabel('Time (sec)')

axs[0].annotate('A',xy=(-0.2,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[1].annotate('B',xy=(-0.2,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')

# Then plot a zoom of membrane potentials and other key variables in a
# zoomed in time window of an individual burst.
fig, axs = plt.subplots(2,2,figsize=(8,6))
plt.tight_layout(pad = 0, h_pad = 2, w_pad = 3)
axs[0,0].plot(t,VS*1e3,'k',label='V$_S$')
axs[0,0].plot(t,VD*1e3,':k',label='V$_D$')
axs[0,0].set_xlim(0.86,0.91)
axs[0,0].set_ylim(-85,50)
axs[0,0].set_ylabel('Membrane Potential (mV)')
axs[0,0].legend()

axs[1,0].plot(t,I_Link*1e9,'k')
axs[1,0].set_ylabel('I$_{Link}$ (nA)')
axs[1,0].set_xlabel('Time (sec)')
axs[1,0].set_xlim(0.86,0.91)
axs[1,0].set_ylim(-2.1,2.1)

axs[0,1].plot(t,Ca*1e3,'k')
axs[0,1].set_xlim(0.85,1.15)
axs[0,1].set_ylim(0,4)
axs[0,1].set_ylabel('[Ca] (mM)')

axs[1,1].plot(t,I_KCa*1e9,'k:',label='I$_{KCa}$')
axs[1,1].plot(t,I_KAHP*1e11,'k--',label='100 x I$_{KAHP}$')
axs[1,1].set_xlim(0.85,1.15)
axs[1,1].set_ylim(-110,0)
axs[1,1].set_ylabel('Dendritic K-currents (nM)')
axs[1,1].set_xlabel('Time (sec)')
axs[1,1].legend()

axs[0,0].annotate('A',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[1,0].annotate('B',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[0,1].annotate('C',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axs[1,1].annotate('D',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
plt.show()
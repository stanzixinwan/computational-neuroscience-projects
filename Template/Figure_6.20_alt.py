#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:36:04 2024

@author: pmiller
"""

# Figure_6.20.py 
# 
#  Firing-rate model code for producing a stable "bump" of activity on a
#  ring, based on a small variation of the orientation selectivity network.
#
#  The bump can move if the inhibitory feedback is directional, in which
#  case it can represent head direction cells.
#  See paper by Compte et al Cerebral Cortex (2000) for a spiking neuron
#  model.
#
#  This code was used to produce Figure 6.19 in the book:
#  An Introductory Course in Computational Neuroscience
#  by Paul Miller (Brandeis University, 2017).
#
###########################################################################

import numpy as np
import matplotlib.pyplot as plt                       # Clear all variables
from numba import jit
# import sys
# sys.path.insert(0,'/Users/pmiller/TEACH/BOOK_COMPNEURO/PYTHON_CODES')
# from pm_package.pm_matmath import pyvmmul,pymatmul

question_part_vec = ['A','B', 'B2','C'] 
stim_random_on = 0              # Set to 1 to include randomness in the stimulus
dt=0.0001                       # Time step in sec (0.1ms)
tmax =8                         # Time step in sec (300ms)
t=np.arange(0,tmax,dt)          # Create time vector
Nt = len(t)

Ncells = 50                     # Number of cells around the ring

## Stimulus input parameters
epsilon = 0.1              # Variation of input across cells from 1-epsilon to 1+epsilon
cuecell = int(round(Ncells/2))         # Cell with peak of the LGN input
hstart = int(round(tmax/4))               # Time to begin input
hend = int(round(tmax/2))                # Time to end input
contrasts = [0.5,1]    # Range of contrasts to use
noise_amp = 0.2            # Noise heterogeneity in the input when stim_random_on=1

## Set up the plotting parameters
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['lines.markersize'] = 8.0
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.labelsize'] = 12.0
plt.rcParams['xtick.labelsize'] = 12.0
plt.rcParams['ytick.labelsize'] = 12.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.labelsize'] = 12.0
size_of_labels =14
     


@jit(nopython=True)
def integrate_model(hE,hI,WEE,WEI,WIE,WII, I0E, I0I):
    rE=np.zeros((Nt,Ncells))        # Rate matrix for all timesteps and all cells
    rI=np.zeros((Nt,Ncells))        # Rate matrix for all timesteps and all cells

    rE[0,10] = 0.1                  # small perturbation to break initial symmetry
    # Begin the time integration
    for i in range(1,Nt):
        
        # Only set the input current to each unit as the input current when
        # the stimulus is on from hstart to hend.
        if ( t[i] >= hstart ) & (t[i] <= hend ):
            IstimE = hE                        # Set input to E cells
            IstimI = hI                        # Set input to I cells
        else:
            IstimE = np.zeros(Ncells)           # Otherwise no input current
            IstimI = np.zeros(Ncells)           # Otherwise no input current

        IstimE = IstimE + 0.1*np.random.randn(1,Ncells)
        IstimI = IstimI + 0.1*np.random.randn(1,Ncells)
        # Update rates of all excitatory units based on rates of all units
        # at the previous time point
        rE[i,:] = rE[i-1,:]*(1-dt/tauE) + \
            dt*(IstimE + (rE[i-1,:]@WEE) + (rI[i-1,:]@WIE) + I0E)/tauE
        
        # Update rates of all inhibitory units based on rates of all units
        # at the previous time point
        rI[i,:] = rI[i-1,:]*(1-dt/tauI) + \
            dt*(IstimI + (rE[i-1,:]@WEI) + (rI[i-1,:]@WII) +I0I)/tauI
        
        for index in range( 0,Ncells) :
            if ( rE[i,index] < 0):
                rE[i,index] = 0
        
        for index in range( 0,Ncells) :
            if ( rI[i,index] < 0):
                rI[i,index] = 0
                
    return rE,rI
    

@jit(nopython=True)
def loop_contrasts(WEE,WEI,WIE,WII,I0E,I0I,AE,AI):
    hE = np.zeros(Ncells)       # Vector for inputs to each E cell when input is on
    hI = np.zeros(Ncells)       # Vector for inputs to each I cell when input is on
    contrasts=[1]
    # Now loop through the set of different contrasts
    for trial in range(0,len(contrasts)):
        c = contrasts[trial]               # Contrast to be used
        
        # Now set the input current that varies across cells
        for cell in range(0,Ncells):
            hE[cell] = AE*c*(1 + epsilon*np.cos(2*np.pi*(cell-cuecell)/Ncells))
            hI[cell] = AI*c*(1 + epsilon*np.cos(2*np.pi*(cell-cuecell)/Ncells))
    
        if ( stim_random_on == 1 ):
            hE = hE*(1+noise_amp*(np.random.rand(Ncells)-0.5))
            hI = hI*(1+noise_amp*(np.random.rand(Ncells)-0.5))
    
        [rE,rI] = integrate_model(hE, hI, WEE,WEI,WIE,WII, I0E, I0I)
                        
    return rE,rI,contrasts


 
for question in range (0,len(question_part_vec)) :
    question_part = question_part_vec[question] 
    print(question_part)
    ## Cortical network parameters
    match question_part:
        # A: # Stationary bump based on excitatory feedback. Activity can be on or off.
        case 'A':
            AE = 2             # Maximum LGN input to E cells
            AI = 0             # Maximum LGN input to I cells
            I0E = -2           # Background input to E cells minus threshold
            I0I = -4           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 8           # Mean E to E connection weight
            WEI0 = 3           # Mean E to I connection weight
            WIE0 = -3          # Mean I to E connection weight
            WIEshift = 0.0       # Difference in tuning preference for I cells connecting to E cells
            WII0 = 0           # Mean I to E connection weight
            WIIshift = 0       # Difference in tuning preference for I cells connecting to E cells
            
            # B: Stationary bump based on excitatory feedback. Activity is always on.
        case 'B':
            AE = 10            # Maximum LGN input to E cells
            AI = 0             # Maximum LGN input to I cells
            I0E = 1            # Background input to E cells minus threshold
            I0I = -4           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 8           # Mean E to E connection weight
            WEI0 = 3           # Mean E to I connection weight
            WIE0 = -3          # Mean I to E connection weight
            WIEshift = 0       # Difference in tuning preference for I cells connecting to E cells
            WII0 = 0           # Mean I to E connection weight
            WIIshift = 0       # Difference in tuning preference for I cells connecting to E cells
        case 'B2':
            AE = 10            # Maximum LGN input to E cells
            AI = 0             # Maximum LGN input to I cells
            I0E = 2            # Background input to E cells minus threshold
            I0I = -1           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 6           # Mean E to E connection weight
            WEI0 = 3           # Mean E to I connection weight
            WIE0 = -2          # Mean I to E connection weight
            WIEshift = 0       # Difference in tuning preference for I cells connecting to E cells
            WII0 = 0           # Mean I to E connection weight
            WIIshift = 0       # Difference in tuning preference for I cells connecting to E cells
            
            # C: Moving bump, symmetric excitatory and asymmetric inhibitory feedback.
        case 'C':
            AE = 0            # Maximum LGN input to E cells
            AI = 0             # Maximum LGN input to I cells
            I0E = 2            # Background input to E cells minus threshold
            I0I = -1           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 6           # Mean E to E connection weight
            WEI0 = 3           # Mean E to I connection weight
            WIE0 = -2          # Mean I to E connection weight
            WIEshift = 0.05*np.pi       # Difference in tuning preference for I cells connecting to E cells
            WII0 = 0           # Mean I to E connection weight
            WIIshift = 0       # Difference in tuning preference for I cells connecting to E cells
        case 'C2':
            AE = 0            # Maximum LGN input to E cells
            AI = 0             # Maximum LGN input to I cells
            I0E = 4            # Background input to E cells minus threshold
            I0I = -1           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 8           # Mean E to E connection weight
            WEI0 = 3           # Mean E to I connection weight
            WIE0 = -3          # Mean I to E connection weight
            WIEshift = 0.05*np.pi       # Difference in tuning preference for I cells connecting to E cells
            WII0 = 0           # Mean I to E connection weight
            WIIshift = 0       # Difference in tuning preference for I cells connecting to E cells
            
            # D: Stationary bump attractor based on symmetric I-to-I feedback.
        case 'D':
            AE = 0             # Maximum LGN input to E cells
            AI = 20            # Maximum LGN input to I cells
            I0E = -10          # Background input to E cells minus threshold
            I0I = 20           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 0           # Mean E to E connection weight
            WEI0 = 0           # Mean E to I connection weight
            WIE0 = 0           # Mean I to E connection weight
            WIEshift = 0       # Difference in tuning preference for I cells connecting to E cells
            WII0 = -10           # Mean I to E connection weight
            WIIshift = np.pi       # Difference in tuning preference for I cells connecting to E cells
            
            # E: A moving bump attractor based on asymmetric I-to-I feedback.
        case 'E':
            AE = 0            # Maximum LGN input to E cells
            AI = 20             # Maximum LGN input to I cells
            I0E = -10            # Background input to E cells minus threshold
            I0I = 20           # Background input to I cells minus threshold
            tauE = 0.050       # Time constant for E cells
            tauI = 0.005       # Time constant for I cells
            WEE0 = 0           # Mean E to E connection weight
            WEI0 = 0           # Mean E to I connection weight
            WIE0 = 0          # Mean I to E connection weight
            WIEshift = 0.05*np.pi       # Difference in tuning preference for I cells connecting to E cells
            WII0 = -10           # Mean I to E connection weight
            WIIshift = np.pi+0.001*np.pi       # Difference in tuning preference for I cells connecting to E cells
    
    # Now produce all the within-network connections as cosine functions. Note
    # that 1 is added to the cosine so the variation of the term within
    # parentheses has a minimum of 0 and a maximum of 2.
    # The initial terms WEE0, WEI0, and WIE0 correspond then to the mean weight
    # of that type of connection.
    # The I-to-E connection has an extra "WIEshift" term which can be set to pi
    # so that inhibition is from cells with opposite LGN input.
    WEE = np.zeros((Ncells,Ncells))
    WEI = np.zeros((Ncells,Ncells))
    WIE = np.zeros((Ncells,Ncells))
    WII = np.zeros((Ncells,Ncells))
    for cell1 in range (0,Ncells):
        for cell2 in range (0,Ncells):
            WEE[cell1,cell2] = WEE0*(1+np.cos(2*np.pi*(cell2-cell1)/Ncells) ) / Ncells
            WEI[cell1,cell2] = WEI0*(1+np.cos(2*np.pi*(cell2-cell1)/Ncells) ) / Ncells
            WIE[cell1,cell2] = WIE0*(1+np.cos(WIEshift+2*np.pi*(cell2-cell1)/Ncells) ) / Ncells
            WII[cell1,cell2] = WII0*(1+np.cos(WIIshift+2*np.pi*(cell2-cell1)/Ncells) ) / Ncells

    [rE,rI,contrasts] = loop_contrasts(WEE,WEI,WIE,WII,I0E,I0I,AE,AI)
    
    fig1, axs1 = plt.subplots(2,1,figsize=(6,4))
    fig2, axs2 = plt.subplots(2,1,figsize=(6,4))
    
    plt.tight_layout(pad = 0, h_pad = 2, w_pad = 2)
    
    ## Now plot the results for the contrast used
    # plot rate of all E cells at end of simulation
    axs1[0].plot(rE[Nt-1,:],'g')
    axs1[0].set_xlabel('cell index')
    axs1[0].set_ylabel('rate of E-neurons')
    
    # plot rate of all I cells at end of simulation
    axs1[1].plot(rI[Nt-1,:],'g')
    axs1[1].set_xlabel('cell index')
    axs1[1].set_ylabel('rate of I-neurons')
    
    
    # plot rate of excitatory cuecell as a function of time
    axs2[0].plot(t,rE[:,cuecell])
    # plot rate of E cell with null direction input as a function of time
    axs2[0].plot(t,rE[:,np.mod(cuecell+int(Ncells/2),Ncells)])
    axs2[0].set_xlabel('time')
    axs2[0].set_ylabel('rate of excitatory cell')
    
    
    # plot rate of inhibitory cuecell as a function of time
    axs2[1].plot(t,rI[:,cuecell])
    
    # plot rate of I cell with null direction input as a function of time
    axs2[1].plot(t,rI[:,np.mod(cuecell+int(Ncells/2),Ncells)])
    axs2[1].set_xlabel('time')
    axs2[1].set_ylabel('rate of inhibitory cell',fontsize=size_of_labels)
    
    
    x = np.arange(0,Ncells)
    y = np.arange(0,Ncells)
    X, Y = np.meshgrid(x, y)
    
    match question_part:
        case 'A':
            fig23 = plt.figure(figsize=(12,6))
            plt.tight_layout(pad = 0, h_pad = 1, w_pad = 1)

            axs230 = fig23.add_subplot(1, 2, 1, projection='3d')
            surf = axs230.plot_surface(X,Y,WEE,cmap='gray')
            axs230.zaxis.set_rotate_label(False) 
            axs230.view_init(elev=30, azim=45)
            axs230.set_xlim(0,50)
            axs230.set_ylim(0,50)
            axs230.set_zlim(0,0.32)
            axs230.set_zticks([0,0.1,0.2,0.3])
            axs230.set_xlabel('Neuron i',fontsize=size_of_labels)
            axs230.set_ylabel('Neuron j',fontsize=size_of_labels)
            axs230.set_zlabel('W$^{EE}_{ij}$',fontsize=size_of_labels,rotation=90)
            axs230.annotate('A',xy=(-0.1,1.0),xycoords='axes fraction',fontsize=20,fontweight='bold')
 
     
        case 'D':
            axs231 = fig23.add_subplot(1, 2, 2, projection='3d')
            surf = axs231.plot_surface(X,Y,WII,cmap='gray')
            axs231.zaxis.set_rotate_label(False) 
            axs231.view_init(elev=30, azim=45)
            axs231.set_xlim(0,50)
            axs231.set_ylim(0,50)
            axs231.set_zlim(-0.4,0)
            axs231.set_zticks([-0.4,-0.3,-0.2,-0.1,0])
            axs231.set_xlabel('Neuron i',fontsize=size_of_labels)
            axs231.set_ylabel('Neuron j',fontsize=size_of_labels)
            axs231.set_zlabel('W$^{II}_{ij}$',fontsize=size_of_labels,rotation=90)
            axs231.annotate('B',xy=(-0.1,1.0),xycoords='axes fraction',fontsize=20,fontweight='bold')

    
    plt_on = 0
    match question_part:
        case 'A':
            fig24,axs24 = plt.subplots(3,1,figsize=(12,10))
            plt.tight_layout(pad = 0, h_pad = 2, w_pad = 2)
    
            use_ax = axs24[0]
    
            axs24[0].annotate('A',xy=(-0.1,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
    #        axs24[0].imshow(np.transpose(rE),cmap='gray',aspect='auto')
            plt_on = 1
    # 
        case 'B':
            use_ax = axs24[1]
            plt_on = 1
            axs24[1].annotate('B',xy=(-0.1,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
    
        case 'C':
            use_ax = axs24[2]
            plt_on = 1       
            axs24[2].annotate('C',xy=(-0.1,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
    
    if ( plt_on  == 1 ):
         im = use_ax.imshow(np.transpose(rE),cmap='gray',vmin=0,vmax=32,aspect='auto')    
         use_ax.set_ylabel('Unit label')
         use_ax.set_xlabel('Time (sec)')
         use_ax.set_xticks([0,int(round(hstart/dt)),int(round(hend/dt)), int(3*Nt/4),Nt])
         use_ax.set_xticklabels(['0',str(hstart),str(hend),str(int(round(3*tmax/4))), str(int(tmax))])
         cbar = plt.colorbar(im) 
         
         cbar.ax.set_title('Rate (Hz)',size=16)

    if ( question_part == 'B2'):
        fig25,axs25 = plt.subplots(1,5,figsize=(10,2.5))
        plt.tight_layout(pad = 0, h_pad = 2, w_pad = 2)
        
        axs25[0].plot(WEE[int(Ncells/2)-1,:],'k')
        axs25[0].plot([0, Ncells], [0, 0], 'k:')
        axs25[0].set_xlim(0,Ncells)
        axs25[0].set_ylim(-0.35,0.35)
        axs25[0].set_xticks([0,int(Ncells/2),Ncells])
        axs25[0].set_xticklabels([r'-$\pi$/2','0',r'$\pi$/2'])
        axs25[0].set_xlabel('Tuning \n Difference',fontsize=size_of_labels)
        axs25[0].set_ylabel('Connection Weight',fontsize=size_of_labels)
        axs25[0].set_title('E-to-E')
        
        axs25[1].plot(WEI[int(Ncells/2)-1,:],'k')
        axs25[1].plot([0, Ncells], [0, 0], 'k:')
        axs25[1].set_xlim(0,Ncells)
        axs25[1].set_ylim(-0.35,0.35)
        axs25[1].set_xticks([0,int(Ncells/2),Ncells])
        axs25[1].set_xticklabels([r'-$\pi$/2','0',r'$\pi$/2'])
        axs25[1].set_xlabel('Tuning \n Difference',fontsize=size_of_labels)
        axs25[1].set_title('E-to-I')
        
        axs25[2].plot(WIE[int(Ncells/2)-1,:],'k')
        axs25[2].plot([0, Ncells], [0, 0], 'k:')
        axs25[2].set_xlim(0,Ncells)
        axs25[2].set_ylim(-0.35,0.35)
        axs25[2].set_xticks([0,int(Ncells/2),Ncells])
        axs25[2].set_xticklabels([r'-$\pi$/2','0',r'$\pi$/2'])
        axs25[2].set_xlabel('Tuning \n Difference',fontsize=size_of_labels)
        axs25[2].set_title('I-to-E')
        
        W_di_EE = WEI@WIE
        axs25[3].plot(W_di_EE[int(Ncells/2)-1,:],'k')
        axs25[3].plot([0, Ncells], [0, 0], 'k:')
        axs25[3].set_xlim(0,Ncells)
        axs25[3].set_ylim(-0.35,0.35)
        axs25[3].set_xticks([0,int(Ncells/2),Ncells])
        axs25[3].set_xticklabels([r'-$\pi$/2','0',r'$\pi$/2'])
        axs25[3].set_xlabel('Tuning \n Difference',fontsize=size_of_labels)
        axs25[3].set_title('Disynaptic E-to-E')
        
        WEE_tot = WEE + W_di_EE 
        axs25[4].plot(WEE_tot[int(Ncells/2)-1,:],'k')
        axs25[4].plot([0, Ncells], [0, 0], 'k:')
        axs25[4].set_xlim(0,Ncells)
        axs25[4].set_ylim(-0.35,0.35)
        axs25[4].set_xticks([0,int(Ncells/2),Ncells])
        axs25[4].set_xticklabels([r'-$\pi$/2','0',r'$\pi$/2'])
        axs25[4].set_xlabel('Tuning \n Difference',fontsize=size_of_labels)
        axs25[4].set_title('Total E-to-E')
        
        axs25[0].annotate('A',xy=(-0.3,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
        axs25[1].annotate('B',xy=(-0.3,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
        axs25[2].annotate('C',xy=(-0.3,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
        axs25[3].annotate('D',xy=(-0.3,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
        axs25[4].annotate('E',xy=(-0.3,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:33:42 2024

@author: pmiller
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def alt_poissrnd( rate_dt ):
#alt_poissrnd Produces random integers from a Poisson distribution with a
#parameter, lambda = rate_dt that can vary from time bin to time bin.
#
#   rate_dt is an input vector of a time-varying or constant rate,
#   multiplied by the time-step, dt, such that the corresponding bin has on
#   average rate_dt spikes. 
#
#   Output is a vector of integers of the same size as the input vector. 
#   Each element of the output is chosen from the Poisson distribution with
#   paramter given by that element of the input vector.
#
#   This code is required for Tutorial 3.2 (Part B) of Chapter 3 of the
#   textbook,
#   An Introductory Course in Computational Neuroscience
#   by Paul Miller, Brandeis University (2017)
#
###########################################################################

    num_events = np.zeros(len(rate_dt)) # initialize with no events per bin
    exprdt = np.exp(-rate_dt)           # calculated once and used frequently
    rnums = np.random.rand(len(rate_dt))       # random numbers to test probabilities
#
    sum_probs = exprdt                  # cumulative sum of all probabilities
    this_prob = exprdt                  # probability of N events
    indices = np.arange(1,len(rate_dt))    # bins to test
    N = 0                               # number of events, N, in a bin
 #
    while (len(indices) > 0 ):         # while there are bins still to test
        indices = np.array([i for i, x in enumerate(rnums-sum_probs) if x > 0])
        N = N + 1               # increase N by 1.
        num_events[indices] = N     # add the new N to each bin
        this_prob = np.multiply(this_prob,rate_dt)/N    # Poisson prob of new N
        sum_probs = sum_probs + this_prob           # cumulative sum of probs
       
    return num_events 


@jit(nopython=True)
def STA_spatial( Iapp, spikes, dt, tminus=0.075, tplus=0.025 ):
    #STA_spatial Spike-triggered-average with a spatial dimension
    #   Requires arrays and variables as follows:
    #   Iapp(Nspatial,length(t)) noisy applied current to each cell to base response on
    #   spikes(length(t)) spike array for each cell
    #   dt , time step
    
# =============================================================================
#     try: 
#         tminus
#     except NameError:
#         tminus = 0.075 # How long before spike time to begin
#     
#     try: 
#         plus
#     except NameError:
#         tplus = 0.025  # How long after spike time to continue
#     
# =============================================================================
    nminus = int(np.ceil(tminus/dt)) # Number of time points before zero
    nplus = int(np.ceil(tplus/dt))   # Number of time points after zero
    [Nspatial, nt] = np.shape(Iapp)     # length of original data set
    sta = np.zeros((Nspatial,nminus+nplus+1))  # STA will accumulate here
    tcorr = np.arange(-nminus*dt,nplus*dt,dt)   # Vector of time points for STA
    Iapp = Iapp - np.mean(Iapp) # Removes mean applied current
    
    spikeposition = np.nonzero(spikes)
    totalspikes = len(spikeposition[0])
    for spike in range(0,totalspikes):
        ispike = int(spikeposition[0][spike])       # Bin to start measuring stimulus
        imin = max(0,ispike-nminus)       # Bin to start measuring stimulus
        imax = min(nt,ispike+nplus)       # Bin to finish measuring
        # The following lines put the stimulus, Iapp, into bins shifted
        # by the spike time (ispike)
        for i in range(imin,imax ):
            sta[:,i-ispike+nminus+1] = sta[:,i-ispike+nminus+1]  + Iapp[:,i]  
    
    sta = sta/totalspikes
    
    return sta, tcorr 

@jit(nopython=True)
def STA( Iapp, spikes, dt, tminus=0.075, tplus=0.025 ):
    #STA_spatial Spike-triggered-average with a spatial dimension
    #   Requires arrays and variables as follows:
    #   Iapp(Nspatial,length(t)) noisy applied current to each cell to base response on
    #   spikes(length(t)) spike array for each cell
    #   dt , time step
    nminus = int(np.ceil(tminus/dt)) # Number of time points before zero
    nplus = int(np.ceil(tplus/dt))   # Number of time points after zero
    nt = len(Iapp)     # length of original data set
    sta = np.zeros((nminus+nplus+1))  # STA will accumulate here
    tcorr = np.arange(-nminus*dt,(nplus+1)*dt,dt)   # Vector of time points for STA
    Iapp = Iapp - np.mean(Iapp) # Removes mean applied current
    
    spikeposition = np.nonzero(spikes)
    totalspikes = len(spikeposition[0])
    for spike in range(0,totalspikes):
        ispike = int(spikeposition[0][spike])       # Bin to start measuring stimulus
        imin = max(0,ispike-nminus)       # Bin to start measuring stimulus
        imax = min(nt,ispike+nplus)       # Bin to finish measuring
        # The following lines put the stimulus, Iapp, into bins shifted
        # by the spike time (ispike)
        for i in range(imin,imax ):
            sta[i-ispike+nminus+1] = sta[i-ispike+nminus+1]  + Iapp[i]  
    
    sta = sta/totalspikes
    
    return sta, tcorr 

    
@jit(nopython=True)
def expandbin( old_vector, old_dt, new_dt ):
    #expandbin.m takes a vector of values with very finely spaced time points
    #and returns a smaller vector with more coarsely spaced time points
    #   [newvector] = expandbin( old_vector, old_dt, new_dt )
    #
    #   The function requires the following inputs:
    #   old_vector contains the values in original time bins.
    #   These will be averaged to generate values in the new time bins and
    #   returned as new_vector
    #
    #   old_dt is the original time-step used
    #
    #   new_dt is the desired new time-step
    #
    #   When analyzing simulated data this function is useful, because the
    #   simulations may require a very small dt for accuracy and stability, yet
    #   the results may only need to be analyzed at a less fine resolution.
    #
    #   This code is needed for Tutorial 3.1 in Chapter 3 of the textbook
    #   An Introductory Course in Computational Neuroscience
    #   by Paul Miller, Brandeis University, 2017
    #
    ###########################################################################
    
    old_Nt = len(old_vector)
    Nscale = int(np.round(new_dt/old_dt))
    new_Nt = int(np.ceil(old_Nt/Nscale))
    new_vector = np.zeros(new_Nt)
    
    for i in range(0,new_Nt-2):
        new_vector[i] = np.mean(old_vector[i*Nscale:(i+1)*Nscale])

    new_vector[new_Nt-1] = np.mean(old_vector[(new_Nt-1)*Nscale:old_Nt-1])
    
    return new_vector

@jit(nopython=True)
def PR_dend_gating( VmD, Ca ):
#PR_dend_gating returns the rate constants for the dendritic gating variables
#of the Pinsky-Rinzel model, as a function of the membrane potential and
#dendritic calcium concentration.
#
#   [ alpha_mca, beta_mca, alpha_kca, beta_kca, alpha_kmahp, beta_kmahp ] = PR_dend_gating( VmD, Ca )
#   Input VmD should be dendritic membrane potential (a scalar or a vector of
#   values).
#   Input Ca should be dendritic calcium concentration (a scalar or a
#   vector of values).
#   Returned voltage-dependent rate constant arrays
#   (alpha_mca, beta_mca, alpha_kca, beta_kca)
#   are each of the same size
#   as the input array of membrane potentials.
#
#   Returned calcium-dependent rate constants (alpha_kmahp and beta_kmahp)
#   are each the same size as the input array of calcium values.
#
#   alpha_mca is calcium activation rate constant
#   beta_mca is calcium deactivation rate constant
#   alpha_kca is calcium-dependent potassium activation rate constant
#   beta_kca is calcium-dependent potassium deactivation rate constant
#   alpha_kmahp is after-hyperpolarization activation rate constant
#   beta_kmahp is after-hyperpolarization deactivation rate constant

    
    
    if ( isinstance(VmD,(int,float))):
        alpha_mca = 1600/( 1+np.exp(-72*(VmD-0.005)) )
        if ( VmD == -0.0089 ):
            beta_mca = 20/0.2
        else:
            beta_mca = 20e3*(VmD+0.0089)/(np.exp(200*(VmD+0.0089))-1)
            
        if (VmD>-0.010):
            alpha_mkca = 2e3*np.exp(-(0.0535+VmD)/0.027)
            beta_mkca = 0
        else:
            alpha_mkca = np.exp( (VmD+0.050)/0.011 -(VmD+0.0535)/0.027 )/0.018975
            beta_mkca = 2e3*np.exp(-(0.0535+VmD)/0.027)-alpha_mkca
        alpha_mkahp = min(20,20e3*Ca)
        beta_mkahp = 4
    else:
        if isinstance(VmD,list):
            n = len(VmD)
            VmDa = np.empty(n, dtype=np.float64)  # Assuming your list contains floats
            Caa = np.empty(n, dtype=np.float64)  # Assuming your list contains floats
            for i in range(n):
                VmDa[i] = VmD[i]
                Caa[i] = Ca[i]
        else:
            VmDa = VmD
            Caa = Ca
            
        alpha_mca = np.zeros(VmDa.size)
        beta_mca = np.zeros(VmDa.size)
        alpha_mkca = np.zeros(VmDa.size)
        beta_mkca = np.zeros(VmDa.size)
        
        for i in range (0,VmDa.size):
            alpha_mca[i] = 1600/( 1+np.exp(-72*(VmD[i]-0.005)) )
            
        ind1 = np.where(VmDa==-0.0089)
        ind2 = np.where(VmDa!=-0.0089)
        for i in ind1:
            beta_mca[i] = 20/0.2
        for i in ind2: 
            beta_mca[i] = 20e3*(VmDa[i]+0.0089)/(np.exp(200*(VmDa[i]+0.0089))-1)

        ind3 = np.where(VmDa>-0.010)
        ind4 = np.where(VmDa<=-0.010)
        for i in ind3:
            alpha_mkca[i] = 2e3*np.exp(-(0.0535+VmDa[i])/0.027)
        for i in ind4:
            alpha_mkca[i] = np.exp( (VmDa[i]+0.050)/0.011 -(VmDa[i]+0.0535)/0.027 )/0.018975
            beta_mkca[i] = 2e3*np.exp(-(0.0535+VmDa[i])/0.027)-alpha_mkca[i]
            
        alpha_mkahp = 20e3*Caa   
        ind5 = np.where(alpha_mkahp>20)
        for i in ind5:
            alpha_mkahp[i] = 20
        beta_mkahp = 4.0*np.ones(alpha_mkahp.size)

        if isinstance(VmD,list):
            r_alpha_mca = alpha_mca
            r_beta_mca = beta_mca
            r_alpha_mkca = alpha_mkca
            r_beta_mkca = beta_mkca
            r_alpha_mkahp = alpha_mkahp
            r_beta_mkahp = beta_mkahp
            alpha_mca = []
            beta_mca = []
            alpha_mkca = []
            beta_mkca = []
            alpha_mkahp = []
            beta_mkahp = []
            for i in range(0,r_alpha_mca.size):
                alpha_mca.append(r_alpha_mca[i])
                beta_mca.append(r_beta_mca[i])
                alpha_mkca.append(r_alpha_mkca[i])
                beta_mkca.append(r_beta_mkca[i])
                alpha_mkahp.append(r_alpha_mkahp[i])
                beta_mkahp.append(r_beta_mkahp[i])

    return  alpha_mca, beta_mca, alpha_mkca, beta_mkca, alpha_mkahp, beta_mkahp 

@jit(nopython=True)
def PR_soma_gating( Vm ):
#PR_soma_gating returns the rate constants for the somatic gating variables
#of the Pinsky-Rinzel model, as a function of the membrane potential.
#
#   [ alpha_m, beta_m, alpha_h, beta_h , alpha_n, beta_n] = PR_soma_gating( Vm )
#   Input, Vm, should be somatic membrane potential (a scalar or a vector of
#   values).
#   Returned voltage-dependent rate constants are each of the same size as
#   the input membrane potential.
#   alpha_m is sodium activation rate constant
#   beta_m is sodium deactivation rate constant
#   alpha_h is sodium inactivation rate constant
#   beta_h is sodium deinactivation rate constant
#   alpha_n is potassium activation rate constant
#   beta_n is potassium deactivation rate constant


    # Sodium and potassium gating variables are defined by the
    # voltage-dependent transition rates between states, labeled alpha and
    # beta. 
    

    if ( isinstance(Vm,(int,float))):
        alpha_h = 128*np.exp(-(Vm+0.043)/0.018)
        beta_h = 4e3/(1+np.exp(-200*(Vm+0.020)))
        beta_n = 250*np.exp(-25*(Vm+0.040))    

        if ( Vm == -0.0199 ):
            beta_m = 280/0.2
        else:
            beta_m = 280e3*(Vm+0.0199) /(np.exp(200*(Vm+0.0199))-1)                                      
                                         
        if ( Vm == -0.0469 ):
            alpha_m = 320/0.25
        else:
            alpha_m = 320*1e3*(Vm+0.0469) /(1-np.exp(-250*(Vm+0.0469)))
    
        if ( Vm == -0.0249 ):                          
            alpha_n =  16/0.2
        else:
            alpha_n = 16e3*(Vm+0.0249)/(1-np.exp(-200*(Vm+0.0249)))          
    else:
        print('not scalar')
        if isinstance(Vm,list):
            print('list')
            n = len(Vm)
            Vma = np.empty(n, dtype=np.float64)  # Assuming your list contains floats
            for i in range(n):
                Vma[i] = Vm[i]
        else:
            Vma = Vm

        alpha_m = np.zeros(Vma.size)
        beta_m = np.zeros(Vma.size)
        alpha_n = np.zeros(Vma.size)
        beta_n = np.zeros(Vma.size)
        beta_h = np.zeros(Vma.size)
        alpha_h = np.zeros(Vma.size)
        
        for i in range(0,Vma.size):
            alpha_h[i] = 128*np.exp(-(Vma[i]+0.043)/0.018)
            beta_h[i] = 4e3/(1+np.exp(-200*(Vma[i]+0.020)))
            beta_n[i] = 250*np.exp(-25*(Vma[i]+0.040))    

        ind1 = np.where(Vma==-0.0199)
        ind2 = np.where(Vma!=-0.0199)
        for i in ind1:
            beta_m[i] = 280/0.2       
        for i in ind2:
            beta_m[i] = 280e3*(Vma[i]+0.0199) /(np.exp(200*(Vma[i]+0.0199))-1) 

        ind3 = np.where(Vma == -0.0469)
        ind4 = np.where(Vma != -0.0469)
        for i in ind3:
            alpha_m[i] = 320/0.25
        for i in ind4:
            alpha_m[i] = 320*1e3*(Vma[i]+0.0469) /(1-np.exp(-250*(Vma[i]+0.0469)))
        
        ind5 = np.where(Vma == -0.0249 )
        ind6 = np.where(Vma != -0.0249 )
        for i in ind5:
            alpha_n[i] =  16/0.2
        for i in ind6:
            alpha_n[i] = 16e3*(Vma[i]+0.0249)/(1-np.exp(-200*(Vma[i]+0.0249)))   
            
        if isinstance(Vm,list):
            r_alpha_m = alpha_m
            r_beta_m = beta_m
            r_alpha_h = alpha_h
            r_beta_h = beta_h
            r_alpha_n = alpha_n
            r_beta_n = beta_n
            alpha_m = []
            beta_m = []
            alpha_h = []
            beta_h = []
            alpha_n = []
            beta_n = []
            for i in range(0,r_alpha_m.size):
                alpha_m.append(r_alpha_m[i])
                beta_m.append(r_beta_m[i])
                alpha_h.append(r_alpha_h[i])
                beta_h.append(r_beta_h[i])
                alpha_n.append(r_alpha_n[i])
                beta_n.append(r_beta_n[i])
                           
    return alpha_m, beta_m, alpha_h, beta_h , alpha_n, beta_n


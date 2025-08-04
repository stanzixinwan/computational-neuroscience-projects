#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:10:57 2024

@author: pmiller
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def integrate_HH(Iapp,t,dt,V_L,E_Na,E_K,G_Na,G_K,G_L,Cm):
   
    V=np.zeros(len(t))           # membrane potential vector
    n=np.zeros(len(t))       # n: potassium activation gating variable
    m=np.zeros(len(t))       # m: sodium activation gating variable
    h=np.zeros(len(t))       # h: sodim inactivation gating variable
    V[0] = V_L
    h[0] = 0.5
    
    Itot=np.zeros(len(t))    # in case we want to plot and look at the total current
    I_Na=np.zeros(len(t))    # record sodium curret
    I_K=np.zeros(len(t))     # record potassium current
    I_L=np.zeros(len(t))     # record leak current

    for i in range(1,len(t)): # now see how things change through time
        
        Vm = V[i-1]          # membrane potential for calculations
        
        # Sodium and potassium gating variables are defined by the
        # voltage-dependent transition rates between states, labeled alpha and
        # beta.
        
        # First, sodium activation rate
        if ( Vm == -0.045 ):     # to avoid dividing zero by zero
            alpha_m = 1e3      # value calculated analytically
        else:
            alpha_m = (1e5*(-Vm-0.045))/(np.exp(100*(-Vm-0.045))-1)
        
        beta_m = 4000*np.exp((-Vm-0.070)/0.018)   # Sodium deactivation rate
        alpha_h = 70*np.exp(50*(-Vm-0.070))       # Sodium inactivation rate
        beta_h = 1000/(1+np.exp(100*(-Vm-0.040))) # Sodium deinactivation rate
        
        if ( Vm == -0.060):     # to avoid dividing by zero
            alpha_n = 100      # value calculated analytically
        else :                 # potassium activation rate
            alpha_n = (1e4*(-Vm-0.060))/(np.exp(100*(-Vm-0.060))-1)
        
        beta_n = 125*np.exp((-Vm-0.070)/0.08)     # potassium deactivation rate
        
        # From the alpha and beta for each gating variable we find the steady
        # state values (_inf) and the time constants (tau_) for each m,h and n.
        
        tau_m = 1/(alpha_m+beta_m)
        m_inf = alpha_m/(alpha_m+beta_m)
        
        tau_h = 1/(alpha_h+beta_h)
        h_inf = alpha_h/(alpha_h+beta_h)
        
        tau_n = 1/(alpha_n+beta_n)
        n_inf = alpha_n/(alpha_n+beta_n)
        
        
        m[i] = m[i-1] + (m_inf-m[i-1])*dt/tau_m    # Update m
        
        h[i] = h[i-1] + (h_inf-h[i-1])*dt/tau_h    # Update h
        
        n[i] = n[i-1] + (n_inf-n[i-1])*dt/tau_n    # Update n
        
        I_Na[i] = G_Na*m[i]*m[i]*m[i]*h[i]*(E_Na-V[i-1]) # total sodium current
        
        I_K[i] = G_K*n[i]*n[i]*n[i]*n[i]*(E_K-V[i-1]) # total potassium current
        
        I_L[i] = G_L*(V_L-V[i-1])    # Leak current is straightforward
        
        Itot[i] = I_L[i]+I_Na[i]+I_K[i]+Iapp[i] # total current is sum of leak + active channels + applied current
        
        V[i] = V[i-1] + Itot[i]*dt/Cm        # Update the membrane potential, V.
        
    return V

#
# This model is the Connor-Stevens model, similar to Hodgkin-Huxley, but
# more like neurons in the cortex, being type-I.
# See Dayan and Abbott Sect 5.5, pp 166-172
# then Sect 6.1-2, pp.196-198 and Sect 6.6 p.224.
#
# For the original article see:
# Connor JA, Stevens CF, J Physiol 213:31-53 (1971)
#
# This adapted code is used to produce Figure 4.11 in the textbook:
# An Introductory Course in Computational Neuroscience,
# by Paul Miller (Brandeis University, 2017)
#
###########################################################################

@jit(nopython=True)
def CSgating(Vm):
    
    # The function "gating" returns the steady states and the time constants of 
    # the gating variables for the Connor-Stevens model.
    # It should be sent the membrane potential, Vm, in units of Volts.
    #
    #
    # Sodium and potassium gating variables are defined by the
    # voltage-dependent transition rates between states, labeled alpha and
    # beta. Written out from Dayan/Abbott, units are 1/ms.
    
    alpha_m = 3.80e5*(Vm+0.0297)/(1-np.exp(-100*(Vm+0.0297)))
    beta_m = 1.52e4*np.exp(-55.6*(Vm+0.0547))
    
    alpha_h = 266*np.exp(-50*(Vm+0.048))
    beta_h = 3800/(1+np.exp(-100*(Vm+0.018)))
    
    alpha_n = 2e4*(Vm+0.0457)/(1-np.exp(-100*(Vm+0.0457)))
    beta_n = 250*np.exp(-12.5*(Vm+0.0557))
    
    # From the alpha and beta for each gating variable we find the steady
    # state values (_inf) and the time constants (tau_) for each m,h and n.
    
    tau_m = 1/(alpha_m+beta_m)      # time constant converted from ms to sec
    m_inf = alpha_m/(alpha_m+beta_m)
    
    tau_h = 1/(alpha_h+beta_h)      # time constant converted from ms to sec
    h_inf = alpha_h/(alpha_h+beta_h)
    
    tau_n = 1/(alpha_n+beta_n)      # time constant converted from ms to sec
    n_inf = alpha_n/(alpha_n+beta_n)
    
    # For the A-type current gating variables, instead of using alpha and
    # beta, we just use the steady-state values a_inf and b_inf along with
    # the time constants tau_a and tau_b that are found empirically
    a_inf = (0.0761*np.exp(31.4*(Vm+0.09422))/(1+np.exp(34.6*(Vm+0.00117))))**(1/3.0)
    tau_a = 0.3632e-3 + 1.158e-3/(1+np.exp(49.7*(Vm+0.05596)))
    
    b_inf = (1/(1+np.exp(68.8*(Vm+0.0533))))**4
    tau_b = 1.24e-3 + 2.678e-3/(1+np.exp(62.4*(Vm+0.050)))
    
    return m_inf, tau_m, h_inf, tau_h, n_inf, tau_n, a_inf, tau_a, b_inf, tau_b

###############################################################################
@jit(nopython=True)
def integrate_CS(Iapp,t,dt,V_L,E_Na,E_K,E_A,G_Na,G_K,G_A,G_L,Cm):
    
    V=np.zeros(len(t)) # voltage vector
    V[0] = V_L    # set the inititial value of voltage
    
    n=np.zeros(len(t))   # n: potassium activation gating variable
    n[0] = 0.0         # start off at zero
    m=np.zeros(len(t))   # m: sodium activation gating variable
    m[0] = 0.0         # start off at zero
    h=np.zeros(len(t))   # h: sodim inactivation gating variable
    h[0] = 0.0         # start off at zero
    
    a=np.zeros(len(t))   # A-current activation gating variable
    a[0] = 0.0         # start off at zero
    b=np.zeros(len(t))   # A-current inactivation gating variable
    b[0] = 0.0         # start off at zero
    
    Itot=np.zeros(len(t)) # in case we want to plot and look at the total current
    I_Na=np.zeros(len(t))    # to store sodium current vs time 
    I_K=np.zeros(len(t))     # to store potassium current vs time
    I_A=np.zeros(len(t))     # to store A-type current vs time
    I_L=np.zeros(len(t))     # to store leak current vs time
    
    for i in range(1,len(t)): # now see how things change through time
        
        Vm = V[i-1] # converts voltages to mV as needed in the equations on p.224 of Dayan/Abbott
        
        # find the steady state and time constant of all gating variables 
        # give the value of the membrane potential
        [m_inf, tau_m, h_inf, tau_h, n_inf, tau_n, a_inf, tau_a, \
            b_inf, tau_b] = CSgating(Vm)
        
        m[i] = m[i-1] + (m_inf-m[i-1])*dt/tau_m    # Update m
        h[i] = h[i-1] + (h_inf-h[i-1])*dt/tau_h    # Update h
        n[i] = n[i-1] + (n_inf-n[i-1])*dt/tau_n    # Update n
        
        a[i] = a[i-1] + (a_inf-a[i-1])*dt/tau_a    # Update a
        b[i] = b[i-1] + (b_inf-b[i-1])*dt/tau_b    # Update b
        
        I_L[i] = G_L*(V_L-V[i-1])                  # leak current
        
        I_Na[i] = G_Na*m[i]*m[i]*m[i]*h[i]*(E_Na-V[i-1])   # sodium current
        
        I_K[i] = G_K*n[i]*n[i]*n[i]*n[i]*(E_K-V[i-1])      # potassium current
        
        I_A[i] = G_A*a[i]*a[i]*a[i]*b[i]*(E_A-V[i-1])      # A-type current
        
        Itot[i] = I_L[i]+I_Na[i]+I_K[i]+I_A[i]+Iapp[i]     # total current is sum of leak + active channels + applied current
        
        V[i] = V[i-1] + Itot[i]*dt/Cm        # Update the membrane potential, V.
        
             
        
    return V 


###############################################################################


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

###############################################################################

@jit(nopython=True)
def integrate_PR(Iapp,t,dt,E_L,E_Na,E_K,E_Ca,G_Na,G_K,G_Ca,G_KCa,G_KAHP,G_LS,G_LD,G_Link,CmS,CmD,tau_Ca,convert_Ca):
    
    VS=np.zeros(len(t))  # somatic voltage vector
    VD=np.zeros(len(t))  # dendritic voltage vector
    VS[0] = E_L    # set the inititial value of somatic voltage
    VD[0] = E_L    # set the inititial value of dendritic voltage
    
    
    Ca=np.zeros(len(t))  # dendritic calcium level (extra Ca above base level)
    Ca[0] = 0          # initialize with no (extra) Ca in cell.
    I_Ca = np.zeros(len(t))  # calcium current (dendrite)
    
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
        
        G_K_now = G_K*n[i]*n[i]
       
        G_Ca_now = G_Ca*mca[i]*mca[i]
        I_Ca[i] = G_Ca_now*(E_Ca-VD[i-1])  # caclium current in dendrite
        
        if ( Ca[i-1] > 250e-6 ): 
            G_KCa_now = G_KCa*mkca[i]
        else:
            G_KCa_now = G_KCa*mkca[i]*Ca[i-1]/250e-6
                
        G_KAHP_now = G_KAHP*mkahp[i]
        
        gS_Tot = G_LS+G_Na_now+G_K_now+G_Link
        VS_inf = (G_LS*E_L + G_Na_now*E_Na + G_K_now*E_K \
                + VD[i-1]*G_Link )/gS_Tot
                       
        gD_Tot = G_LD+G_Ca_now+G_KCa_now+G_KAHP_now+G_Link
        VD_inf = (G_LD*E_L + G_Ca_now*E_Ca + G_KCa_now*E_K + G_KAHP_now*E_K \
                + VS[i-1]*G_Link +Iapp[i])/gD_Tot
                       
        VS[i] = VS_inf - (VS_inf-VS[i-1])*np.exp(-dt*gS_Tot/CmS)  # Update the membrane potential, V.
        VD[i] = VD_inf - (VD_inf-VD[i-1])*np.exp(-dt*gD_Tot/CmD)  # Update the membrane potential, V.
        Ca_inf = tau_Ca*convert_Ca*I_Ca[i]
        Ca[i] = Ca_inf - (Ca_inf-Ca[i-1])*np.exp(-dt/tau_Ca)  # update Ca level
            
    return VS


###############################################################################

@jit(nopython=True)
def integrate_PRH(Iapp,t,dt,E_L,E_Na,E_K,E_Ca,E_H,G_Na,G_K,G_Ca,G_KCa,G_KAHP,G_H,G_LS,G_LD,G_Link,CmS,CmD,tau_Ca,convert_Ca):
    VS=np.zeros(len(t))  # somatic voltage vector
    VD=np.zeros(len(t))  # dendritic voltage vector
    VS[0] = E_L    # set the inititial value of somatic voltage
    VD[0] = E_L    # set the inititial value of dendritic voltage


    Ca=np.zeros(len(t))  # dendritic calcium level (extra Ca above base level)
    Ca[0] = 0          # initialize with no (extra) Ca in cell.

    I_Ca = np.zeros(len(t))  # calcium current (dendrite)

    n=np.zeros(len(t))   # n: potassium activation gating variable
    m=np.zeros(len(t))   # m: sodium activation gating variable
    h=np.zeros(len(t))   # h: sodim inactivation gating variplot(t,V)able
    n[0] = 0.4         # initialize near steady state at resting potential
    h[0] = 0.5         # initialize near steady state at resting potential

    mca=np.zeros(len(t))     # Ca current activation gating variable
    mkca=np.zeros(len(t))    # K_Ca current activation gating variable
    mkahp = np.zeros(len(t)) # K_AHP current activation gating variable
    mH = np.zeros(len(t))    # I_H activation gating variable
    mkahp[0] = 0.2         # initialize near steady state at resting potential
    mkca[0] = 0.2          # initialize near steady state at resting potential
    Ca[0] = 1e-6           # initialize near steady state at resting potential
    mH[0] = 0.5            # initialize near steady state at resting potential

    for i in range(1,len(t)): # now see how things change through time

        # Take variables from last time-point to update all variables in
        # this time-point ("tmp" stands for temporary)
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
        
        # Update somatic gating variables based on rate constants
        m[i] = mtmp + dt*( alpha_m*(1-mtmp) - beta_m*mtmp )
        h[i] = htmp + dt*( alpha_h*(1-htmp) - beta_h*htmp )
        n[i] = ntmp + dt*( alpha_n*(1-ntmp) - beta_n*ntmp )
        
        # Update dendritic gating variables based on rate constants
        mca[i] = mcatmp + dt*( alpha_mca*(1-mcatmp) - beta_mca*mcatmp )
        mkca[i] = mkcatmp + dt*( alpha_mkca*(1-mkcatmp) - beta_mkca*mkcatmp )
        mkahp[i] = mkahptmp + dt*( alpha_mkahp*(1-mkahptmp) - beta_mkahp*mkahptmp )
        
        # Gating variables for hyperpolarization-activated conductance, G_H
        mH_inf = 1/(1+np.exp((VmD+0.070)/0.006))
        tau_mH = 0.272 + 1.499/(1 + np.exp(-(VmD+0.0422)/0.00873))
        mH[i] = mH[i-1] + (mH_inf-mH[i-1])*dt/tau_mH    # Update mH
   
 #       print([VmD,mH[i],mH[i-1],mH_inf,tau_mH])
        # Now update all conductances and currents
        G_Na_now = G_Na*m[i]*m[i]*h[i]     # instantaneous sodium conductance

        G_K_now = G_K*n[i]*n[i]            # instantaneous potassium conductance

        G_Ca_now = G_Ca*mca[i]*mca[i]      # instantaneous calcium conductance

        # G_KCa depends on both [Ca] and V
        if ( Ca[i-1] > 250e-6 ):
            G_KCa_now = G_KCa*mkca[i]
        else:
            G_KCa_now = G_KCa*mkca[i]*Ca[i-1]/250e-6

        G_KAHP_now = G_KAHP*mkahp[i]           # K_AHP instantaneous conductance

        G_H_now = G_H*mH[i]            # Ih instantaneous conductance
        
        gS_Tot = G_LS+G_Na_now+G_K_now+G_Link
        VS_inf = (G_LS*E_L + G_Na_now*E_Na + G_K_now*E_K \
            + VD[i-1]*G_Link )/gS_Tot
        
        gD_Tot = G_LD+G_Ca_now+G_KCa_now+G_KAHP_now+G_H_now+G_Link
        VD_inf = (G_LD*E_L + G_Ca_now*E_Ca + G_KCa_now*E_K + G_KAHP_now*E_K \
            + G_H_now*E_H + VS[i-1]*G_Link )/gD_Tot
        
        VS[i] = VS_inf - (VS_inf-VS[i-1])*np.exp(-dt*gS_Tot/CmS)  # Update the membrane potential, V.
        VD[i] = VD_inf - (VD_inf-VD[i-1])*np.exp(-dt*gD_Tot/CmD)  # Update the membrane potential, V.
        Ca_inf = tau_Ca*convert_Ca*I_Ca[i]
        Ca[i] = Ca_inf - (Ca_inf-Ca[i-1])*np.exp(-dt/tau_Ca)  # update Ca level

    return VS



@jit(nopython=True)
def integrate_PR_H(Iapp,t,dt,E_L,E_Na,E_K,E_Ca,E_H,G_Na,G_K,G_Ca,G_KCa,G_KAHP,G_H,G_LS,G_LD,G_Link,CmS,CmD,tau_Ca,convert_Ca):
    VS=np.zeros(len(t))  # somatic voltage vector
    VD=np.zeros(len(t))  # dendritic voltage vector
    VS[0] = E_L    # set the inititial value of somatic voltage
    VD[0] = E_L    # set the inititial value of dendritic voltage

    Ca=np.zeros(len(t))  # dendritic calcium level (extra Ca above base level)
    Ca[0] = 0          # initialize with no (extra) Ca in cell.

    I_LD= np.zeros(len(t))   # leak current in dendrite
    I_LS= np.zeros(len(t))   # leak current in soma
    I_Na = np.zeros(len(t))  # sodium current (soma)
    I_K = np.zeros(len(t))   # potassium current (soma)
    I_Ca = np.zeros(len(t))  # calcium current (dendrite)
    I_KAHP = np.zeros(len(t)) # after-hyperpolarization current (dendrite)
    I_KCa = np.zeros(len(t)) # calcium-dependent potassium current (dendrite)
    I_H = np.zeros(len(t))   # hyperpolarization-activated current (dendrite)
    ID= np.zeros(len(t))    # total current in dendrite
    IS= np.zeros(len(t))    # total current in soma
    I_Link= np.zeros(len(t))   # current between compartments

    n=np.zeros(len(t))   # n: potassium activation gating variable
    m=np.zeros(len(t))   # m: sodium activation gating variable
    h=np.zeros(len(t))   # h: sodim inactivation gating variplot(t,V)able
    n[0] = 0.4         # initialize near steady state at resting potential
    h[0] = 0.5         # initialize near steady state at resting potential

    mca=np.zeros(len(t))     # Ca current activation gating variable
    mkca=np.zeros(len(t))    # K_Ca current activation gating variable
    mkahp = np.zeros(len(t)) # K_AHP current activation gating variable
    mH = np.zeros(len(t))    # I_H activation gating variable
    mkahp[0] = 0.2         # initialize near steady state at resting potential
    mkca[0] = 0.2          # initialize near steady state at resting potential
    Ca[0] = 1e-6           # initialize near steady state at resting potential
    mH[0] = 0.5            # initialize near steady state at resting potential

    for i in range(1,len(t)): # now see how things change through time
        I_LS[i] = G_LS*(E_L-VS[i-1])
        I_LD[i] = G_LD*(E_L-VD[i-1])
        
        # Take variables from last time-point to update all variables in
        # this time-point ("tmp" stands for temporary)
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
        
        # Update somatic gating variables based on rate constants
        m[i] = mtmp + dt*( alpha_m*(1-mtmp) - beta_m*mtmp )
        h[i] = htmp + dt*( alpha_h*(1-htmp) - beta_h*htmp )
        n[i] = ntmp + dt*( alpha_n*(1-ntmp) - beta_n*ntmp )
        
        # Update dendritic gating variables based on rate constants
        mca[i] = mcatmp + dt*( alpha_mca*(1-mcatmp) - beta_mca*mcatmp )
        mkca[i] = mkcatmp + dt*( alpha_mkca*(1-mkcatmp) - beta_mkca*mkcatmp )
        mkahp[i] = mkahptmp + dt*( alpha_mkahp*(1-mkahptmp) - beta_mkahp*mkahptmp )
        
        # Gating variables for hyperpolarization-activated conductance, G_H
        mH_inf = 1/(1+np.exp((VmD+0.070)/0.006))
        tau_mH = 0.272 + 1.499/(1 + np.exp(-(VmD+0.0422)/0.00873))
        mH[i] = mH[i-1] + (mH_inf-mH[i-1])*dt/tau_mH    # Update mH
        
 #       print([VmD,mH[i],mH[i-1],mH_inf,tau_mH])
        # Now update all conductances and currents
        G_Na_now = G_Na*m[i]*m[i]*h[i]     # instantaneous sodium conductance
        I_Na[i] = G_Na_now*(E_Na-VS[i-1])  # sodium current in soma
        
        G_K_now = G_K*n[i]*n[i]            # instantaneous potassium conductance
        I_K[i] = G_K_now*(E_K-VS[i-1])     # potassium delayed rectifier current, soma
        
        G_Ca_now = G_Ca*mca[i]*mca[i]      # instantaneous calcium conductance
        I_Ca[i] = G_Ca_now*(E_Ca-VD[i-1])  # caclium current in dendrite
        
        # G_KCa depends on both [Ca] and V
        if ( Ca[i-1] > 250e-6 ):
            G_KCa_now = G_KCa*mkca[i]
        else:
            G_KCa_now = G_KCa*mkca[i]*Ca[i-1]/250e-6
        
        I_KCa[i] = G_KCa_now*(E_K-VD[i-1]) # calcium-dependent potassium current in dendrite
        
        G_KAHP_now = G_KAHP*mkahp[i]           # K_AHP instantaneous conductance
        I_KAHP[i] = G_KAHP_now*(E_K-VD[i-1])   # after-hyperoloarization potassium current in dendrite
        
        G_H_now = G_H*mH[i]            # Ih instantaneous conductance
        I_H[i] = G_H_now*(E_H-VD[i-1]) # hyperpolarization-activated mixed cation current in dendrite
        
        I_Link[i] = G_Link*(VD[i-1]-VS[i-1])   # current from dendrite to soma
        
        IS[i] = I_LS[i]+I_Na[i]+I_K[i]+I_Link[i] # total current in soma
        ID[i] = I_LD[i]+I_Ca[i]+I_KCa[i]+I_KAHP[i]+I_H[i]-I_Link[i] # total current in dendrite
        
        gS_Tot = G_LS+G_Na_now+G_K_now+G_Link
        VS_inf = (G_LS*E_L + G_Na_now*E_Na + G_K_now*E_K \
            + VD[i-1]*G_Link )/gS_Tot
        
        gD_Tot = G_LD+G_Ca_now+G_KCa_now+G_KAHP_now+G_H_now+G_Link
        VD_inf = (G_LD*E_L + G_Ca_now*E_Ca + G_KCa_now*E_K + G_KAHP_now*E_K \
            + G_H_now*E_H + VS[i-1]*G_Link +Iapp[i] )/gD_Tot
        
        VS[i] = VS_inf - (VS_inf-VS[i-1])*np.exp(-dt*gS_Tot/CmS)  # Update the membrane potential, V.
        VD[i] = VD_inf - (VD_inf-VD[i-1])*np.exp(-dt*gD_Tot/CmD)  # Update the membrane potential, V.
        Ca_inf = tau_Ca*convert_Ca*I_Ca[i]
        Ca[i] = Ca_inf - (Ca_inf-Ca[i-1])*np.exp(-dt/tau_Ca)  # update Ca level

    return VS, VD

###############################################################################

@jit(nopython=True)
def integrate_CaT(Iapp,t,dt,E_L,E_Na,E_K,E_Ca,G_Na,G_K,G_CaT,G_L,Cm):
    ## Initialize variables used in the simulation
    I_L= np.zeros(len(t))    # to store leak current
    I_Na= np.zeros(len(t))   # to store sodium current
    I_K= np.zeros(len(t))    # to store potassium current
    I_CaT = np.zeros(len(t)) # to store T-type calcium current

    V=np.zeros(len(t))   # membrane potential vector
    V[0] = E_L     # initialize membrane potential
    n=np.zeros(len(t))   # n: potassium activation gating variable
    m=np.zeros(len(t))   # m: sodium activation gating variable
    h=np.zeros(len(t))   # h: sodim inactivation gating variplot(t,V)able
 
    mca=np.zeros(len(t)) # CaT current activation gating variable 
    hca=np.zeros(len(t)) # CaT current inactivation gating variable
    
    Itot=np.zeros(len(t)) # in case we want to plot and look at the total current

    for i in range(1,len(t)): # now see how things change through time
        Vm = V[i-1] 
        
        # Sodium and potassium gating variables are defined by the
        # voltage-dependent transition rates between states, labeled alpha and
        # beta. Written out from Dayan/Abbott, units are 1/sec.
        if ( Vm == -0.035 ): 
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
        if ( Vm < -0.080 ): 
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

    return V

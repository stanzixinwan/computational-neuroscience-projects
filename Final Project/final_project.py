#!/usr/bin/env python3
"""
run_PR_with_Voltage.py

1) Sweep sodium conductance and plot mean dendritic Ca vs. g_Na (as before)
2) For a chosen g_Na (e.g. the baseline), plot V_soma(t) and V_dend(t)
   so you can inspect the spike/burst waveforms.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from pm_functions import PR_soma_gating, PR_dend_gating


# 1) Simulation parameters
dt    = 2e-6
tmax  = 2.0
t     = np.arange(0.0, tmax, dt)

E_L,  E_Na,   E_K,   E_Ca = -0.060, 0.060, -0.075, 0.080
S_frac = 1/3;   D_frac = 1 - S_frac

G_LS = 5e-9 * S_frac
G_LD = 5e-9 * D_frac

baseline_G_Na = 3e-6 * S_frac
G_K           = 2e-6 * S_frac
G_Ca          = 2e-6 * D_frac
G_KCa         = 2.5e-6 * D_frac
G_KAHP        = 0.04e-6 * D_frac

tau_Ca     = 50e-3
convert_Ca = 5e6 / D_frac

CmS = 100e-12 * S_frac
CmD = 100e-12 * D_frac

# 2) PR integrator
@jit(nopython=True)
def integrate_PR(Iapp, G_Link, gNa_max):
    n_steps = t.size
    VS  = np.empty(n_steps)
    VD  = np.empty(n_steps)
    Ca  = np.empty(n_steps)

    # initial conditions
    VS[0] = E_L
    VD[0] = E_L
    Ca[0] = 1e-6

    m    = np.zeros(n_steps)
    h    = np.zeros(n_steps)
    n    = np.zeros(n_steps)
    mca  = np.zeros(n_steps)
    mkca = np.zeros(n_steps)
    mkahp= np.zeros(n_steps)
    n[0], h[0], mkca[0], mkahp[0] = 0.4, 0.5, 0.2, 0.2

    for i in range(1, n_steps):
        Vs, Vd, Ca_prev = VS[i-1], VD[i-1], Ca[i-1]
        αm, βm, αh, βh, αn, βn = PR_soma_gating(Vs)
        αmca, βmca, αmkca, βmkca, αmkahp, βmkahp = PR_dend_gating(Vd, Ca_prev)

        # update gating
        m[i]    = m[i-1]   + dt*(αm   *(1-m[i-1])   - βm   *m[i-1])
        h[i]    = h[i-1]   + dt*(αh   *(1-h[i-1])   - βh   *h[i-1])
        n[i]    = n[i-1]   + dt*(αn   *(1-n[i-1])   - βn   *n[i-1])
        mca[i]  = mca[i-1] + dt*(αmca *(1-mca[i-1]) - βmca *mca[i-1])
        mkca[i] = mkca[i-1]+ dt*(αmkca*(1-mkca[i-1]) - βmkca*mkca[i-1])
        mkahp[i]= mkahp[i-1]+ dt*(αmkahp*(1-mkahp[i-1]) - βmkahp*mkahp[i-1])

        # conductances
        G_Na_now  = gNa_max * m[i]**2 * h[i]
        G_K_now   = G_K    * n[i]**2
        G_Ca_now  = G_Ca   * mca[i]**2
        if Ca_prev > 250e-6:
            G_KCa_now = G_KCa * mkca[i]
        else:
            G_KCa_now = G_KCa * mkca[i] * (Ca_prev/250e-6)
        G_KAHP_now = G_KAHP * mkahp[i]

        # currents
        I_Na   = G_Na_now *(E_Na - Vs)
        I_K    = G_K_now  *(E_K  - Vs)
        I_Ca   = G_Ca_now *(E_Ca - Vd)
        I_KCa  = G_KCa_now*(E_K - Vd)
        I_KAHP = G_KAHP_now*(E_K - Vd)
        I_Link = G_Link   *(Vd - Vs)

        # voltage updates
        VS[i]   = Vs + dt*((I_Na + I_K + I_Link + Iapp[i]) / CmS)
        VD[i]   = Vd + dt*((I_Ca + I_KCa + I_KAHP - I_Link)  / CmD)

        # Ca update
        Ca_inf = tau_Ca * convert_Ca * I_Ca
        Ca[i]  = Ca_inf - (Ca_inf - Ca_prev)*np.exp(-dt/tau_Ca)

    return VS, VD, Ca


# 3) Main analysis: mean Ca sweep + voltage plotting
def main():
    # Injected current into soma
    Applied_current = 0e-12
    Iapp            = np.full_like(t, Applied_current)
    # Conductance coupling
    G_Link          = 50e-9
    Iapp_pA  = Applied_current * 1e12  # pA
    Glink_nS = G_Link          * 1e9    # nS

    # sweep mean-Ca
    alphas = np.linspace(0.0, 3.0, 20)
    t_start, t_end = 0.50, 0.75
    mean_Ca = np.zeros_like(alphas)
    for i, a in enumerate(alphas):
        _, _, Ca = integrate_PR(Iapp, G_Link, baseline_G_Na * a)
        start = int(t_start / dt)
        end   = int(t_end / dt)
        mean_Ca[i] = np.mean(Ca[start:end])

    # Figure 1: mean-Ca alone
    plt.figure(figsize=(6,4))
    plt.plot(alphas*baseline_G_Na*1e9, mean_Ca*1e6, 'o-')
    plt.xlabel('ḡ$_{Na}$ (nS)')
    plt.ylabel('Mean dendritic [Ca²⁺] (µM)')
    plt.title(f"PR model: mean Ca ({Iapp_pA:.0f} pA soma, {Glink_nS:.0f} nS coupling [1 spike])")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Figures 2+3 combined
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Voltage traces (baseline gNa)
    VSb, VDb, _ = integrate_PR(Iapp, G_Link, baseline_G_Na)
    axes[0].plot(t, VSb*1e3, label='V_soma (mV)')
    axes[0].plot(t, VDb*1e3, label='V_dend (mV)', alpha=0.8)
    axes[0].set_title(f'Voltage traces (ḡ$_{{Na}}$=baseline, {Iapp_pA:.0f} pA soma, {Glink_nS:.0f} nS coupling)')
    axes[0].set_ylabel('Membrane V (mV)')
    axes[0].legend()
    axes[0].grid(True)

    # Somatic voltage for various gNa
    scale_factors = [baseline_G_Na, 1.5, 3]
    for sf, color in zip(scale_factors, ['C0','C1','C2']):
        VSf, _, _ = integrate_PR(Iapp, G_Link, baseline_G_Na * sf)
        axes[1].plot(t, VSf*1e3, label=f'ḡ$_{{Na}}$={baseline_G_Na*sf*1e9:.0f} nS', color=color)
    axes[1].set_title('Effect of varying ḡ$_{Na}$ on somatic voltage')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('V_soma (mV)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

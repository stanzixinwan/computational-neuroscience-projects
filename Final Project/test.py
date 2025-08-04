import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from pm_functions import PR_soma_gating, PR_dend_gating

# Simulation parameters
dt = 2e-6        # time step (s)
tmax = 2.0       # total time (s)
t = np.arange(0, tmax, dt)

# Biophysical constants
E_L, E_Na, E_K, E_Ca = -0.060, 0.060, -0.075, 0.080
S_frac = 1/3
D_frac = 1 - S_frac

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

# Integrator function
@jit(nopython=True)
def integrate_PR(Iapp, G_Link, gNa_max):
    n_steps = t.size
    VS = np.empty(n_steps)
    VD = np.empty(n_steps)
    Ca = np.empty(n_steps)

    # initial states
    VS[0], VD[0] = E_L, E_L
    Ca[0] = 1e-6

    # gating variables
    m = np.zeros(n_steps)
    h = np.zeros(n_steps)
    n_var = np.zeros(n_steps)
    mca = np.zeros(n_steps)
    mkca = np.zeros(n_steps)
    mkahp = np.zeros(n_steps)
    n_var[0], h[0], mkca[0], mkahp[0] = 0.4, 0.5, 0.2, 0.2

    for i in range(1, n_steps):
        Vs, Vd, Ca_prev = VS[i-1], VD[i-1], Ca[i-1]

        # gating rates
        αm, βm, αh, βh, αn, βn = PR_soma_gating(Vs)
        αmca, βmca, αmkca, βmkca, αmkahp, βmkahp = PR_dend_gating(Vd, Ca_prev)

        # update gating
        m[i]    = m[i-1]   + dt*(αm   *(1-m[i-1])   - βm   *m[i-1])
        h[i]    = h[i-1]   + dt*(αh   *(1-h[i-1])   - βh   *h[i-1])
        n_var[i]= n_var[i-1] + dt*(αn  *(1-n_var[i-1]) - βn  *n_var[i-1])
        mca[i]  = mca[i-1] + dt*(αmca *(1-mca[i-1]) - βmca *mca[i-1])
        mkca[i] = mkca[i-1]+ dt*(αmkca*(1-mkca[i-1]) - βmkca*mkca[i-1])
        mkahp[i]= mkahp[i-1]+ dt*(αmkahp*(1-mkahp[i-1]) - βmkahp*mkahp[i-1])

        # conductances
        G_Na_now = gNa_max * m[i]**2 * h[i]
        G_K_now  = G_K    * n_var[i]**2
        G_Ca_now = G_Ca   * mca[i]**2
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
        VS[i] = Vs + dt*((I_Na + I_K + I_Link + Iapp[i]) / CmS)
        VD[i] = Vd + dt*((I_Ca + I_KCa + I_KAHP - I_Link) / CmD)

        # Ca update
        Ca_inf = tau_Ca * convert_Ca * I_Ca
        Ca[i]  = Ca_inf - (Ca_inf - Ca_prev)*np.exp(-dt/tau_Ca)

    return VS, VD, Ca

# Choose injection and coupling
Iapp = np.full_like(t, 100e-12)  # 100 pA to soma
G_Link = 50e-9                   # 50 nS coupling

# Select G_Na scale factors to compare
scale_factors = [2.0, 2.2, 2.4]
colors = ['C0', 'C1', 'C2']

plt.figure(figsize=(8, 4))
for sf, c in zip(scale_factors, colors):
    gNa_val = baseline_G_Na * sf
    VS, VD, _ = integrate_PR(Iapp, G_Link, gNa_val)
    plt.plot(t, VS * 1e3, label=f'ḡ_Na = {gNa_val*1e9:.0f} nS', color=c)

plt.xlim(0.0, 2.0)  # zoom in after transient
plt.xlabel('Time (s)')
plt.ylabel('Somatic V (mV)')
plt.title('Effect of varying ḡ_Na on somatic voltage')
plt.legend()
plt.tight_layout()
plt.show()

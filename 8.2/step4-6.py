import numpy as np
import matplotlib.pyplot as plt

# Reuse functions from steps 1-5
def generate_spike_trains(n_in, rates, dt):
    return (np.random.rand(n_in, rates.shape[1]) < (rates * dt)).astype(float)

def compute_gsyn(spikes, G, dt, tau_syn):
    n_ts = spikes.shape[1]
    g_syn = np.zeros(n_ts)
    for t in range(1, n_ts):
        g_syn[t] = g_syn[t-1] * np.exp(-dt/tau_syn) + np.sum(G * spikes[:, t])
    return g_syn

def simulate_LIF(g_syn, time, dt, C_m, g_L, E_L, E_syn, V_th, V_reset, t_ref):
    V = np.full_like(time, E_L)
    last_spike = -np.inf
    spike_times = []
    for i in range(1, len(time)):
        t = time[i]
        if t < last_spike + t_ref:
            V[i] = V_reset
        else:
            dV = (g_L * (E_L - V[i-1]) + g_syn[i-1] * (E_syn - V[i-1])) / C_m
            V[i] = V[i-1] + dt * dV
            if V[i] >= V_th:
                V[i] = V_reset
                last_spike = t
                spike_times.append(t)
    return V, spike_times

def apply_STDP(spikes, spike_times, G, dt, ΔG_LTP, ΔG_LTD, tau_LTP, tau_LTD, G_max):
    for i in range(spikes.shape[0]):
        pre_idxs = np.where(spikes[i] == 1)[0]
        for idx in pre_idxs:
            t_pre = idx * dt
            for t_post in spike_times:
                if t_pre < t_post:
                    G[i] += ΔG_LTP * np.exp((t_pre - t_post) / tau_LTP)
                else:
                    G[i] -= ΔG_LTD * np.exp((t_post - t_pre) / tau_LTD)
        G[i] = np.clip(G[i], 0, G_max)
    return G

# Simulation parameters
dt    = 0.1e-3
T     = 0.5
time  = np.arange(0, T, dt)
n_ts  = len(time)
n_in  = 50

r_max = 60
nu    = 20

C_m     = 200e-12
tau_m   = 20e-3
g_L     = C_m / tau_m
E_L     = -70e-3
E_syn   = 0e-3
V_th    = -54e-3
V_reset = -80e-3
t_ref   = 5e-3

ΔG_LTP  = 20e-12
ΔG_LTD  = 25e-12
tau_LTP = 20e-3
tau_LTD = 20e-3
G_max   = 2e-9
tau_syn = 2e-3

n_trials = 200

# a) Uncorrelated stimuli: each input has its own random phase
phi_i = np.random.uniform(0, 2 * np.pi, size=n_in)
mean_UA, mean_UB = [], []

for trial in range(1, n_trials + 1):
    # Build rate matrix with independent phases
    rates = np.array([(r_max/2) * (1 + np.sin(2 * np.pi * nu * time + phi))
                      for phi in phi_i])
    
    # Initialize synaptic strengths (pS -> S)
    G = np.random.normal(500, 25, size=n_in) * 1e-12
    
    # Run trial steps
    spikes = generate_spike_trains(n_in, rates, dt)
    g_syn  = compute_gsyn(spikes, G, dt, tau_syn)
    V, spike_times = simulate_LIF(g_syn, time, dt, C_m, g_L, E_L, E_syn, V_th, V_reset, t_ref)
    G = apply_STDP(spikes, spike_times, G, dt, ΔG_LTP, ΔG_LTD, tau_LTP, tau_LTD, G_max)
    
    # Record means in nS
    mean_UA.append(np.mean(G[:25]) * 1e9)
    mean_UB.append(np.mean(G[25:]) * 1e9)
    
    # Plot every 20 trials
    if trial % 20 == 0:
        plt.figure(figsize=(6,3))
        plt.plot(G * 1e9, 'o-')
        plt.title(f'Uncorrelated Trial {trial}')
        plt.xlabel('Input Index')
        plt.ylabel('G (nS)')
        plt.tight_layout()
        plt.show()

# Summary of uncorrelated case
plt.figure(figsize=(6,4))
plt.plot(mean_UA, label='Subset A (uncorr)')
plt.plot(mean_UB, label='Subset B (uncorr)')
plt.xlabel('Trial')
plt.ylabel('Mean G (nS)')
plt.legend()
plt.title('Mean Synaptic Strengths (Uncorrelated Inputs)')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming phi_i and G from your Step 6 are still in scope

# Convert phase offsets to multiples of π/2
x_units = phi_i / (np.pi / 2)

plt.figure(figsize=(6,4))
plt.scatter(x_units, G * 1e9)
plt.xlabel('Phase offset (multiples of π/2)')
plt.ylabel('Final synaptic strength G (nS)')
plt.title('Final G vs. Phase Offset (Uncorrelated Inputs)')
plt.xticks(np.arange(0, 5), [f'{i}π/2' for i in range(5)])
plt.grid(True)
plt.tight_layout()
plt.show()

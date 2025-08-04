import numpy as np
import matplotlib.pyplot as plt

# 1) GENERATION OF INPUT SPIKE TRAINS AND SYNAPTIC CONDUCTANCE

# Simulation parameters
dt    = 0.1e-3      # time step: 0.1 ms
T     = 0.5         # trial duration: 0.5 s
time  = np.arange(0, T, dt)
n_ts  = len(time)   # time steps
n_in  = 50          # total inputs (25 in subset a, 25 in subset b)

# a) Define time-varying rates
r_max = 60          # Hz
nu    = 20          # Hz
phi_a = 0
phi_b = np.pi

ra = (r_max/2) * (1 + np.sin(2*np.pi*nu*time + phi_a))
rb = (r_max/2) * (1 + np.sin(2*np.pi*nu*time + phi_b))

rates = np.zeros((n_in, n_ts))
rates[:25, :] = ra
rates[25:, :] = rb

# b) Generate inhomogeneous Poisson spike trains
spikes = (np.random.rand(n_in, n_ts) < (rates * dt)).astype(float)

# c) Initial synaptic strengths
G_pS = np.random.normal(loc=500, scale=25, size=n_in)   # mean=500 pS, σ=25 pS
G    = G_pS * 1e-12                                    # convert to S

# d) Compute total excitatory conductance g_syn(t)
tau_syn = 2e-3   # 2 ms decay constant
g_syn   = np.zeros(n_ts)
for t in range(1, n_ts):
    g_syn[t] = g_syn[t-1] * np.exp(-dt/tau_syn) + np.sum(G * spikes[:, t])

# 2) SIMULATION OF A LIF NEURON RECEIVING THESE INPUTS

# LIF parameters (example values; adjust as needed)
C_m = 200e-12    # 200 pF
tau_m = 20e-3      # 20 ms → g_L = C_m / tau_m
g_L = C_m / tau_m
E_L = -70e-3     # resting potential
E_syn = 0e-3     # excitatory reversal
V_th = -54e-3     # spike threshold
V_reset = -80e-3     # reset potential
t_ref = 5e-3     # 5 ms refractory

# Initialize membrane potential and recording
V = np.full(n_ts, E_L)
last_spike = -np.inf
spike_times = []

for i in range(1, n_ts):
    t = time[i]
    # check refractory
    if t < last_spike + t_ref:
        V[i] = V_reset
    else:
        # Euler update
        dV = ( g_L*(E_L - V[i-1])
             + g_syn[i-1]*(E_syn - V[i-1]) ) / C_m
        V[i] = V[i-1] + dt * dV
        # spike condition
        if V[i] >= V_th:
            V[i] = V_reset
            last_spike = t
            spike_times.append(t)

#plot results
plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(time, g_syn*1e9)
plt.ylabel('g_syn (nS)')
plt.subplot(2,1,2)
plt.plot(time, V*1e3)
plt.ylabel('V (mV)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()



# 3) UPDATE SYNAPTIC STRENGTHS USING BATCH STDP
# ---------------------------------------------

 # Parameters for STDP
ΔG_LTP = 20e-12   
delta_G_LTD = 25e-12
tau_LTP = 20e-3
tau_LTD = 20e-3
G_max = 2e-9     # 2 nS


# Apply STDP for all pre-post spike pairs
for i in range(n_in):
    pre_idxs = np.where(spikes[i] == 1)[0]
    for idx in pre_idxs:
        t_pre = idx * dt
        for t_post in spike_times:
            if t_pre < t_post:
                G[i] += ΔG_LTP * np.exp((t_pre - t_post) / tau_LTP)
            elif t_pre > t_post:
                G[i] -= delta_G_LTD * np.exp((t_post - t_pre) / tau_LTD)
    # clamp to [0, G_max]
    G[i] = np.clip(G[i], 0, G_max)

# 3c) Record and plot synaptic strengths
mean_A = np.mean(G[:25])
mean_B = np.mean(G[25:])    

print(f"Mean G subset A: {mean_A*1e9:.2f} nS, subset B: {mean_B*1e9:.2f} nS")

plt.figure(figsize=(8,4))
plt.plot(G * 1e9, 'o-')
plt.xlabel('Input index')
plt.ylabel('Synaptic strength (nS)')
plt.title('Updated synaptic strengths after STDP')
plt.show()

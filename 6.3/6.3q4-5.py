import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.0001
T = 1.5
tvec = np.arange(0, T, dt)
Nt = len(tvec)
transient_idx = int(0.5 / dt)

# Constants for firing-rate model
Vth = -0.050
Vreset = -0.080
delta_V = Vth - Vreset
sigma_V = 0.001
a = 1
tau = 0.003
E_L = -0.070
E_I = -0.065
G_L = 50e-12

# Synaptic parameters
alpha = 0.2
Wee = 25e-9
Wei = 4e-9
Wie = 800e-9
Wii = 0
tau_sE = 0.002
tau_sI = 0.005

# Connectivity matrices
Wee_mat = np.array([[Wee, 0], [Wei, 0]])
Wei_mat = np.array([[0, Wie], [0, Wii]])

# Firing rate function
def firing_rate(Vss):
    if np.isclose(Vss, Vth):
        return sigma_V / (tau * delta_V * a)
    else:
        return (Vss - Vth) / (tau * delta_V * (1 - np.exp(-a * (Vss - Vth) / sigma_V)))

def run_sim(G_in1, G_in2):
    r = np.zeros((2, Nt))
    sE = np.zeros((2, Nt))
    sI = np.zeros((2, Nt))

    G_stim = np.zeros((2, Nt))
    G_stim[0, :] = G_in1
    G_stim[1, :] = G_in2

    for i in range(1, Nt):
        G_E = Wee_mat @ sE[:, i-1]
        G_I = Wei_mat @ sI[:, i-1]
        G_total = G_L + G_E + G_I + G_stim[:, i]
        Vss = (G_L * E_L + G_I * E_I) / G_total

        for j in range(2):
            r[j, i] = r[j, i-1] * (1 - dt / tau) + dt / tau * firing_rate(Vss[j])

        sE[:, i] = sE[:, i-1] * (1 - dt / tau_sE) + dt * alpha * r[:, i-1] * (1 - sE[:, i-1])
        sI[:, i] = sI[:, i-1] * (1 - dt / tau_sI) + dt * alpha * r[:, i-1] * (1 - sI[:, i-1])

    return r

def analyze_r(r):
    r1 = r[0, transient_idx:]
    r2 = r[1, transient_idx:]

    r1_amp = np.max(r1) - np.min(r1)
    r2_amp = np.max(r2) - np.min(r2)
    mean_r1 = np.mean(r1)
    mean_r2 = np.mean(r2)

    high = 0.75 * np.max(r1) + 0.25 * np.min(r1)
    low = 0.25 * np.max(r1) + 0.75 * np.min(r1)
    crossings = []
    above = False

    for i in range(len(r1)):
        if not above and r1[i] > high:
            crossings.append(i)
            above = True
        elif r1[i] < low:
            above = False

    if len(crossings) > 1:
        period = (crossings[-1] - crossings[0]) * dt / (len(crossings) - 1)
        freq = 1 / period
    else:
        freq = 0

    return freq, r1_amp, r2_amp, mean_r1, mean_r2

# Choose question
question = 5
if question == 4:
    G1_range = np.linspace(0, 10e-9, 21)
    G2_range = np.zeros_like(G1_range)
    x_label = 'G_in^(1) (nS)'
    x_vals = G1_range * 1e9
else:
    G2_range = np.linspace(0, 25e-12, 11)
    G1_range = np.full_like(G2_range, 2e-9)
    x_label = 'G_in^(2) (pS)'
    x_vals = G2_range * 1e12

# Run simulations
freqs = []
amps1 = []
amps2 = []
means1 = []
means2 = []

for G1, G2 in zip(G1_range, G2_range):
    r = run_sim(G1, G2)
    f, a1, a2, m1, m2 = analyze_r(r)
    freqs.append(f)
    amps1.append(a1)
    amps2.append(a2)
    means1.append(m1)
    means2.append(m2)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(6, 10))
plt.tight_layout(pad=0, h_pad=2)

axs[0].plot(x_vals, freqs, 'k')
axs[0].set_ylabel('Oscillation\nfrequency (Hz)')
axs[0].set_xlabel(x_label)

axs[1].plot(x_vals, amps1, 'k', label='E unit')
axs[1].plot(x_vals, amps2, 'k--', label='I unit')
axs[1].set_ylabel('Oscillation\namplitude')
axs[1].set_xlabel(x_label)
axs[1].legend()

axs[2].plot(x_vals, means1, 'k', label='E unit')
axs[2].plot(x_vals, means2, 'k--', label='I unit')
axs[2].set_ylabel('Mean\nfiring rate (Hz)')
axs[2].set_xlabel(x_label)
axs[2].legend()

plt.show()

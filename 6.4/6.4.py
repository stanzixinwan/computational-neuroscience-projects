import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.0001  # s
tmax = 0.3  # s
t = np.arange(0, tmax, dt)
Nt = len(t)
N = 50  # number of units per population
theta = np.pi * np.arange(N) / N
cue_angle = np.pi / 2

# Contrast levels
contrasts = [0, 0.25, 0.5, 0.75, 1.0]

# Indices of interest
idx_pi_2 = np.argmin(np.abs(theta - np.pi / 2))
idx_pi = np.argmin(np.abs(theta - np.pi))

# Network parameters
network_params = {
    "A": {"I0E": -2, "I0I": -4, "AE": 100, "AI": 0, "tauE": 0.05, "tauI": 0.005},
    "B": {"I0E": 1, "I0I": -4, "AE": 100, "AI": 0, "tauE": 0.05, "tauI": 0.005},
    "C": {"I0E": 2, "I0I": 0.5, "AE": 100, "AI": 0, "tauE": 0.05, "tauI": 0.005}
}

results = {}

# Run simulations for each network
for net_name, params in network_params.items():
    I0E = params["I0E"]
    I0I = params["I0I"]
    AE = params["AE"]
    AI = params["AI"]
    tauE = params["tauE"]
    tauI = params["tauI"]

    # Build connectivity
    WEE = np.zeros((N, N))
    WEI = np.zeros((N, N))
    WIE = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            WEE[i, j] = 5 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N
            WEI[i, j] = 3 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N
            WIE[i, j] = -4 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N

    rE_dict = {}
    rI_dict = {}

    for c in contrasts:
        SE = AE * c * (1 + 0.5 * np.cos(2 * (cue_angle - theta)))
        SI = AI * c * (1 + 0.5 * np.cos(2 * (cue_angle - theta)))

        rE = np.zeros((Nt, N))
        rI = np.zeros((Nt, N))

        for i in range(1, Nt):
            input_E = SE + rE[i - 1, :] @ WEE + rI[i - 1, :] @ WIE + I0E
            input_I = SI + rE[i - 1, :] @ WEI + I0I

            rE[i, :] = rE[i - 1, :] + dt / tauE * (-rE[i - 1, :] + input_E)
            rI[i, :] = rI[i - 1, :] + dt / tauI * (-rI[i - 1, :] + input_I)

            rE[i, rE[i, :] < 0] = 0
            rI[i, rI[i, :] < 0] = 0

        rE_dict[c] = rE
        rI_dict[c] = rI

    results[net_name] = {"rE": rE_dict, "rI": rI_dict}

# Heatmap comparison for networks A, B, C
fig, axs = plt.subplots(3, 5, figsize=(18, 9), sharex=True, sharey=True)

for r, net_name in enumerate(["A", "B", "C"]):
    for c_idx, c in enumerate(contrasts):
        rE_map = results[net_name]["rE"][c]
        axs[r, c_idx].imshow(rE_map.T, cmap='hot', aspect='auto', extent=[0, tmax, N, 0])
        axs[r, c_idx].set_title(f'{net_name}: c={c}')
        axs[r, c_idx].set_xlabel('Time (s)')
        if c_idx == 0:
            axs[r, c_idx].set_ylabel(f'{net_name}\nNeuron index')

plt.tight_layout()
plt.show()

# Question 2: Time-varying firing rates at θ = π/2 and θ = π (inhibitory)
fig, axs = plt.subplots(3, 5, figsize=(18, 9), sharex=True, sharey=True)

for r, net_name in enumerate(["A", "B", "C"]):
    for c_idx, c in enumerate(contrasts):
        rI_map = results[net_name]["rI"][c]
        axs[r, c_idx].imshow(rI_map.T, cmap='hot', aspect='auto', extent=[0, tmax, N, 0])
        axs[r, c_idx].set_title(f'{net_name}: c={c}')
        axs[r, c_idx].set_xlabel('Time (s)')
        if c_idx == 0:
            axs[r, c_idx].set_ylabel(f'{net_name} Neuron index')

plt.tight_layout()
plt.show()

# Question 3: Final-timepoint firing rates (excitatory)
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for i, net_name in enumerate(["A", "B", "C"]):
    rE_matrix = np.array([results[net_name]["rE"][c][-1, :] for c in contrasts]).T
    axs[i].imshow(rE_matrix, cmap='hot', aspect='auto', extent=[contrasts[0], contrasts[-1], N, 0])
    axs[i].set_title(f'{net_name} Excitatory')
    axs[i].set_xlabel('Contrast')
    axs[i].set_ylabel('Neuron index')
plt.tight_layout()
plt.show()

# Question 3: Final-timepoint firing rates (inhibitory)
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for i, net_name in enumerate(["A", "B", "C"]):
    rI_matrix = np.array([results[net_name]["rI"][c][-1, :] for c in contrasts]).T
    axs[i].imshow(rI_matrix, cmap='hot', aspect='auto', extent=[contrasts[0], contrasts[-1], N, 0])
    axs[i].set_title(f'{net_name} Inhibitory')
    axs[i].set_xlabel('Contrast')
    axs[i].set_ylabel('Neuron index')
plt.tight_layout()
plt.show()

# Question 4: Normalize each column by mean (excitatory)
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for i, net_name in enumerate(["A", "B", "C"]):
    valid_contrasts = [c for c in contrasts if c > 0]
    rE_matrix = np.array([results[net_name]["rE"][c][-1, :] for c in valid_contrasts]).T
    col_mean = np.mean(rE_matrix, axis=0, keepdims=True)
    threshold = 0.1
    col_mean[col_mean < threshold] = 1
    rE_norm = rE_matrix / col_mean
    axs[i].imshow(rE_norm, cmap='hot', aspect='auto', extent=[valid_contrasts[0], valid_contrasts[-1], N, 0])
    axs[i].set_title(f'{net_name} Excitatory (Norm by Mean)')
    axs[i].set_xlabel('Contrast')
    axs[i].set_ylabel('Neuron index')
plt.tight_layout()
plt.show()

# Question 4: Normalize each column by mean (inhibitory)
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for i, net_name in enumerate(["A", "B", "C"]):
    valid_contrasts = [c for c in contrasts if c > 0]
    rI_matrix = np.array([results[net_name]["rI"][c][-1, :] for c in valid_contrasts]).T
    col_mean = np.mean(rI_matrix, axis=0, keepdims=True)
    col_mean[col_mean < threshold] = 1
    rI_norm = rI_matrix / col_mean
    axs[i].imshow(rI_norm, cmap='hot', aspect='auto', extent=[valid_contrasts[0], valid_contrasts[-1], N, 0])
    axs[i].set_title(f'{net_name} Inhibitory (Norm by Mean)')
    axs[i].set_xlabel('Contrast')
    axs[i].set_ylabel('Neuron index')
plt.tight_layout()
plt.show()


# Question 5: Heatmaps of firing rate and inhibitory input across orientations
cue_angles = np.linspace(0, np.pi, 50)
fig_e, axs_e = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig_i, axs_i = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for net_idx, net_name in enumerate(["A", "B", "C"]):
    rE_by_angle = []
    inh_by_angle = []

    I0E = network_params[net_name]["I0E"]
    I0I = network_params[net_name]["I0I"]
    AE = network_params[net_name]["AE"]
    AI = network_params[net_name]["AI"]
    tauE = network_params[net_name]["tauE"]
    tauI = network_params[net_name]["tauI"]

    WEE = np.zeros((N, N))
    WEI = np.zeros((N, N))
    WIE = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            WEE[i, j] = 5 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N
            WEI[i, j] = 3 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N
            WIE[i, j] = -4 * (1 + np.cos(2 * (theta[i] - theta[j]))) / N

    for angle in cue_angles:
        SE = AE * 1.0 * (1 + 0.5 * np.cos(2 * (angle - theta)))
        SI = AI * 1.0 * (1 + 0.5 * np.cos(2 * (angle - theta)))

        rE = np.zeros((Nt, N))
        rI = np.zeros((Nt, N))

        for i in range(1, Nt):
            input_E = SE + rE[i - 1, :] @ WEE + rI[i - 1, :] @ WIE + I0E
            input_I = SI + rE[i - 1, :] @ WEI + I0I

            rE[i, :] = rE[i - 1, :] + dt / tauE * (-rE[i - 1, :] + input_E)
            rI[i, :] = rI[i - 1, :] + dt / tauI * (-rI[i - 1, :] + input_I)

            rE[i, rE[i, :] < 0] = 0
            rI[i, rI[i, :] < 0] = 0

        rE_by_angle.append(rE[-1, :])
        inh_by_angle.append(input_I)

    rE_matrix = np.array(rE_by_angle).T
    inh_matrix = np.array(inh_by_angle).T

    # Normalize each neuron's tuning curve by its maximum (row-wise)
    rE_matrix /= np.max(rE_matrix, axis=1, keepdims=True) + 1e-6
    inh_matrix /= np.max(inh_matrix, axis=1, keepdims=True) + 1e-6

    axs_e[net_idx].imshow(rE_matrix, cmap='hot', aspect='auto', extent=[0, 180, N, 0])
    axs_e[net_idx].set_title(f'{net_name} Excitatory')
    axs_e[net_idx].set_xlabel("Cue angle (deg)")
    axs_e[net_idx].set_ylabel("Neuron index")

    axs_i[net_idx].imshow(inh_matrix, cmap='hot', aspect='auto', extent=[0, 180, N, 0])
    axs_i[net_idx].set_title(f'{net_name} Inhibitory Input')
    axs_i[net_idx].set_xlabel("Cue angle (deg)")
    axs_i[net_idx].set_ylabel("Neuron index")

fig_e.tight_layout()
fig_i.tight_layout()
plt.show()

# Question 6: Tuning curve of a single neuron across cue angles
single_neuron_index = 25  # Chose neuron index 25
plt.figure(figsize=(6, 4))
plt.plot(np.degrees(cue_angles), rE_matrix[single_neuron_index], label='Excitatory response')
plt.plot(np.degrees(cue_angles), inh_matrix[single_neuron_index], label='Inhibitory input')
plt.title(f'Tuning curve of neuron {single_neuron_index}')
plt.xlabel('Cue angle (degrees)')
plt.ylabel('Normalized response')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





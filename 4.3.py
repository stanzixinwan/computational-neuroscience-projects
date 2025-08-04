import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/stanw/OneDrive/document/Brandeis/NBIO 136/Packages')
from pm_functions import PR_dend_gating, PR_soma_gating
from numba import jit

# Define the range of membrane potential (Vm) from -0.085V to 0.050V
Vm_values = np.linspace(-0.085, 0.050, 200)  # 200 points

# Define the range of calcium concentration (Ca) from 0 to 2 * 10^-3 M
Ca_values = np.linspace(0, 2e-3, 200)

# Compute rate constants using PR_dend_gating function
alpha_mca, beta_mca, alpha_kca, beta_kca, alpha_kmahp, beta_kmahp = PR_dend_gating(Vm_values, Ca_values)

# Plot results
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

axs[0, 0].plot(Vm_values, alpha_mca, label="α_mca")
axs[0, 0].set_title("Calcium Activation Rate")
axs[0, 1].plot(Vm_values, beta_mca, label="β_mca")
axs[0, 1].set_title("Calcium Deactivation Rate")
axs[0, 2].plot(Vm_values, alpha_kca, label="α_kca")
axs[0, 2].set_title("Potassium Activation Rate")

axs[1, 0].plot(Vm_values, beta_kca, label="β_kca")
axs[1, 0].set_title("Potassium Deactivation Rate")
axs[1, 1].plot(Ca_values, alpha_kmahp, label="α_kmahp")
axs[1, 1].set_title("AHP Activation Rate")
axs[1, 2].plot(Ca_values, beta_kmahp, label="β_kmahp")
axs[1, 2].set_title("AHP Deactivation Rate")

for ax in axs.flat:
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# Compute rate constants using PR_soma_gating function
alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = PR_soma_gating(Vm_values)

# Create subplots for each rate constant
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

axs[0, 0].plot(Vm_values, alpha_m, label="α_m (Sodium Activation)")
axs[0, 0].set_title("Sodium Activation Rate")

axs[0, 1].plot(Vm_values, beta_m, label="β_m (Sodium Deactivation)")
axs[0, 1].set_title("Sodium Deactivation Rate")

axs[0, 2].plot(Vm_values, alpha_h, label="α_h (Sodium Inactivation)")
axs[0, 2].set_title("Sodium Inactivation Rate")

axs[1, 0].plot(Vm_values, beta_h, label="β_h (Sodium Deinactivation)")
axs[1, 0].set_title("Sodium Deinactivation Rate")

axs[1, 1].plot(Vm_values, alpha_n, label="α_n (Potassium Activation)")
axs[1, 1].set_title("Potassium Activation Rate")

axs[1, 2].plot(Vm_values, beta_n, label="β_n (Potassium Deactivation)")
axs[1, 2].set_title("Potassium Deactivation Rate")

# Format plots
for ax in axs.flat:
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()


# Simulation Parameters
dt = 2e-6  
tmax = 2   
t = np.arange(0, tmax, dt)  # Time vector

# Reversal Potentials
E_L = -0.060  # Leak reversal potential (V)
E_Na = 0.060  # Sodium (V)
E_K = -0.075  # Potassium (V)
E_Ca = 0.080  # Calcium (V)

# Soma & Dendrite Properties
S_frac = 1/3  # Soma area fraction
D_frac = 1 - S_frac  # Dendrite area fraction

# Conductances (Siemens)
G_LS = 5e-9 * S_frac  # Leak conductance (Soma)
G_Na = 3e-6 * S_frac  # Sodium conductance
G_K = 2e-6 * S_frac  # Potassium conductance

G_LD = 5e-9 * D_frac  # Leak conductance (Dendrite)
G_Ca = 2e-6 * D_frac  # Calcium conductance
G_KAHP = 0.04e-6 * D_frac  # AHP Potassium
G_KCa = 2.5e-6 * D_frac  # Calcium-dependent Potassium

G_Link = 50e-9  # Conductance linking soma & dendrite

# Calcium dynamics
tau_Ca = 50e-3  # Calcium buffering time constant (s)
convert_Ca = 5e6 / D_frac  # Converts calcium charge entry to concentration

# Membrane Capacitance
CmS = 100e-12 * S_frac  # Soma capacitance (Farads)
CmD = 100e-12 * D_frac  # Dendrite capacitance (Farads)

@jit(nopython=True)
def integrate_PR(Iapp, G_Link):
    # Initialize variables
    VS = np.zeros(len(t))  # Soma membrane potential
    VD = np.zeros(len(t))  # Dendrite membrane potential
    VS[0] = E_L  # Initial soma voltage
    VD[0] = E_L  # Initial dendritic voltage

    Ca = np.zeros(len(t))  # Dendritic calcium level
    Ca[0] = 1e-6  # Initial Ca

    # Ionic Currents
    I_Na = np.zeros(len(t))  # Sodium current
    I_K = np.zeros(len(t))  # Potassium current
    I_Ca = np.zeros(len(t))  # Calcium current
    I_KCa = np.zeros(len(t))  # Ca-dependent Potassium
    I_KAHP = np.zeros(len(t))  # AHP Potassium
    I_Link = np.zeros(len(t))  # Current between soma & dendrite

    # Gating variables
    n = np.zeros(len(t))
    m = np.zeros(len(t))
    h = np.zeros(len(t))
    mca = np.zeros(len(t))
    mkca = np.zeros(len(t))
    mkahp = np.zeros(len(t))

    n[0] = 0.4
    h[0] = 0.5
    mkahp[0] = 0.2
    mkca[0] = 0.2

    # Time evolution
    for i in range(1, len(t)):
        VmS = VS[i-1]
        VmD = VD[i-1]
        Ca_prev = Ca[i-1]

        # Compute gating variables
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = PR_soma_gating(VmS)
        alpha_mca, beta_mca, alpha_mkca, beta_mkca, alpha_mkahp, beta_mkahp = PR_dend_gating(VmD, Ca_prev)

        # Update gating variables
        m[i] = m[i-1] + dt * (alpha_m * (1 - m[i-1]) - beta_m * m[i-1])
        h[i] = h[i-1] + dt * (alpha_h * (1 - h[i-1]) - beta_h * h[i-1])
        n[i] = n[i-1] + dt * (alpha_n * (1 - n[i-1]) - beta_n * n[i-1])

        mca[i] = mca[i-1] + dt * (alpha_mca * (1 - mca[i-1]) - beta_mca * mca[i-1])
        mkca[i] = mkca[i-1] + dt * (alpha_mkca * (1 - mkca[i-1]) - beta_mkca * mkca[i-1])
        mkahp[i] = mkahp[i-1] + dt * (alpha_mkahp * (1 - mkahp[i-1]) - beta_mkahp * mkahp[i-1])

        # Compute Conductances
        G_Na_now = G_Na * m[i] ** 2 * h[i]
        G_K_now = G_K * n[i] ** 2
        G_Ca_now = G_Ca * mca[i] ** 2

        if Ca[i-1] > 250e-6:
            G_KCa_now = G_KCa * mkca[i]
        else:
            G_KCa_now = G_KCa * mkca[i] * Ca[i-1] / 250e-6

        G_KAHP_now = G_KAHP * mkahp[i]

        # Compute Currents
        I_Na[i] = G_Na_now * (E_Na - VS[i-1])
        I_K[i] = G_K_now * (E_K - VS[i-1])
        I_Ca[i] = G_Ca_now * (E_Ca - VD[i-1])
        I_KCa[i] = G_KCa_now * (E_K - VD[i-1])
        I_KAHP[i] = G_KAHP_now * (E_K - VD[i-1])
        I_Link[i] = G_Link * (VD[i-1] - VS[i-1])

        # Update Membrane Potentials
        VS[i] = VS[i-1] + dt * ((I_Na[i] + I_K[i] + I_Link[i] + Iapp[i]) / CmS)
        VD[i] = VD[i-1] + dt * ((I_Ca[i] + I_KCa[i] + I_KAHP[i] - I_Link[i]) / CmD)

        # Update Calcium concentration
        Ca_inf = tau_Ca * convert_Ca * I_Ca[i]
        Ca[i] = Ca_inf - (Ca_inf - Ca[i-1]) * np.exp(-dt / tau_Ca)

    return VS, VD, Ca

# Run Simulation
Iapp = np.zeros(len(t))  # No applied current
VS, VD, Ca = integrate_PR(Iapp, G_Link)

# Detect somatic spikes (crossing -10mV threshold from below)
spike_threshold = -0.01  # -10mV in volts
spikes = np.where((VS[:-1] < spike_threshold) & (VS[1:] >= spike_threshold))[0]

# Plot with detected spikes
plt.figure(figsize=(10, 4))
plt.plot(t, VS * 1000, 'k', label="Soma Membrane Potential")
plt.scatter(t[spikes], VS[spikes] * 1000, color='red', marker='x', label="Detected Spikes")
plt.title("Detected Spikes in Soma")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.show()

# Print spike count
print(f"Number of detected spikes: {len(spikes)}")


G_values = [0, 10e-9, 100e-9]  # 0nS, 10nS, 100nS

plt.figure(figsize=(10, 4))

for G in G_values:
    VS, VD, Ca = integrate_PR(Iapp, G)  # Pass G_Link as an argument
    
    plt.plot(t, VS * 1000, label=f"G_Link = {G*1e9:.0f} nS")

plt.title("Effect of G_Link on Soma Potential")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid()
plt.show()

I_values = [50e-12, 100e-12, 200e-12]  # 50pA, 100pA, 200pA

# Plot Effect of Injecting Current into the Dendrite
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

for i, I in enumerate(I_values):
    Iapp = np.full(len(t), I)  # Apply constant current to dendrite
    VS, VD, Ca = integrate_PR(Iapp, G_Link)  # Simulate
    
    axs[i].plot(t, VD * 1000, label=f"Dendritic Current = {I*1e12:.0f} pA", color="black")
    axs[i].set_ylabel("V$_D$ (mV)")
    axs[i].legend()
    axs[i].grid()

axs[-1].set_xlabel("Time (s)")
fig.suptitle("Effect of Current Injection in Dendrite")
plt.tight_layout()
plt.show()

# Plot Effect of Injecting Current into the Soma
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

for i, I in enumerate(I_values):
    Iapp = np.full(len(t), I)  # Apply constant current to soma
    VS, VD, Ca = integrate_PR(Iapp, G_Link)  # Simulate
    
    axs[i].plot(t, VS * 1000, label=f"Somatic Current = {I*1e12:.0f} pA", color="black")
    axs[i].set_ylabel("V$_S$ (mV)")
    axs[i].legend()
    axs[i].grid()

axs[-1].set_xlabel("Time (s)")
fig.suptitle("Effect of Current Injection in Soma")
plt.tight_layout()
plt.show()

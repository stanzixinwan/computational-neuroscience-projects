import numpy as np
import matplotlib.pyplot as plt

def threshold_linear(I, r_max):
    return np.clip(I, 0, r_max)

def simulate_with_initial(W, Theta, I_app, r_init=None, r_max=100, tau=0.01, T=3.0, dt=0.0001):
    n_units = len(Theta)
    n_steps = int(T / dt)
    r = np.zeros((n_steps, n_units))
    
    if r_init is not None:
        r[0] = np.array(r_init)
    
    for t in range(1, n_steps):
        I = I_app[t] + W.T @ r[t-1] # computes the total input current to each unit
        drdt = -r[t-1] + threshold_linear(I - Theta, r_max)
        r[t] = r[t-1] + dt / tau * drdt
        
    return r

def get_params(q_num):
    if q_num == 1:
        W = np.array([[0.6, 1.0], [-0.2, 0]])
        Theta = np.array([-5, -10])
    elif q_num == 2:
        W = np.array([[1.2, -0.3], [-0.2, 1.1]])
        Theta = np.array([10, 5])
    elif q_num == 3:
        W = np.array([[2.5, 2.0], [-3.0, -2.0]])
        Theta = np.array([-10, 0])
    elif q_num == 4:
        W = np.array([[0.8, -0.2], [-0.4, 0.6]])
        Theta = np.array([-10, -10])
    elif q_num == 5:
        W = np.array([[2.0, 1.0], [-1.5, 0.0]])
        Theta = np.array([0, 20])
    elif q_num == 6:
        W = np.array([[1.5, 0, 1], [0, 2, 1], [-2.5, -3, -1]])
        Theta = np.array([-10, -5, 5])
    elif q_num == 7:
        W = np.array([[2.2, -0.5, 0.9], [-0.7, 2, 1.2], [-1.6, -1.2, 0]])
        Theta = np.array([-18, -15, 0])
    elif q_num == 8:
        W = np.array([[2.05, -0.2, 1.2], [-0.05, 2.1, 0.5], [-1.6, -4, 0]])
        Theta = np.array([-10, -20, 10])
    elif q_num == 9:
        W = np.array([[0.98, -0.015, -0.01], [0, 0.99, -0.02], [-0.02, 0.005, 1.01]])
        Theta = np.array([-2, -1, -1])
    else:
        raise ValueError("Invalid question number")
    return W, Theta

def plot_time_series(time, r, title):
    plt.figure(figsize=(8, 4))
    for i in range(r.shape[1]):
        plt.plot(time, r[:, i], label=f'r{i+1}')
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_plane(r, title):
    if r.shape[1] == 2:
        plt.figure(figsize=(5, 5))
        plt.plot(r[:, 0], r[:, 1])
        plt.xlabel("r1 (Hz)")
        plt.ylabel("r2 (Hz)")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Full loop for all questions
def run_all_questions():
    T = 3.0
    dt = 0.0001
    tau = 0.01
    r_max = 100
    pulse_info = (1.0, 2.0, 20, 0)
    time = np.arange(0, T, dt)

    for q_num in range(1, 10):
        W, Theta = get_params(q_num)
        n_units = len(Theta)
        initial_conditions = [[0] * n_units, [50] * n_units]

        for r_init in initial_conditions:
            I_app = np.zeros((len(time), n_units))
            t_on, t_off, amp, unit = pulse_info
            if unit < n_units:
                I_app[int(t_on/dt):int(t_off/dt), unit] = amp

            r = simulate_with_initial(W, Theta, I_app, r_init=r_init, r_max=r_max, tau=tau, T=T, dt=dt)

            title = f"Q{q_num} | Init {r_init}"
            plot_time_series(time, r, title + " - Time Series")
            if n_units == 2:
                plot_phase_plane(r, title + " - Phase Plane")

run_all_questions()

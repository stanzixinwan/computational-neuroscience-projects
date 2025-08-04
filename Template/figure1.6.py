import numpy as np
import matplotlib.pyplot as plt

# Global Parameters
dt = 0.01  # Time-step
t = np.arange(0, 5, dt)  # Time vector

## First the calculated results
a = -10             # Constant acceleration (gravity)
v = a*t;            # Linearly varying velocity
y = 0.5*a*t*t       # Quadratically varying position

## Next do the simulation as in Table 1.2 but with small time-steps
v_sim = np.zeros(len(t))
y_sim = np.zeros(len(t))
for i in range( 2, len(t) ):
    v_sim[i] = v_sim[i-1] + dt*a
    y_sim[i] = y_sim[i-1] + dt*0.5*(v_sim[i-1]+v_sim[i])

## Set up the plotting parameters and plot the results
plt.rcParams.update({
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 2,
    'axes.labelsize': 10,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


plt.figure(figsize=(10,10))
fig, axes = plt.subplots(2,1)

axes[0].plot(t,v,'r')
axes[0].set_xlabel('Time (sec)')
axes[0].set_ylabel('Velocity (ms$^{-1}$)')
axes[0].plot([0,1,2,3,4],[0,-10,-20,-30,-40],'xk')
axes[0].set_xlim(0, 5)
axes[0].set_ylim(-50, 0)
axes[0].plot(t,v_sim,'k')

axes[1].plot(t,y,'r')
axes[1].set_xlabel('Time (sec)')
axes[1].set_ylabel('Height (m)')
axes[1].plot([0, 1, 2, 3, 4],[0, -5, -20, -45, -80],'xk')
axes[1].set_xlim(0, 5)
axes[1].set_ylim(-130, 0)
axes[1].plot(t,y_sim,'k')

axes[0].annotate('A',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')
axes[1].annotate('B',xy=(-0.25,1.05),xycoords='axes fraction',fontsize=16,fontweight='bold')

plt.tight_layout()

fig.subplots_adjust(hspace=0.75)
plt.show()
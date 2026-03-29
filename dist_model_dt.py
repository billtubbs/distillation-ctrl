import control as con
import numpy as np
import matplotlib.pyplot as plt
from distillation_ctrl.c2d_utils import c2d_with_delay


# -----------------------------------------------------------------------------
# Distillation column discrete-time dynamic model
# (derived from Simulink CT model)
# Author: Michael Grieve (base model author)
# Created by: Bill Tubbs
# Date: March 2026
# Model strategy: discretize continuous SISO blocks with Ts=1 min via ZOH,
# then append pure discrete delay (z^-N) as sample delays.
#
# Inputs (manipulated variables):
#   V = boilup rate (BPH)
#   D = distillate draw (BPH)
# Outputs (controlled variables):
#   OHt  = tower overhead temperature (degF)
#   L    = reflux rate (BPH)
#   BmT  = tower bottom temperature (degF)
# -----------------------------------------------------------------------------

Ts = 1.0  # minutes
s = con.TransferFunction.s

# Continuous SISO plant parts (no dead-time yet)
G_V_OHt_ct = -0.2 * 1.0/(6*s + 1)
G_D_OHt_ct = 0.3 * 1.0/(4*s + 1)

G_V_L_ct = 1.0 * 1.0/(8*s + 1)
G_D_L_ct = -1.0 * 1.0/(4*s + 1)

G_V_BmT_ct = 1.0 * 1.0/(4*s + 1)
G_D_BmT_ct = 1.2 * 1.0/(6*s + 1)

# Discrete-time plant with internal delays
# (use c2d_with_delay from module variant)
# Note: local helper removed to avoid shadowing the imported function

G_V_OHt = c2d_with_delay(G_V_OHt_ct, 4)
G_D_OHt = c2d_with_delay(G_D_OHt_ct, 3)

G_V_L = c2d_with_delay(G_V_L_ct, 5)
G_D_L = c2d_with_delay(G_D_L_ct, 1)

G_V_BmT = c2d_with_delay(G_V_BmT_ct, 2)
G_D_BmT = c2d_with_delay(G_D_BmT_ct, 5)

# Build a true MIMO TF object for analysis (using combine_tf explicitly)
# combine_tf automatically propagates the discrete sampling time from blocks.
T = con.combine_tf(
    [
        [G_V_OHt, G_D_OHt],
        [G_V_L,   G_D_L],
        [G_V_BmT, G_D_BmT]
    ]
)
print('Discrete-time MIMO TF T(z):')
print(T)

# step response grid (ny, nu)
ny, nu = 3, 2

t = np.arange(0, 61, Ts)
fig, axs = plt.subplots(ny, nu, figsize=(8, 5.5), sharex=True)
fig.suptitle('Discrete-time MIMO Step Responses (Ts=1 min)')

for j in range(nu):
    for i in range(ny):
        ytf = T[i, j]
        t_resp, y_resp = con.step_response(ytf, t)
        ax = axs[i, j] if (ny > 1 and nu > 1) else axs[max(i, j)]
        ax.step(t_resp, y_resp, where='post')
        ax.grid(True, alpha=0.3)
        if i == ny - 1:
            ax.set_xlabel('time [min]')
        if j == 0:
            out_names = ['OHt', 'L', 'BmT']
            ax.set_ylabel(out_names[i])
        ax.set_title(f'Out {i+1} vs In {j+1}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

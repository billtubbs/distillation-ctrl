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
# Model strategy: discretize continuous SISO plant blocks with Ts=1 min via ZOH,
# then append pure discrete delay (z^-N) as sample delays.
#
# Inputs (manipulated variables):
#   V = boilup rate (BPH)
#   D = distillate draw (BPH)
# Outputs (controlled variables):
#   OHt  = tower overhead temperature (degF)
#   L    = reflux rate (BPH)
#   BmT  = tower bottom temperature (degF)
#   Vdot = rate of change of V
#   Ddot = rate of change of D
#   V-   = observed passed-through V (observation path)
#   D-   = observed passed-through D (observation path)
# -----------------------------------------------------------------------------

Ts = 1.0  # minutes
s = con.TransferFunction.s

# Continuous SISO plant parts (no dead-time yet)
G_V_OHt_ct = 0.2 * 1.0/(6*s + 1)
G_D_OHt_ct = 0.3 * 1.0/(4*s + 1)

G_V_L_ct   = 1.0 * 1.0/(8*s + 1)
G_D_L_ct   = 1.0 * 1.0/(4*s + 1)

G_V_BmT_ct = 1.0 * 1.0/(4*s + 1)
G_D_BmT_ct = 1.2 * 1.0/(6*s + 1)

G_low_ct = 1.0/(0.001*s + 1)

# Discrete-time plant with internal delays
# (use c2d_with_delay from module variant)
# Note: local helper removed to avoid shadowing the imported function

G_V_OHt = c2d_with_delay(G_V_OHt_ct, 4)
G_D_OHt = c2d_with_delay(G_D_OHt_ct, 3)

G_V_L = c2d_with_delay(G_V_L_ct, 5)
G_D_L = c2d_with_delay(G_D_L_ct, 1)

G_V_BmT = c2d_with_delay(G_V_BmT_ct, 2)
G_D_BmT = c2d_with_delay(G_D_BmT_ct, 5)

G_Vdot = c2d_with_delay(G_low_ct, 1)
G_Ddot = c2d_with_delay(G_low_ct, 1)
G_Vm   = c2d_with_delay(G_low_ct, 1)
G_Dm   = c2d_with_delay(G_low_ct, 1)

# Build nested SISO matrix
T_blocks = [
    [G_V_OHt, G_D_OHt],
    [G_V_L,   G_D_L],
    [G_V_BmT, G_D_BmT],
    [G_Vdot,  0],
    [0,       G_Ddot],
    [G_Vm,    0],
    [0,       G_Dm]
]

print('discrete-time MIMO blocks (7 outputs x 2 inputs):')
for i, row in enumerate(T_blocks):
    for j, entry in enumerate(row):
        print(f'T[{i},{j}] = {entry}')

# Build a true MIMO TF object for analysis (using combine_tf explicitly)
# combine_tf automatically propagates the discrete sampling time from blocks.
T = con.combine_tf(
    [
        [G_V_OHt, G_D_OHt],
        [G_V_L,   G_D_L],
        [G_V_BmT, G_D_BmT],
        [G_Vdot,  0],
        [0,       G_Ddot],
        [G_Vm,    0],
        [0,       G_Dm]
    ]
)
print('Discrete-time TRUE MIMO TF T(z):')
print(T)

# Quick debug print to demonstrate sample delay in a concrete branch
yy_t, yy_y = con.step_response(T[0, 0], np.arange(0, 10))
print('Delayed path T[0,0] step response first 8 samples:', yy_y.flatten()[:8])

# step response grid (ny=7, nu=2), total 60 min
ny, nu = 7, 2

t = np.arange(0, 61, Ts)
fig, axs = plt.subplots(ny, nu, figsize=(12, 10), sharex=True)
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
            out_names = ['OHt', 'L', 'BmT', 'Vdot', 'Ddot', 'V-', 'D-']
            ax.set_ylabel(out_names[i])
        ax.set_title(f'Out {i+1} vs In {j+1}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

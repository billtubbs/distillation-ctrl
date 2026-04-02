import control as con
import numpy as np
import matplotlib.pyplot as plt
from dist_model_lin_ctrl.delay_ct import delay_tf
from dist_model_lin_ctrl.sim_utils import mimo_forced_response


# -----------------------------------------------------------------------------
# Distillation column linear dynamic model (Simulink -> python-control)
# Author: Michael Grieve (model author)
# Provided to Bill Tubbs August 2020
# Source: Simulink screenshot + spreadsheet commentary
#
# Inputs (manipulated variables):
#   V = boilup rate (BPH)
#   D = distillate draw (BPH)
# Outputs (controlled variables):
#   OHt  = tower overhead temperature (degF)
#   L    = reflux rate (BPH)
#   BmT  = tower bottom temperature (degF)
#
# Timing conventions from spreadsheet:
#   time constants are first-order lags in minutes
#   dead times (z-shifts) are in minutes
#
# Equation examples from sheet:
#   OHt = 0.2 * V (deadtime 4, lag 6) + 0.3 * D (deadtime 3, lag 4)
#   (plus similarly for L and BmT paths)
# -----------------------------------------------------------------------------

# Laplace variable
s = con.TransferFunction.s

# core continuous paths
G_V_OHt = -0.2 * delay_tf(4) * 1.0 / (6 * s + 1)
G_D_OHt = 0.3 * delay_tf(3) * 1.0 / (4 * s + 1)

G_V_L = 1.0 * delay_tf(5) * 1.0 / (8 * s + 1)
G_D_L = -1.0 * delay_tf(1) * 1.0 / (4 * s + 1)

G_V_BmT = 1.0 * delay_tf(2) * 1.0 / (4 * s + 1)
G_D_BmT = 1.2 * delay_tf(5) * 1.0 / (6 * s + 1)

# build 2-input, 3-output MIMO transfer function directly as matrix
# Using combine_tf ensures MATLAB-like MIMO assembly
# without manual num/den handling
# output ordering: OHt, L, BmT, Vdot, Ddot, V-, D-
# input ordering: V, D

T_blocks = [[G_V_OHt, G_D_OHt], [G_V_L, G_D_L], [G_V_BmT, G_D_BmT]]

print("System T(s) matrix entries (decomposed):")
for i, row in enumerate(T_blocks):
    for j, entry in enumerate(row):
        print(f"T[{i},{j}] = {entry}")

# Build true MIMO TF object via combine_tf
# (transfers continuous systems directly)
T = con.combine_tf(T_blocks)
print("Continuous-time MIMO TF T(s):")
print(T)

# verify step responses quickly
# ny = 3 outputs, nu = 2 inputs
ny, nu = 3, 2

# Plot multi-input multi-output step responses in a grid

# Simulation time steps
t = np.linspace(0, 60, 601)

# Get full MIMO step response: y shape is (ny, len(t), nu)
# control.step_response for MIMO returns y, x, and u;
# we can step each input separately.
fig, axs = plt.subplots(ny, nu, figsize=(8, 5.5), sharex=True)
fig.suptitle("MIMO Step Responses for Distillation Model")

for j in range(nu):
    # Step only input j (other input zero)
    for i in range(ny):
        # create SISO TF for y_i / u_j
        ytf = T[i, j]  # MIMO TF access uses tuple indices
        t_resp, y_resp = con.step_response(ytf, t)
        ax = axs[i, j] if (ny > 1 and nu > 1) else axs[max(i, j)]
        ax.plot(t_resp, y_resp, lw=1.2)
        ax.grid(True, alpha=0.4)
        if i == ny - 1:
            ax.set_xlabel("time [min]")
        if j == 0:
            out_names = ["OHt", "L", "BmT", "Vdot", "Ddot", "V-", "D-"]
            ax.set_ylabel(out_names[i])
        ax.set_title(f"Output {i + 1} vs Input {j + 1}")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Nominal operating points (M. Grieve estimates)
V_NOM = 135.0   # BPH  (= D_NOM + L_NOM)
D_NOM = 45.0    # BPH
OHT_NOM = 190.0  # degF
L_NOM = 90.0    # BPH
BMT_NOM = 420.0  # degF

# Step sizes: half of estimated maximum variation (±)
# V varies with D (±15) and L (±40), so ±55 BPH total; half = 27.5
# D varies ±15 BPH; half = 7.5
V_STEP = 27.5  # BPH
D_STEP = 7.5   # BPH

# Additional scenario: step in V at t=5, step in D at t=30
t = np.arange(0, 61, 1)
u_V = np.zeros_like(t, dtype=float)
u_D = np.zeros_like(t, dtype=float)
u_V[t >= 5] = V_STEP
u_D[t >= 30] = D_STEP

# Simulate the step profile using fallback MIMO routine.
# This does not require the Slycot dependency.
U = np.vstack([u_V, u_D])
yout = mimo_forced_response(T, t, U)

# Convert deviations to absolute engineering values
v_abs = V_NOM + u_V
d_abs = D_NOM + u_D
oht_abs = OHT_NOM + yout[0]
l_abs = L_NOM + yout[1]
bmt_abs = BMT_NOM + yout[2]

fig, axs = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
fig.suptitle(
    f"CT scenario: V +{V_STEP} BPH @t=5, D +{D_STEP} BPH @t=30 (60 min)"
)

axs[0].plot(t, oht_abs, drawstyle="steps-post")
axs[0].set_ylabel("OHt [degF]")
axs[0].grid(True, alpha=0.4)

axs[1].plot(t, l_abs, drawstyle="steps-post")
axs[1].set_ylabel("L [BPH]")
axs[1].grid(True, alpha=0.4)

axs[2].plot(t, bmt_abs, drawstyle="steps-post")
axs[2].set_ylabel("BmT [degF]")
axs[2].grid(True, alpha=0.4)

axs[3].plot(t, v_abs, drawstyle="steps-post")
axs[3].set_ylabel("V [BPH]")
axs[3].grid(True, alpha=0.4)

axs[4].plot(t, d_abs, drawstyle="steps-post")
axs[4].set_ylabel("D [BPH]")
axs[4].set_xlabel("time [min]")
axs[4].grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

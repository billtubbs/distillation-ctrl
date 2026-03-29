import control as con
import numpy as np

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
#   Vdot = rate of change of V
#   Ddot = rate of change of D
#   V-   = observed passed-through V (observation path)
#   D-   = observed passed-through D (observation path)
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

# approximate delay settings (assume 1 minute sample interval because
# deadtime entries from the spreadsheet are in minutes)
Ts = 1.0
def delay_tf(N, pade_order=3):
    if N <= 0:
        return con.TransferFunction([1], [1])
    num, den = con.pade(N * Ts, pade_order)
    return con.TransferFunction(num, den)

# core continuous paths
G_V_OHt = 0.2 * delay_tf(4) * 1.0/(6*s + 1)
G_D_OHt = 0.3 * delay_tf(3) * 1.0/(4*s + 1)

G_V_L   = 1.0 * delay_tf(5) * 1.0/(8*s + 1)
G_D_L   = 1.0 * delay_tf(1) * 1.0/(4*s + 1)

G_V_BmT = 1.0 * delay_tf(2) * 1.0/(4*s + 1)
G_D_BmT = 1.2 * delay_tf(5) * 1.0/(6*s + 1)

G_low = 1.0/(0.001*s + 1)  # same first-order block reused for rate / observation paths

# Vdot and Ddot are rate-of-change monitoring signals (from manipulations)
# V- and D- are direct pass-through observation elements used for diagnostics.
# In the Simulink screenshot, these are implemented as 1-step delay + small lag.
G_Vdot = delay_tf(1) * G_low
G_Ddot = delay_tf(1) * G_low
G_Vm   = delay_tf(1) * G_low
G_Dm   = delay_tf(1) * G_low

# build 2-input, 7-output MIMO transfer function directly as matrix
# (avoid con.append() because dt mismatches with mixed 0/None dt in some versions)
# output ordering: OHt, L, BmT, Vdot, Ddot, V-, D-
# input ordering: V, D

# easiest: keep as a nested Python list of SISO TF block entries
# for plotting and diagnostics. Build a true MIMO TF object from numerators/denominators.
T_blocks = [
    [G_V_OHt, G_D_OHt],
    [G_V_L,   G_D_L],
    [G_V_BmT, G_D_BmT],
    [G_Vdot,  0],
    [0,       G_Ddot],
    [G_Vm,    0],
    [0,       G_Dm]
]

print("System T(s) matrix entries (decomposed):")
for i, row in enumerate(T_blocks):
    for j, entry in enumerate(row):
        print(f"T[{i},{j}] = {entry}")

# Build true MIMO TF object (from python-control docs style)
def tf_num_den(entry):
    if isinstance(entry, (int, float)):
        return [entry], [1]
    # entry is a TransferFunction object
    # each g.num/g.den is nested like [[num_coefs]] for SISO
    return entry.num[0][0], entry.den[0][0]

num = []
den = []
for row in T_blocks:
    num_row = []
    den_row = []
    for entry in row:
        n, d = tf_num_den(entry)
        num_row.append(n)
        den_row.append(d)
    num.append(num_row)
    den.append(den_row)

T = con.tf(num, den)
print("True MIMO TF T(s):")
print(T)

# verify step responses quickly
# ny = 7 outputs, nu = 2 inputs
ny, nu = 7, 2

# Plot multi-input multi-output step responses in a grid
import matplotlib.pyplot as plt

# Simulation time steps
t = np.linspace(0, 60, 601)

# Get full MIMO step response: y shape is (ny, len(t), nu)
# control.step_response for MIMO returns y, x, and u; we can step each input separately.
fig, axs = plt.subplots(ny, nu, figsize=(12, 10), sharex=True)
fig.suptitle('MIMO Step Responses for Distillation Model')

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
            ax.set_xlabel('time [min]')
        if j == 0:
            out_names = ['OHt', 'L', 'BmT', 'Vdot', 'Ddot', 'V-', 'D-']
            ax.set_ylabel(out_names[i])
        ax.set_title(f'Output {i+1} vs Input {j+1}')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
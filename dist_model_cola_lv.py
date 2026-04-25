"""
dist_model_cola_lv.py – Step responses for the LV closed-loop
Skogestad–Morari Column A distillation model with built-in P-controllers.

Model: Nonlinear Column A with built-in P-controllers for level
       stabilisation (build_cola_lv_ct_model from cola_lv_model.py).
       D and B are computed internally from holdup states.

Inputs (MVs, nu=5):
    u[0]  LT  : reflux flow          [kmol/min]  nominal 2.70629
    u[1]  VB  : boilup flow          [kmol/min]  nominal 3.20629
    u[2]  F   : feed flow rate       [kmol/min]  nominal 1.0
    u[3]  zF  : feed composition     [mol frac]  nominal 0.5
    u[4]  qF  : feed liquid fraction             nominal 1.0

Controlled variables (CVs) monitored (key product compositions):
    x[0]  xB  : bottoms composition  [mol frac]  SS ~0.01
    x[40] xD  : distillate compos.   [mol frac]  SS ~0.99

For each non-zero MV step, one figure is produced with:
    upper subplots (one per CV): CV deviation from steady state
    bottom subplot: MV step

Additional scenario: step in each main MV (LT then VB) once during
a single 200-min simulation, with all CVs overlaid.
"""

import os
import sys
from pathlib import Path

import casadi as cas  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

from dist_model_cola_cas.cola_model import (  # noqa: E402
    F0_DEFAULT,
    L0_DEFAULT,
    NT,
    QF0_DEFAULT,
    V0_DEFAULT,
)
from dist_model_cola_cas.cola_lv_model import (  # noqa: E402
    X_SS,
    build_cola_lv_sim_function,
    make_nominal_sim_param_values,
)

# ── Configuration ─────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Nominal inputs [LT, VB, F, zF, qF] ───────────────────────────────────
ZF_NOM = 0.5
U_NOM = np.array([L0_DEFAULT, V0_DEFAULT, F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

# ── CV selection ──────────────────────────────────────────────────────────
CV_INDICES = [0, 40]  # x[0]=xB (reboiler), x[40]=xD (condenser)
CV_NAMES = ["xB", "xD"]
CV_LABELS = ["xB [mol frac]", "xD [mol frac]"]

# ── MV definitions and step sizes ─────────────────────────────────────────
MV_NAMES = ["LT", "VB", "F", "zF", "qF"]
MV_UNITS = ["kmol/min", "kmol/min", "kmol/min", "mol frac", "-"]
MV_STEPS = [0.1, 0.1, 0.1, 0.05, 0.1]

# ── Simulation parameters ─────────────────────────────────────────────────
DT = 1.0  # integration step [min]
T_HORIZON = 300  # step-response simulation length [min]
NT_SIM = T_HORIZON  # number of time intervals (dt=1 min)

# Maximum absolute deviation below which a response is considered zero
NONZERO_THR = 1e-5

# ── Build simulation function (compiles CasADi code — takes a moment) ────
print("Building CasADi simulation function …")
sim_func, model, _ = build_cola_lv_sim_function(dt=DT, nT=NT_SIM)
param_vals = make_nominal_sim_param_values()
print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")
print(f"  Inputs: {model.input_names}")

t_eval = np.linspace(0.0, NT_SIM * DT, NT_SIM + 1)


def run_sim(u_seq: np.ndarray) -> np.ndarray:
    """Run NT_SIM-step simulation from X_SS.

    Parameters
    ----------
    u_seq : shape (NT_SIM, 5)
        Input sequence.  u_seq[k] is applied during [k*DT, (k+1)*DT].

    Returns
    -------
    x_traj : shape (NT_SIM+1, 82)
        State trajectory (x_traj[0] = X_SS).
    """
    x_traj, _ = sim_func(
        cas.DM(t_eval),
        cas.DM(u_seq),
        cas.DM(X_SS),
        *param_vals.values(),
    )
    return np.array(x_traj)


# ── Step responses ────────────────────────────────────────────────────────
print("\nComputing step responses …")

n_cv = len(CV_INDICES)

for j, (mv_name, mv_unit, step_size) in enumerate(
    zip(MV_NAMES, MV_UNITS, MV_STEPS)
):
    # Apply step in MV j from t=20 min onwards
    u_step = np.tile(U_NOM, (NT_SIM, 1))
    u_step[20:, j] += step_size

    x_step = run_sim(u_step)  # shape (NT_SIM+1, 82)

    deltas = [x_step[:, cv_idx] - X_SS[cv_idx] for cv_idx in CV_INDICES]
    max_devs = [np.abs(d).max() for d in deltas]

    if all(m < NONZERO_THR for m in max_devs):
        print(f"  Skip {mv_name}: all CVs near zero")
        continue

    for cv_name, delta in zip(CV_NAMES, deltas):
        print(
            f"  {mv_name} → {cv_name}: "
            f"max|Δ| = {np.abs(delta).max():.4f}, "
            f"SS Δ = {delta[-1]:.4f}"
        )

    # One figure per MV: CV subplots on top, MV subplot on bottom
    fig, axs = plt.subplots(
        n_cv + 1, 1, figsize=(7, 1 + 1.5 * (n_cv + 1)), sharex=True
    )
    fig.suptitle(f"Step response: Δ{mv_name} = {step_size:+g} {mv_unit}")

    for i, (delta, cv_label) in enumerate(zip(deltas, CV_LABELS)):
        axs[i].plot(t_eval, delta, color="tab:blue")
        axs[i].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
        axs[i].set_ylabel(f"Δ{cv_label}")
        axs[i].grid(True, alpha=0.3)

    # Bottom subplot: MV step (deviation from nominal, applied at t=20)
    mv_dev = np.zeros(NT_SIM + 1)
    mv_dev[20:] = step_size
    axs[-1].step(t_eval, mv_dev, where="post", color="tab:orange")
    axs[-1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    axs[-1].set_ylabel(f"Δ{mv_name} [{mv_unit}]")
    axs[-1].set_xlabel("Time [min]")
    axs[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"dist_model_cola_lv_step_{mv_name}.png", dpi=300)
    plt.show()

# ── Additional scenario ───────────────────────────────────────────────────
# Step in each main MV (LT, VB) once during the same 200-min simulation.
print("\nRunning additional scenario (ΔLT at t=30, ΔVB at t=120) …")

LT_STEP_TIME = 20  # [min]
VB_STEP_TIME = 170  # [min]
LT_STEP = 0.1  # [kmol/min]
VB_STEP = 0.1  # [kmol/min]

u_scen = np.tile(U_NOM, (NT_SIM, 1))
u_scen[LT_STEP_TIME:, 0] += LT_STEP  # LT step at t = LT_STEP_TIME
u_scen[VB_STEP_TIME:, 1] += VB_STEP  # VB step at t = VB_STEP_TIME

x_scen = run_sim(u_scen)  # shape (NT_SIM+1, 82)

# Time axis for inputs (u_scen[k] applied during [t_eval[k], t_eval[k+1]])
t_u = t_eval[:-1]

fig, axs = plt.subplots(
    n_cv + 1, 1, figsize=(8, 1 + 1.5 * (n_cv + 1)), sharex=True
)
fig.suptitle(
    f"LV Column A scenario: "
    f"ΔLT={LT_STEP:+.2f} kmol/min @t={LT_STEP_TIME} min, "
    f"ΔVB={VB_STEP:+.2f} kmol/min @t={VB_STEP_TIME} min"
)

# Upper subplots: one per CV (absolute composition, not deviation)
for i, (cv_idx, cv_name, cv_label) in enumerate(
    zip(CV_INDICES, CV_NAMES, CV_LABELS)
):
    axs[i].plot(t_eval, x_scen[:, cv_idx], color="tab:blue", label=cv_name)
    axs[i].axhline(
        X_SS[cv_idx], color="k", linewidth=0.5, linestyle="--", label="SS"
    )
    axs[i].set_ylabel(cv_label)
    axs[i].legend(loc="best", fontsize=8)
    axs[i].grid(True, alpha=0.3)

# Bottom subplot: MVs (LT and VB)
axs[-1].step(t_u, u_scen[:, 0], where="post", label="LT")
axs[-1].step(t_u, u_scen[:, 1], where="post", label="VB")
axs[-1].set_ylabel("Flow [kmol/min]")
axs[-1].set_xlabel("Time [min]")
axs[-1].legend(loc="best")
axs[-1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "dist_model_cola_lv_scenario.png", dpi=300)
plt.show()

# ── Column profile plots from scenario simulation ─────────────────────────
print("\nPlotting column profile plots from scenario …")

x_comp_scen = x_scen[:, :NT]  # light-component mole fractions (NT_SIM+1, 41)
M_scen = x_scen[:, NT:]  # molar holdups in kmol          (NT_SIM+1, 41)

PROFILE_WIDTH = 8.0  # figure width [in]
TRAY_HEIGHT = 0.22  # height per tray subplot [in]
SCENARIO_TITLE = (
    f"ΔLT={LT_STEP:+.2f} kmol/min @t={LT_STEP_TIME} min, "
    f"ΔVB={VB_STEP:+.2f} kmol/min @t={VB_STEP_TIME} min"
)


def _make_profile_axes(title: str):
    """Return (fig, axs) for a compact NT-row profile figure."""
    fig_h = TRAY_HEIGHT * NT + 0.7
    fig, axs = plt.subplots(NT, 1, figsize=(PROFILE_WIDTH, fig_h), sharex=True)
    fig.subplots_adjust(
        hspace=0,
        top=1 - 0.45 / fig_h,
        bottom=0.55 / fig_h,
        left=0.04,
        right=0.98,
    )
    fig.suptitle(title, fontsize=8)
    return fig, axs


def _style_profile_axes(axs, ylim):
    """Remove inner decorations; keep black spines; x-axis on bottom only."""
    for ax in axs:
        ax.set_ylim(ylim)
        ax.set_yticks([])
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
    axs[-1].tick_params(axis="x", bottom=True, labelbottom=True)
    axs[-1].set_xlabel("Time [min]", fontsize=8)


# ── Composition profile ───────────────────────────────────────────────────
fig_comp, axs_comp = _make_profile_axes(
    f"Tray compositions (light=blue, heavy=orange) – {SCENARIO_TITLE}"
)

for plot_row in range(NT):
    tray_idx = (
        NT - 1 - plot_row
    )  # tray 40 (condenser) at top, 0 (reboiler) at bottom
    ax = axs_comp[plot_row]
    x_i = x_comp_scen[:, tray_idx]
    ax.fill_between(t_eval, 0, 1 - x_i, color="tab:orange", linewidth=0)
    ax.fill_between(t_eval, 1 - x_i, 1.0, color="tab:blue", linewidth=0)

_style_profile_axes(axs_comp, ylim=(0.0, 1.0))
plt.savefig(
    PLOT_DIR / "dist_model_cola_lv_scenario_comp_profiles.png", dpi=150
)
plt.show()

# ── Holdup profile ────────────────────────────────────────────────────────
fig_hold, axs_hold = _make_profile_axes(
    f"Tray holdups [kmol] – {SCENARIO_TITLE}"
)

M_lo = M_scen.min()
M_hi = M_scen.max()
pad = max(0.005 * (M_hi - M_lo), 1e-4)

for plot_row in range(NT):
    tray_idx = NT - 1 - plot_row
    ax = axs_hold[plot_row]
    ax.fill_between(
        t_eval, M_lo - pad, M_scen[:, tray_idx], color="tab:green", linewidth=0
    )

_style_profile_axes(axs_hold, ylim=(M_lo - pad, M_hi + pad))
plt.savefig(
    PLOT_DIR / "dist_model_cola_lv_scenario_holdup_profiles.png", dpi=150
)
plt.show()

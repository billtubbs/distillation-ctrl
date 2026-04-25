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
import pandas as pd

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
from dist_model_cola_cas.var_info import var_info  # noqa: E402
from plot_utils import make_input_output_tsplot  # noqa: E402

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

# State names for the monitored CVs (e.g. "x0"=xB, "x40"=xD)
CV_STATE_NAMES = [model.state_names[i] for i in CV_INDICES]

# ── Extend var_info with per-stage state variables ────────────────────────
# Stage compositions x0..x40 (x0 = reboiler/bottoms, x40 = condenser/distillate)
_COMP_NAMES = {0: "Bottoms composition", NT - 1: "Distillate composition"}
for _i in range(NT):
    var_info[f"x{_i}"] = {
        "name": _COMP_NAMES.get(_i, f"Stage {_i} composition"),
        "symbol": rf"$x_{{{_i}}}$",
        "units": "mol/mol",
    }

# Stage holdups M0..M40 (M0 = reboiler, M40 = condenser)
_HOLDUP_NAMES = {0: "Reboiler holdup", NT - 1: "Condenser holdup"}
for _i in range(NT):
    var_info[f"M{_i}"] = {
        "name": _HOLDUP_NAMES.get(_i, f"Stage {_i} holdup"),
        "symbol": rf"$M_{{{_i}}}$",
        "units": "kmol",
    }

t_eval = pd.Series(np.linspace(0.0, NT_SIM * DT, NT_SIM + 1), name="Time [min]")


def run_sim(t: pd.Series, U: pd.DataFrame) -> pd.DataFrame:
    """Run simulation from X_SS and return results as a tidy DataFrame.

    Parameters
    ----------
    t : pd.Series
        Time points, length NT_SIM+1.  The first entry is t=0 (initial
        condition); subsequent entries are the output sample times.
    U : pd.DataFrame
        Input sequence, shape (NT_SIM, nu).  Columns must be the model
        input names.  U.iloc[k] is applied during [t[k], t[k+1]].

    Returns
    -------
    pd.DataFrame
        MultiIndex-column DataFrame indexed by t.  Level-0 column names are
        "Inputs" (forward-filled from U) and "Outputs" (all model states).
    """
    x_traj, _ = sim_func(
        cas.DM(t.values),
        cas.DM(U.values),
        cas.DM(X_SS),
        *param_vals.values(),
    )
    x_arr = np.array(x_traj)  # (NT_SIM+1, 82)

    # Forward-fill inputs: repeat last row so inputs align with all t points.
    # Convention: U.iloc[k] is the input applied from t[k] onwards.
    U_full = pd.concat([U, U.iloc[[-1]]], ignore_index=True).values

    cols = pd.MultiIndex.from_tuples(
        [("Inputs", n) for n in model.input_names]
        + [("Outputs", n) for n in model.state_names]
    )
    return pd.DataFrame(
        np.hstack([U_full, x_arr]),
        index=t.values,
        columns=cols,
    )


# ── Step responses ────────────────────────────────────────────────────────
print("\nComputing step responses …")

# Reference values for deviation plots
output_refs = {sn: X_SS[ci] for sn, ci in zip(CV_STATE_NAMES, CV_INDICES)}

for j, (mv_name, mv_unit, step_size) in enumerate(
    zip(MV_NAMES, MV_UNITS, MV_STEPS)
):
    # Apply step in MV j from t=20 min onwards
    U_arr = np.tile(U_NOM, (NT_SIM, 1))
    U_arr[20:, j] += step_size
    U_df = pd.DataFrame(U_arr, columns=model.input_names)

    sim_results = run_sim(t_eval, U_df)

    max_devs = [
        abs(sim_results["Outputs", sn] - X_SS[ci]).max()
        for sn, ci in zip(CV_STATE_NAMES, CV_INDICES)
    ]
    if all(m < NONZERO_THR for m in max_devs):
        print(f"  Skip {mv_name}: all CVs near zero")
        continue

    for cv_name, sn, ci in zip(CV_NAMES, CV_STATE_NAMES, CV_INDICES):
        delta = sim_results["Outputs", sn] - X_SS[ci]
        print(
            f"  {mv_name} → {cv_name}: "
            f"max|Δ| = {abs(delta).max():.4f}, "
            f"SS Δ = {delta.iloc[-1]:.4f}"
        )

    fig, axs = make_input_output_tsplot(
        sim_results,
        output_names=CV_STATE_NAMES,
        input_names=[mv_name],
        output_refs=output_refs,
        input_refs={mv_name: U_NOM[j]},
        deviation=True,
        var_info=var_info,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"dist_model_cola_lv_step_{mv_name}.png", dpi=300)
    plt.show()

# ── Additional scenario ───────────────────────────────────────────────────
print("\nRunning additional scenario (ΔLT at t=30, ΔVB at t=120) …")

LT_STEP_TIME = 20  # [min]
VB_STEP_TIME = 170  # [min]
LT_STEP = 0.1  # [kmol/min]
VB_STEP = 0.1  # [kmol/min]

U_arr = np.tile(U_NOM, (NT_SIM, 1))
U_arr[LT_STEP_TIME:, 0] += LT_STEP  # LT step at t = LT_STEP_TIME
U_arr[VB_STEP_TIME:, 1] += VB_STEP  # VB step at t = VB_STEP_TIME
U_scen = pd.DataFrame(U_arr, columns=model.input_names)

sim_results_scen = run_sim(t_eval, U_scen)

SCENARIO_TITLE = (
    f"LV Column A scenario: "
    f"ΔLT={LT_STEP:+.2f} kmol/min @t={LT_STEP_TIME} min, "
    f"ΔVB={VB_STEP:+.2f} kmol/min @t={VB_STEP_TIME} min"
)

fig, axs = make_input_output_tsplot(
    sim_results_scen,
    output_names=CV_STATE_NAMES,
    input_names=["LT", "VB"],
    output_refs=output_refs,
    deviation=False,
    output_line_labels=CV_NAMES,
    var_info=var_info,
    figsize=(8, 1 + 1.5 * (len(CV_STATE_NAMES) + 1)),
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "dist_model_cola_lv_scenario.png", dpi=300)
plt.show()

# ── Column profile plots from scenario simulation ─────────────────────────
print("\nPlotting column profile plots from scenario …")

x_scen = sim_results_scen["Outputs"].values  # (NT_SIM+1, 82)
x_comp_scen = x_scen[:, :NT]  # light-component mole fractions (NT_SIM+1, 41)
M_scen = x_scen[:, NT:]  # molar holdups in kmol          (NT_SIM+1, 41)

t_arr = t_eval.values

PROFILE_WIDTH = 8.0  # figure width [in]
TRAY_HEIGHT = 0.22  # height per tray subplot [in]
PROFILE_TITLE = (
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
    f"Tray compositions (light=blue, heavy=orange) – {PROFILE_TITLE}"
)

for plot_row in range(NT):
    tray_idx = (
        NT - 1 - plot_row
    )  # tray 40 (condenser) at top, 0 (reboiler) at bottom
    ax = axs_comp[plot_row]
    x_i = x_comp_scen[:, tray_idx]
    ax.fill_between(t_arr, 0, 1 - x_i, color="tab:orange", linewidth=0)
    ax.fill_between(t_arr, 1 - x_i, 1.0, color="tab:blue", linewidth=0)

_style_profile_axes(axs_comp, ylim=(0.0, 1.0))
plt.savefig(
    PLOT_DIR / "dist_model_cola_lv_scenario_comp_profiles.png", dpi=150
)
plt.show()

# ── Holdup profile ────────────────────────────────────────────────────────
fig_hold, axs_hold = _make_profile_axes(
    f"Tray holdups [kmol] – {PROFILE_TITLE}"
)

M_lo = M_scen.min()
M_hi = M_scen.max()
pad = max(0.005 * (M_hi - M_lo), 1e-4)

for plot_row in range(NT):
    tray_idx = NT - 1 - plot_row
    ax = axs_hold[plot_row]
    ax.fill_between(
        t_arr, M_lo - pad, M_scen[:, tray_idx], color="tab:green", linewidth=0
    )

_style_profile_axes(axs_hold, ylim=(M_lo - pad, M_hi + pad))
plt.savefig(
    PLOT_DIR / "dist_model_cola_lv_scenario_holdup_profiles.png", dpi=150
)
plt.show()

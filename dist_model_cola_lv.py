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

Outputs (ny=82):
    x[0..40]  : liquid mole fraction of light component A  [mol/mol]
                (x[0] = reboiler/bottoms, x[40] = condenser/distillate)
    M[0..40]  : molar liquid holdup on each stage           [kmol]
                (M[0] = reboiler, M[40] = condenser)

Monitored outputs (CVs):
    x[0]   xB  : bottoms composition     [mol/mol]  SS ~0.01
    x[40]  xD  : distillate composition  [mol/mol]  SS ~0.99

For each non-zero MV step, one figure is produced with:
    upper subplots (one per CV): CV deviation from steady state
    bottom subplot: MV step

Additional scenario: step in each main MV (LT then VB) once during
a single 200-min simulation, with all CVs overlaid.
"""

from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dist_model_cola_cas.cola_model import (
    F0_DEFAULT,
    L0_DEFAULT,
    NT,
    QF0_DEFAULT,
    V0_DEFAULT,
)
from dist_model_cola_cas.cola_lv_model import (
    BS_DEFAULT,
    DS_DEFAULT,
    X_SS,
    build_cola_lv_sim_function,
    make_nominal_lv_param_values,
)
from dist_model_cola_cas.var_info import var_info
from plot_utils import make_input_output_tsplot
from sim_utils import run_simulation

# ── Configuration ─────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Nominal inputs [LT, VB, F, zF, qF] ───────────────────────────────────
ZF_NOM = 0.5
U_NOM = np.array([L0_DEFAULT, V0_DEFAULT, F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

# ── CV selection ──────────────────────────────────────────────────────────
# Output names as they appear in sim_results["Outputs"]
CV_OUTPUT_NAMES = ["x0", "x40", "D", "B"]
CV_NAMES = ["xB", "xD", "D", "B"]  # short labels for print/legend

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
sim_func, model = build_cola_lv_sim_function(dt=DT, nT=NT_SIM)
param_vals = make_nominal_lv_param_values()
print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")
print(f"  Inputs: {model.input_names}")

# Steady-state reference values for each CV output
CV_OUTPUT_REFS = {
    "x0": X_SS[0],
    "x40": X_SS[40],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

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

t_eval = pd.Series(
    np.linspace(0.0, NT_SIM * DT, NT_SIM + 1), name="Time [min]"
)


# ── Step responses ────────────────────────────────────────────────────────
print("\nComputing step responses …")

for j, (mv_name, mv_unit, step_size) in enumerate(
    zip(MV_NAMES, MV_UNITS, MV_STEPS)
):
    # Apply step in MV j from t=20 min onwards
    U_arr = np.tile(U_NOM, (NT_SIM, 1))
    U_arr[20:, j] += step_size
    U_df = pd.DataFrame(U_arr, columns=model.input_names)

    sim_results = run_simulation(t_eval, U_df, model, param_vals, X_SS, sim_func=sim_func)

    max_devs = [
        abs(sim_results["Outputs", sn] - CV_OUTPUT_REFS[sn]).max()
        for sn in CV_OUTPUT_NAMES
    ]
    if all(m < NONZERO_THR for m in max_devs):
        print(f"  Skip {mv_name}: all CVs near zero")
        continue

    for cv_name, sn in zip(CV_NAMES, CV_OUTPUT_NAMES):
        delta = sim_results["Outputs", sn] - CV_OUTPUT_REFS[sn]
        print(
            f"  {mv_name} → {cv_name}: "
            f"max|Δ| = {abs(delta).max():.4f}, "
            f"SS Δ = {delta.iloc[-1]:.4f}"
        )

    fig, axs = make_input_output_tsplot(
        sim_results,
        output_names=CV_OUTPUT_NAMES,
        input_names=[mv_name],
        output_refs=CV_OUTPUT_REFS,
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

sim_results_scen = run_simulation(t_eval, U_scen, model, param_vals, X_SS, sim_func=sim_func)

SCENARIO_TITLE = (
    f"LV Column A scenario: "
    f"ΔLT={LT_STEP:+.2f} kmol/min @t={LT_STEP_TIME} min, "
    f"ΔVB={VB_STEP:+.2f} kmol/min @t={VB_STEP_TIME} min"
)

fig, axs = make_input_output_tsplot(
    sim_results_scen,
    output_names=CV_OUTPUT_NAMES,
    input_names=["LT", "VB"],
    output_refs=CV_OUTPUT_REFS,
    deviation=False,
    output_line_labels=CV_NAMES,
    var_info=var_info,
    figsize=(8, 1 + 1.5 * (len(CV_OUTPUT_NAMES) + 1)),
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "dist_model_cola_lv_scenario.png", dpi=300)
plt.show()

# ── Column profile plots from scenario simulation ─────────────────────────
print("\nPlotting column profile plots from scenario …")

x_scen = sim_results_scen["States"].values  # (NT_SIM+1, 82)
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

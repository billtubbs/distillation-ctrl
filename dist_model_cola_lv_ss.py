"""
dist_model_cola_lv_ss.py – Steady-state I/O characteristics of the
LV closed-loop Column A distillation model.

For each MV/DV (LT, VB, F, zF, qF), the steady-state values of the four
key CVs (xB, xD, D, B) are computed over a range of input values using
Newton rootfinding on the model ODE RHS.  Starting from the known nominal
steady state, each successive sweep point is warm-started from the previous
solution, acting as a simple continuation method.

Produces a 4 × 5 grid of static gain curves, one subplot per (CV, MV) pair.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dist_model_cola_cas.cola_model import (
    F0_DEFAULT,
    L0_DEFAULT,
    QF0_DEFAULT,
    V0_DEFAULT,
)
from dist_model_cola_cas.cola_lv_model import (
    BS_DEFAULT,
    DS_DEFAULT,
    X_SS,
    build_cola_lv_ct_model,
    make_nominal_lv_param_values,
)
from dist_model_cola_cas.var_info import var_info
from sim_utils import make_steady_state_solver

# ── Configuration ─────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ZF_NOM = 0.5
U_NOM = np.array([L0_DEFAULT, V0_DEFAULT, F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

MV_NAMES = ["LT", "VB", "F", "zF", "qF"]
CV_OUTPUT_NAMES = ["x0", "x40", "D", "B"]

# Display labels and units for each CV (keyed by output name)
CV_INFO = {
    "x0": {"label": "xB", "units": "mol/mol"},
    "x40": {"label": "xD", "units": "mol/mol"},
    "D": {"label": "D", "units": "kmol/min"},
    "B": {"label": "B", "units": "kmol/min"},
}

# Nominal steady-state CV values (for reference lines on plots)
CV_NOM = {
    "x0": X_SS[0],
    "x40": X_SS[40],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

# Sweep ranges: ±step_size around nominal, matching MV_STEPS in dist_model_cola_lv.py
MV_STEPS = [0.05, 0.05, 0.05, 0.025, 0.05]
MV_SWEEPS = {
    mv: np.linspace(U_NOM[j] - step, U_NOM[j] + step, 31)
    for j, (mv, step) in enumerate(zip(MV_NAMES, MV_STEPS))
}

# ── Build model and compile steady-state solver ───────────────────────────
print("Building model and compiling steady-state solver …")
model = build_cola_lv_ct_model()
param_vals = make_nominal_lv_param_values()
ss_solver = make_steady_state_solver(model)
print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")

output_idx = {name: i for i, name in enumerate(model.output_names)}

# Sanity check: solver should recover the known nominal steady state.
x_ss_nom, y_ss_nom = ss_solver(X_SS, U_NOM, param_vals)
print(
    f"  Nominal SS: xB={y_ss_nom[output_idx['x0']]:.6f}, "
    f"xD={y_ss_nom[output_idx['x40']]:.6f}, "
    f"D={y_ss_nom[output_idx['D']]:.6f}, "
    f"B={y_ss_nom[output_idx['B']]:.6f}"
)

# ── Steady-state sweeps ───────────────────────────────────────────────────
print("\nComputing steady-state sweeps …")

# results[mv_name][cv_name] → np.ndarray of CV values over the sweep
results = {}

for j, mv_name in enumerate(MV_NAMES):
    mv_vals = MV_SWEEPS[mv_name]
    n_pts = len(mv_vals)
    cv_vals = {cv: np.full(n_pts, np.nan) for cv in CV_OUTPUT_NAMES}

    # Split at the nominal MV value; march outward in both directions so
    # each point warm-starts from a nearby converged solution.
    nom_idx = int(np.searchsorted(mv_vals, U_NOM[j]))

    # Upward sweep: nominal → upper bound
    x0 = X_SS.copy()
    for k in range(nom_idx, n_pts):
        u = U_NOM.copy()
        u[j] = mv_vals[k]
        x_ss, y_ss = ss_solver(x0, u, param_vals)
        for cv in CV_OUTPUT_NAMES:
            cv_vals[cv][k] = y_ss[output_idx[cv]]
        x0 = x_ss

    # Downward sweep: nominal → lower bound
    x0 = X_SS.copy()
    for k in range(nom_idx - 1, -1, -1):
        u = U_NOM.copy()
        u[j] = mv_vals[k]
        x_ss, y_ss = ss_solver(x0, u, param_vals)
        for cv in CV_OUTPUT_NAMES:
            cv_vals[cv][k] = y_ss[output_idx[cv]]
        x0 = x_ss

    results[mv_name] = cv_vals
    print(f"  {mv_name}: {n_pts} points")

# ── Plot: 4 CVs × 5 MVs grid ─────────────────────────────────────────────
print("\nPlotting …")

n_cvs = len(CV_OUTPUT_NAMES)
n_mvs = len(MV_NAMES)

fig, axs = plt.subplots(
    n_cvs,
    n_mvs,
    figsize=(3.0 * n_mvs, 2.5 * n_cvs),
    sharex="col",
    sharey="row",
    constrained_layout=True,
)

for col, mv_name in enumerate(MV_NAMES):
    mv_info = var_info.get(mv_name, {})
    mv_units = mv_info.get("units", "")
    if mv_units == "dimensionless":
        mv_units = "—"
    mv_vals = MV_SWEEPS[mv_name]
    nom_mv = U_NOM[col]

    for row, cv_name in enumerate(CV_OUTPUT_NAMES):
        ax = axs[row, col]
        cv_info = CV_INFO[cv_name]

        ax.plot(mv_vals, results[mv_name][cv_name], color="C0")
        ax.axvline(nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(
            CV_NOM[cv_name],
            color="k",
            linewidth=0.6,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(nom_mv, CV_NOM[cv_name], "ko", markersize=4)
        ax.grid(True, alpha=0.3)

        if row == 0:
            ax.set_title(
                f"{mv_info.get('name', mv_name)}",
                fontsize=8,
            )
        if col == 0:
            ax.set_ylabel(
                f"{cv_info['label']} [{cv_info['units']}]",
                fontsize=8,
            )
        if row == n_cvs - 1:
            mv_symbol = mv_info.get("symbol", mv_name)
            ax.set_xlabel(f"{mv_symbol} [{mv_units}]", fontsize=7)

fig.suptitle(
    "Column A Model – Steady-state I/O characteristics (LV configuration)",
    fontsize=9,
)
plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss.png", dpi=150)
plt.show()

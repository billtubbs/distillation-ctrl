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
import pandas as pd

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
    "x0": {"name": "Bottoms comp.", "symbol": r"$x_B$", "units": "mol/mol"},
    "x40": {
        "name": "Distillate comp.",
        "symbol": r"$x_D$",
        "units": "mol/mol",
    },
    "D": {"name": "Distillate flow", "symbol": r"$D$", "units": "kmol/min"},
    "B": {"name": "Bottoms flow", "symbol": r"$B$", "units": "kmol/min"},
}

# Nominal steady-state CV values (for reference lines on plots)
CV_NOM = {
    "x0": X_SS[0],
    "x40": X_SS[40],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

# Sweep ranges: ±step_size around nominal
MV_STEPS = [0.10, 0.10, 0.10, 0.05, 0.10]
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

# ── Export results to CSV ─────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for mv_name in MV_NAMES:
    df = pd.DataFrame(
        {mv_name: MV_SWEEPS[mv_name], **results[mv_name]}
    )
    csv_path = DATA_DIR / f"ss_{mv_name}.csv"
    df.to_csv(csv_path, index=False, float_format="%.8g")

print(f"\nSaved CSVs to {DATA_DIR}/")

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

        if col == 0:
            ax.set_ylabel(
                f"{cv_info['name']} ({cv_info['symbol']})\n[{cv_info['units']}]",
                fontsize=8,
            )
        if row == n_cvs - 1:
            mv_symbol = mv_info.get("symbol", mv_name)
            mv_label = mv_info.get("name", mv_name)
            ax.set_xlabel(
                f"{mv_label} ({mv_symbol})\n[{mv_units}]", fontsize=7
            )

fig.suptitle(
    "Column A Model – Steady-state I/O characteristics (LV configuration)",
    fontsize=9,
)
plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss.png", dpi=150)
plt.show()

# ── Special plot: xD and xB vs F and zF (deviations) ────────────────────
print("Plotting feed conditions plots …")

FEED_MVS = ["F", "zF"]

# Pre-compute data for reuse across both figures.
feed_data = {}
for mv_name in FEED_MVS:
    nom_mv = U_NOM[MV_NAMES.index(mv_name)]
    feed_data[mv_name] = {
        "d_mv": MV_SWEEPS[mv_name] - nom_mv,
        "d_xD": results[mv_name]["x40"] - CV_NOM["x40"],
        "d_xB": results[mv_name]["x0"] - CV_NOM["x0"],
        "xD":   results[mv_name]["x40"],
        "xB":   results[mv_name]["x0"],
    }


def _plot_feed_linear(axs):
    """Impurity compositions on a linear scale vs MV deviation."""
    for ax, mv_name in zip(axs, FEED_MVS):
        mv_inf = var_info.get(mv_name, {})
        mv_sym = mv_inf.get("symbol", mv_name)
        mv_unt = mv_inf.get("units", "")
        if mv_unt == "dimensionless":
            mv_unt = "—"
        d = feed_data[mv_name]
        ax.plot(d["d_mv"], 1 - d["xD"], color="C0", label=r"$1 - x_D$ (distillate)")
        ax.plot(d["d_mv"], d["xB"],      color="C1", label=r"$x_B$ (bottoms)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.plot(0, 1 - CV_NOM["x40"], "o", color="C0", markersize=5)
        ax.plot(0, CV_NOM["x0"],       "o", color="C1", markersize=5)
        ax.set_xlabel(
            f"Δ {mv_inf.get('name', mv_name)} (Δ{mv_sym})\n[{mv_unt}]", fontsize=9
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)


def _plot_feed_log(axs):
    """Impurity compositions on a log scale vs MV deviation.

    Plots xB and (1 - xD) so both curves occupy the same region of the
    log scale (both ≈ 0.01 at the nominal operating point).
    """
    for ax, mv_name in zip(axs, FEED_MVS):
        mv_inf = var_info.get(mv_name, {})
        mv_sym = mv_inf.get("symbol", mv_name)
        mv_unt = mv_inf.get("units", "")
        if mv_unt == "dimensionless":
            mv_unt = "—"
        d = feed_data[mv_name]
        ax.plot(d["d_mv"], 1 - d["xD"], color="C0", label=r"$1 - x_D$ (distillate)")
        ax.plot(d["d_mv"], d["xB"],      color="C1", label=r"$x_B$ (bottoms)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.plot(0, 1 - CV_NOM["x40"], "o", color="C0", markersize=5)
        ax.plot(0, CV_NOM["x0"],       "o", color="C1", markersize=5)
        ax.set_yscale("log")
        ax.set_xlabel(
            f"Δ {mv_inf.get('name', mv_name)} (Δ{mv_sym})\n[{mv_unt}]", fontsize=9
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, which="both")


fig2, axs2 = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True, sharey=True)
_plot_feed_linear(axs2)
axs2[0].set_ylabel("Impurity [mol/mol]", fontsize=9)
fig2.suptitle(
    "Column A Model – Product stream impurities vs feed conditions (LV configuration)",
    fontsize=9,
)
plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_feed.png", dpi=150)
plt.show()

fig3, axs3 = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True, sharey=True)
_plot_feed_log(axs3)
axs3[0].set_ylabel("Impurity [mol/mol]", fontsize=9)
fig3.suptitle(
    "Column A Model – Product stream impurities vs feed conditions (LV configuration) – log scale",
    fontsize=9,
)
plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_feed_log.png", dpi=150)
plt.show()

# ── Joint LT / VB sweeps ─────────────────────────────────────────────────
print("\nComputing joint LT/VB sweeps …")

LT_IDX = MV_NAMES.index("LT")
VB_IDX = MV_NAMES.index("VB")
LT_STEP = MV_STEPS[LT_IDX]
VB_STEP = MV_STEPS[VB_IDX]
N_JOINT = 31
t_vals = np.linspace(-1, 1, N_JOINT)
nom_t_idx = int(np.searchsorted(t_vals, 0))

# Each entry: (column title, LT values, VB values)
JOINT_SWEEPS = [
    (
        r"$\Delta L_T = \Delta V_B$ (together)",
        U_NOM[LT_IDX] + t_vals * LT_STEP,
        U_NOM[VB_IDX] + t_vals * VB_STEP,
    ),
    (
        r"$\Delta L_T = -\Delta V_B$ (opposite)",
        U_NOM[LT_IDX] + t_vals * LT_STEP,
        U_NOM[VB_IDX] - t_vals * VB_STEP,
    ),
]

joint_results = []
for title, lt_vals, vb_vals in JOINT_SWEEPS:
    cv_vals = {cv: np.full(N_JOINT, np.nan) for cv in CV_OUTPUT_NAMES}

    x0 = X_SS.copy()
    for k in range(nom_t_idx, N_JOINT):
        u = U_NOM.copy()
        u[LT_IDX] = lt_vals[k]
        u[VB_IDX] = vb_vals[k]
        x_ss, y_ss = ss_solver(x0, u, param_vals)
        for cv in CV_OUTPUT_NAMES:
            cv_vals[cv][k] = y_ss[output_idx[cv]]
        x0 = x_ss

    x0 = X_SS.copy()
    for k in range(nom_t_idx - 1, -1, -1):
        u = U_NOM.copy()
        u[LT_IDX] = lt_vals[k]
        u[VB_IDX] = vb_vals[k]
        x_ss, y_ss = ss_solver(x0, u, param_vals)
        for cv in CV_OUTPUT_NAMES:
            cv_vals[cv][k] = y_ss[output_idx[cv]]
        x0 = x_ss

    joint_results.append(cv_vals)
    print(f"  {title!r}: {N_JOINT} points")

# ── Plot: 4 CVs × 2 joint sweeps ─────────────────────────────────────────
print("\nPlotting joint LT/VB sweeps …")

d_lt = t_vals * LT_STEP  # ΔLT — common x-axis for both columns

fig4, axs4 = plt.subplots(
    n_cvs,
    2,
    figsize=(6, 2.5 * n_cvs),
    sharex=True,
    sharey="row",
    constrained_layout=True,
)

for col, ((title, _, _), cv_vals) in enumerate(zip(JOINT_SWEEPS, joint_results)):
    for row, cv_name in enumerate(CV_OUTPUT_NAMES):
        ax = axs4[row, col]
        cv_info = CV_INFO[cv_name]

        ax.plot(d_lt, cv_vals[cv_name], color="C0")
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(CV_NOM[cv_name], color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.plot(0, CV_NOM[cv_name], "ko", markersize=4)
        ax.grid(True, alpha=0.3)

        if row == 0:
            ax.set_title(title, fontsize=8)
        if col == 0:
            ax.set_ylabel(
                f"{cv_info['name']} ({cv_info['symbol']})\n[{cv_info['units']}]",
                fontsize=8,
            )
        if row == n_cvs - 1:
            ax.set_xlabel(
                r"$\Delta L_T$ [kmol/min]" + f"\n"
                + (r"$\Delta V_B = \Delta L_T$" if col == 0 else r"$\Delta V_B = -\Delta L_T$"),
                fontsize=7,
            )

fig4.suptitle(
    "Column A Model – Joint $L_T$ / $V_B$ steady-state sweeps (LV configuration)",
    fontsize=9,
)
plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_joint.png", dpi=150)
plt.show()

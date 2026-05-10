"""
dist_model_cola_lv_ss.py – Steady-state I/O characteristics of the
LV closed-loop Column A distillation model.

For each MV/DV (LT, VB, F, zF, qF), the steady-state values of the four
key CVs (xB, xD, D, B) are computed over a range of input values using
Newton rootfinding on the model ODE RHS.  Starting from the known nominal
steady state, each successive sweep point is warm-started from the previous
solution, acting as a simple continuation method.

Produces a 4 × 5 grid of static gain curves, one subplot per (CV, MV) pair.

Usage:
    python dist_model_cola_lv_ss.py                       # all plots
    python dist_model_cola_lv_ss.py main abstract         # selected plots

Available plot names: main, abstract, feed, feed_log, joint,
                      targeting, regression, sensitivity, total_impurity
"""

import argparse
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
from sim_utils import (
    make_composition_targeting_solver,
    make_steady_state_solver,
)

# ── Configuration ─────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ZF_NOM = 0.5
U_NOM = np.array([L0_DEFAULT, V0_DEFAULT, F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

MV_NAMES = ["LT", "VB", "F", "zF", "qF"]
CV_OUTPUT_NAMES = ["x0", "x40", "D", "B"]

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

CV_NOM = {
    "x0": X_SS[0],
    "x40": X_SS[40],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

MV_STEPS = [0.10, 0.10, 0.10, 0.05, 0.10]
MV_SWEEPS = {
    mv: np.linspace(U_NOM[j] - step, U_NOM[j] + step, 31)
    for j, (mv, step) in enumerate(zip(MV_NAMES, MV_STEPS))
}

# ── Analysis configuration ─────────────────────────────────────────────────
LT_IDX = MV_NAMES.index("LT")
VB_IDX = MV_NAMES.index("VB")
LT_STEP = MV_STEPS[LT_IDX]
VB_STEP = MV_STEPS[VB_IDX]

FEED_MVS = ["F", "zF"]

FEED_SWEEP_NAMES = ["F", "zF", "qF"]
U_FIXED_NOM = np.array([F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

TARGET_OUTPUT_NAMES = ["LT", "VB", "D", "B"]
TARGET_OUTPUT_NOM = {
    "LT": U_NOM[0],
    "VB": U_NOM[1],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

REG_OUTPUT_NAMES = ["LT", "VB", "D", "B", "x40", "x0"]
REG_OUTPUT_INFO = {
    "LT": var_info["LT"],
    "VB": var_info["VB"],
    "D": var_info["D"],
    "B": var_info["B"],
    "x40": CV_INFO["x40"],
    "x0": CV_INFO["x0"],
}
REG_OUTPUT_NOM = {
    "LT": U_NOM[0],
    "VB": U_NOM[1],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
    "x40": CV_NOM["x40"],
    "x0": CV_NOM["x0"],
}

SCENARIOS = [
    ("Base (nominal)", 1.00, "C0", "-"),
    ("+1% $L_T$, $V_B$", 1.01, "C2", "--"),
    ("−1% $L_T$, $V_B$", 0.99, "C3", ":"),
]

SENS_FEED_MVS = ["F", "zF"]
SENS_CVS = ["x0", "x40"]

PLOT_DESCRIPTIONS = {
    "main": "Steady-state I/O characteristics (4 CVs x 5 MVs grid)",
    "abstract": "Steady-state I/O characteristics (qualitative version for publication)",
    "feed": "Product stream impurities vs feed conditions",
    "feed_log": "Product stream impurities vs feed conditions (log scale)",
    "joint": "Joint LT/VB steady-state sweeps",
    "targeting": "Required LT, VB to maintain nominal compositions vs feed conditions",
    "regression": "Regression-predicted LT, VB and resulting compositions vs feed conditions",
    "sensitivity": "Composition sensitivity at nominal, +1% and -1% LT/VB",
    "total_impurity": "Combined purity metric xB + xD - 1 vs feed flow rate",
}


# ── Computation functions ──────────────────────────────────────────────────
def compute_ss_sweeps(ss_solver, param_vals, output_idx):
    print("\nComputing steady-state sweeps …")
    results = {}
    for j, mv_name in enumerate(MV_NAMES):
        mv_vals = MV_SWEEPS[mv_name]
        n_pts = len(mv_vals)
        cv_vals = {cv: np.full(n_pts, np.nan) for cv in CV_OUTPUT_NAMES}
        nom_idx = int(np.searchsorted(mv_vals, U_NOM[j]))

        x0 = X_SS.copy()
        for k in range(nom_idx, n_pts):
            u = U_NOM.copy()
            u[j] = mv_vals[k]
            x_ss, y_ss = ss_solver(x0, u, param_vals)
            for cv in CV_OUTPUT_NAMES:
                cv_vals[cv][k] = y_ss[output_idx[cv]]
            x0 = x_ss

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
    return results


def compute_feed_data(results):
    feed_data = {}
    for mv_name in FEED_MVS:
        nom_mv = U_NOM[MV_NAMES.index(mv_name)]
        feed_data[mv_name] = {
            "d_mv": MV_SWEEPS[mv_name] - nom_mv,
            "d_xD": results[mv_name]["x40"] - CV_NOM["x40"],
            "d_xB": results[mv_name]["x0"] - CV_NOM["x0"],
            "xD": results[mv_name]["x40"],
            "xB": results[mv_name]["x0"],
        }
    return feed_data


def compute_joint_results(ss_solver, param_vals, output_idx):
    print("\nComputing joint LT/VB sweeps …")
    N_JOINT = 31
    t_vals = np.linspace(-1, 1, N_JOINT)
    nom_t_idx = int(np.searchsorted(t_vals, 0))

    joint_sweeps = [
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
    for title, lt_vals, vb_vals in joint_sweeps:
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

    return joint_sweeps, joint_results, t_vals


def compute_targeting_results(model, ss_solver, param_vals, output_idx):
    print("\nBuilding composition-targeting solver …")
    comp_solver = make_composition_targeting_solver(
        model,
        free_input_indices=[0, 1],  # LT, VB
        target_state_indices=[0, 40],  # xB = x[0], xD = x[40]
    )

    XB_TARGET = CV_NOM["x0"]
    XD_TARGET = CV_NOM["x40"]
    Z0_NOM = np.concatenate([X_SS, [L0_DEFAULT, V0_DEFAULT]])

    print("Computing composition-targeting sweeps …")
    targeting_results = {}

    for j_feed, feed_mv_name in enumerate(FEED_SWEEP_NAMES):
        mv_vals = MV_SWEEPS[feed_mv_name]
        n_pts = len(mv_vals)
        out_vals = {k: np.full(n_pts, np.nan) for k in TARGET_OUTPUT_NAMES}
        nom_idx = int(
            np.searchsorted(mv_vals, U_NOM[MV_NAMES.index(feed_mv_name)])
        )

        for direction, rng in [
            ("up", range(nom_idx, n_pts)),
            ("down", range(nom_idx - 1, -1, -1)),
        ]:
            z0 = Z0_NOM.copy()
            for k in rng:
                u_fixed = U_FIXED_NOM.copy()
                u_fixed[j_feed] = mv_vals[k]
                x_ss, u_free_ss, y_ss = comp_solver(
                    z0, u_fixed, [XB_TARGET, XD_TARGET], param_vals
                )
                out_vals["LT"][k] = u_free_ss[0]
                out_vals["VB"][k] = u_free_ss[1]
                out_vals["D"][k] = y_ss[output_idx["D"]]
                out_vals["B"][k] = y_ss[output_idx["B"]]
                z0 = np.concatenate([x_ss, u_free_ss])

        targeting_results[feed_mv_name] = out_vals
        print(f"  {feed_mv_name}: {n_pts} points")

    return targeting_results


def compute_regression(targeting_results, ss_solver, param_vals, output_idx):
    print("\nFitting linear regression models …")

    X_rows, Y_dLT_rows, Y_dVB_rows = [], [], []
    for j_feed, feed_mv_name in enumerate(FEED_SWEEP_NAMES):
        mv_vals = MV_SWEEPS[feed_mv_name]
        for k, mv_val in enumerate(mv_vals):
            u_fixed = U_FIXED_NOM.copy()
            u_fixed[j_feed] = mv_val
            F_val, zF_val, qF_val = u_fixed
            X_rows.append(
                [F_val - F0_DEFAULT, zF_val - ZF_NOM, qF_val - QF0_DEFAULT]
            )
            Y_dLT_rows.append(
                targeting_results[feed_mv_name]["LT"][k] - U_NOM[0]
            )
            Y_dVB_rows.append(
                targeting_results[feed_mv_name]["VB"][k] - U_NOM[1]
            )

    X_reg = np.array(X_rows)
    Y_dLT_reg = np.array(Y_dLT_rows)
    Y_dVB_reg = np.array(Y_dVB_rows)

    coeffs_LT, _, _, _ = np.linalg.lstsq(X_reg, Y_dLT_reg, rcond=None)
    coeffs_VB, _, _, _ = np.linalg.lstsq(X_reg, Y_dVB_reg, rcond=None)

    print(
        f"  ΔLT = {coeffs_LT[0]:.4f}·ΔF"
        f" + {coeffs_LT[1]:.4f}·ΔzF"
        f" + {coeffs_LT[2]:.4f}·ΔqF"
    )
    print(
        f"  ΔVB = {coeffs_VB[0]:.4f}·ΔF"
        f" + {coeffs_VB[1]:.4f}·ΔzF"
        f" + {coeffs_VB[2]:.4f}·ΔqF"
    )

    LT_nom, VB_nom = U_NOM[0], U_NOM[1]
    print(
        f"  LT = {coeffs_LT[0]:.4f}·(F - {F0_DEFAULT})"
        f" + {coeffs_LT[1]:.4f}·(zF - {ZF_NOM})"
        f" + {coeffs_LT[2]:.4f}·(qF - {QF0_DEFAULT})"
        f" + {LT_nom:.5f}"
    )
    print(
        f"  VB = {coeffs_VB[0]:.4f}·(F - {F0_DEFAULT})"
        f" + {coeffs_VB[1]:.4f}·(zF - {ZF_NOM})"
        f" + {coeffs_VB[2]:.4f}·(qF - {QF0_DEFAULT})"
        f" + {VB_nom:.5f}"
    )

    def predict_lt_vb(F_val, zF_val, qF_val):
        x_feat = np.array(
            [F_val - F0_DEFAULT, zF_val - ZF_NOM, qF_val - QF0_DEFAULT]
        )
        return (
            float(U_NOM[0] + coeffs_LT @ x_feat),
            float(U_NOM[1] + coeffs_VB @ x_feat),
        )

    print("Computing regression-based sweeps …")
    regression_results = {}

    for j_feed, feed_mv_name in enumerate(FEED_SWEEP_NAMES):
        mv_vals = MV_SWEEPS[feed_mv_name]
        n_pts = len(mv_vals)
        out_vals = {k: np.full(n_pts, np.nan) for k in REG_OUTPUT_NAMES}
        nom_idx = int(
            np.searchsorted(mv_vals, U_NOM[MV_NAMES.index(feed_mv_name)])
        )

        for direction, rng in [
            ("up", range(nom_idx, n_pts)),
            ("down", range(nom_idx - 1, -1, -1)),
        ]:
            x0 = X_SS.copy()
            for k in rng:
                u_fixed = U_FIXED_NOM.copy()
                u_fixed[j_feed] = mv_vals[k]
                F_val, zF_val, qF_val = u_fixed
                LT_pred, VB_pred = predict_lt_vb(F_val, zF_val, qF_val)
                u = np.array([LT_pred, VB_pred, F_val, zF_val, qF_val])
                x_ss, y_ss = ss_solver(x0, u, param_vals)
                out_vals["LT"][k] = LT_pred
                out_vals["VB"][k] = VB_pred
                out_vals["D"][k] = y_ss[output_idx["D"]]
                out_vals["B"][k] = y_ss[output_idx["B"]]
                out_vals["x40"][k] = y_ss[output_idx["x40"]]
                out_vals["x0"][k] = y_ss[output_idx["x0"]]
                x0 = x_ss

        regression_results[feed_mv_name] = out_vals
        print(f"  {feed_mv_name}: {n_pts} points")

    return regression_results


def compute_sensitivity(ss_solver, param_vals, output_idx):
    print("\nComputing base / high / low LT-VB sensitivity sweeps …")
    sens_results = {}

    for scen_label, scale, color, ls in SCENARIOS:
        lt_val = scale * L0_DEFAULT
        vb_val = scale * V0_DEFAULT
        scen_res = {}

        for feed_mv_name in SENS_FEED_MVS:
            mv_vals = MV_SWEEPS[feed_mv_name]
            n_pts = len(mv_vals)
            cv_vals = {cv: np.full(n_pts, np.nan) for cv in SENS_CVS}
            nom_idx = int(
                np.searchsorted(mv_vals, U_NOM[MV_NAMES.index(feed_mv_name)])
            )

            u_nom_scen = U_NOM.copy()
            u_nom_scen[0] = lt_val
            u_nom_scen[1] = vb_val
            x0_scen, _ = ss_solver(X_SS, u_nom_scen, param_vals)

            for rng in [range(nom_idx, n_pts), range(nom_idx - 1, -1, -1)]:
                x0 = x0_scen.copy()
                for k in rng:
                    u = u_nom_scen.copy()
                    u[MV_NAMES.index(feed_mv_name)] = mv_vals[k]
                    x_ss, y_ss = ss_solver(x0, u, param_vals)
                    for cv in SENS_CVS:
                        cv_vals[cv][k] = y_ss[output_idx[cv]]
                    x0 = x_ss

            scen_res[feed_mv_name] = cv_vals
        sens_results[scen_label] = scen_res
        print(f"  {scen_label}: done")

    return sens_results


# ── Plot helper functions ──────────────────────────────────────────────────
def _plot_feed_linear(axs, feed_data):
    """Impurity compositions on a linear scale vs MV deviation."""
    for ax, mv_name in zip(axs, FEED_MVS):
        mv_inf = var_info.get(mv_name, {})
        mv_sym = mv_inf.get("symbol", mv_name)
        mv_unt = mv_inf.get("units", "")
        if mv_unt == "dimensionless":
            mv_unt = "—"
        d = feed_data[mv_name]
        ax.plot(
            d["d_mv"], 1 - d["xD"], color="C0", label=r"$1 - x_D$ (distillate)"
        )
        ax.plot(d["d_mv"], d["xB"], color="C1", label=r"$x_B$ (bottoms)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.plot(0, 1 - CV_NOM["x40"], "o", color="C0", markersize=5)
        ax.plot(0, CV_NOM["x0"], "o", color="C1", markersize=5)
        ax.set_xlabel(
            f"Δ {mv_inf.get('name', mv_name)} (Δ{mv_sym})\n[{mv_unt}]",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)


def _plot_feed_log(axs, feed_data):
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
        ax.plot(
            d["d_mv"], 1 - d["xD"], color="C0", label=r"$1 - x_D$ (distillate)"
        )
        ax.plot(d["d_mv"], d["xB"], color="C1", label=r"$x_B$ (bottoms)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.plot(0, 1 - CV_NOM["x40"], "o", color="C0", markersize=5)
        ax.plot(0, CV_NOM["x0"], "o", color="C1", markersize=5)
        ax.set_yscale("log")
        ax.set_xlabel(
            f"Δ {mv_inf.get('name', mv_name)} (Δ{mv_sym})\n[{mv_unt}]",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, which="both")


# ── Plot functions ─────────────────────────────────────────────────────────
def plot_main(results):
    print("\nPlotting main grid …")
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
            ax.axvline(
                nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
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


def plot_abstract(results, subplot_size=(1.75, 1.75)):
    print("\nPlotting abstract version …")
    n_cvs = len(CV_OUTPUT_NAMES)
    n_mvs = len(MV_NAMES)

    fig, axs = plt.subplots(
        n_cvs,
        n_mvs,
        figsize=(subplot_size[0] * n_mvs, subplot_size[1] * n_cvs),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    for col, mv_name in enumerate(MV_NAMES):
        mv_info = var_info.get(mv_name, {})
        mv_vals = MV_SWEEPS[mv_name]
        nom_mv = U_NOM[col]

        for row, cv_name in enumerate(CV_OUTPUT_NAMES):
            ax = axs[row, col]
            cv_info = CV_INFO[cv_name]

            ax.plot(mv_vals, results[mv_name][cv_name], color="C0")
            ax.axvline(
                nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                CV_NOM[cv_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
            ax.plot(nom_mv, CV_NOM[cv_name], "ko", markersize=3)

            ax.margins(0.12)
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            ax.grid(False)

            if col == 0:
                ax.set_ylabel(f"{cv_info['name']}\n({cv_info['symbol']})")
            if row == n_cvs - 1:
                mv_symbol = mv_info.get("symbol", mv_name)
                mv_label = mv_info.get("name", mv_name)
                ax.set_xlabel(f"{mv_label}\n({mv_symbol})")

    fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.94])
    fig.suptitle(
        "Column A Model – Steady-state I/O characteristics (LV configuration)",
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_abstract.png", dpi=150)
    plt.show()


def plot_feed(feed_data):
    print("\nPlotting feed conditions plots …")
    fig, axs = plt.subplots(
        1, 2, figsize=(9, 4), constrained_layout=True, sharey=True
    )
    _plot_feed_linear(axs, feed_data)
    axs[0].set_ylabel("Impurity [mol/mol]", fontsize=9)
    fig.suptitle(
        "Column A Model – Product stream impurities vs feed conditions "
        "(LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_feed.png", dpi=150)
    plt.show()


def plot_feed_log(feed_data):
    print("\nPlotting feed conditions plots (log scale) …")
    fig, axs = plt.subplots(
        1, 2, figsize=(9, 4), constrained_layout=True, sharey=True
    )
    _plot_feed_log(axs, feed_data)
    axs[0].set_ylabel("Impurity [mol/mol]", fontsize=9)
    fig.suptitle(
        "Column A Model – Product stream impurities vs feed conditions "
        "(LV configuration) – log scale",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_feed_log.png", dpi=150)
    plt.show()


def plot_joint(joint_sweeps, joint_results, t_vals):
    print("\nPlotting joint LT/VB sweeps …")
    n_cvs = len(CV_OUTPUT_NAMES)
    d_lt = t_vals * LT_STEP

    fig, axs = plt.subplots(
        n_cvs,
        2,
        figsize=(6, 2.5 * n_cvs),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for col, ((title, _, _), cv_vals) in enumerate(
        zip(joint_sweeps, joint_results)
    ):
        for row, cv_name in enumerate(CV_OUTPUT_NAMES):
            ax = axs[row, col]
            cv_info = CV_INFO[cv_name]

            ax.plot(d_lt, cv_vals[cv_name], color="C0")
            ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.axhline(
                CV_NOM[cv_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.6,
            )
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
                    r"$\Delta L_T$ [kmol/min]"
                    + "\n"
                    + (
                        r"$\Delta V_B = \Delta L_T$"
                        if col == 0
                        else r"$\Delta V_B = -\Delta L_T$"
                    ),
                    fontsize=7,
                )

    fig.suptitle(
        "Column A Model – Joint $L_T$ / $V_B$ steady-state sweeps "
        "(LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_joint.png", dpi=150)
    plt.show()


def plot_targeting(targeting_results):
    print("\nPlotting composition-targeting results …")
    n_out = len(TARGET_OUTPUT_NAMES)
    n_feed = len(FEED_SWEEP_NAMES)

    fig, axs = plt.subplots(
        n_out,
        n_feed,
        figsize=(3.0 * n_feed, 2.5 * n_out),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for col, feed_mv_name in enumerate(FEED_SWEEP_NAMES):
        feed_mv_info = var_info.get(feed_mv_name, {})
        feed_mv_sym = feed_mv_info.get("symbol", feed_mv_name)
        feed_mv_unt = feed_mv_info.get("units", "")
        if feed_mv_unt == "dimensionless":
            feed_mv_unt = "—"
        mv_vals = MV_SWEEPS[feed_mv_name]
        nom_mv = U_NOM[MV_NAMES.index(feed_mv_name)]

        for row, out_name in enumerate(TARGET_OUTPUT_NAMES):
            ax = axs[row, col]
            out_info = var_info.get(out_name, {})
            nom_val = TARGET_OUTPUT_NOM[out_name]

            ax.plot(
                mv_vals, targeting_results[feed_mv_name][out_name], color="C2"
            )
            ax.axvline(
                nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                nom_val, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.plot(nom_mv, nom_val, "ko", markersize=4)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(
                    f"{out_info.get('name', out_name)} "
                    f"({out_info.get('symbol', out_name)})"
                    f"\n[{out_info.get('units', '')}]",
                    fontsize=8,
                )
            if row == n_out - 1:
                ax.set_xlabel(
                    f"{feed_mv_info.get('name', feed_mv_name)} ({feed_mv_sym})\n"
                    f"[{feed_mv_unt}]",
                    fontsize=7,
                )

    fig.suptitle(
        "Column A Model – Required $L_T$, $V_B$ to maintain nominal compositions"
        " vs feed conditions (LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_targeting.png", dpi=150)
    plt.show()


def plot_regression(regression_results):
    print("\nPlotting regression results …")
    n_reg_out = len(REG_OUTPUT_NAMES)
    n_feed = len(FEED_SWEEP_NAMES)

    fig, axs = plt.subplots(
        n_reg_out,
        n_feed,
        figsize=(3.0 * n_feed, 2.5 * n_reg_out),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for col, feed_mv_name in enumerate(FEED_SWEEP_NAMES):
        feed_mv_info = var_info.get(feed_mv_name, {})
        feed_mv_sym = feed_mv_info.get("symbol", feed_mv_name)
        feed_mv_unt = feed_mv_info.get("units", "")
        if feed_mv_unt == "dimensionless":
            feed_mv_unt = "—"
        mv_vals = MV_SWEEPS[feed_mv_name]
        nom_mv = U_NOM[MV_NAMES.index(feed_mv_name)]

        for row, out_name in enumerate(REG_OUTPUT_NAMES):
            ax = axs[row, col]
            out_info = REG_OUTPUT_INFO[out_name]
            nom_val = REG_OUTPUT_NOM[out_name]

            ax.plot(
                mv_vals, regression_results[feed_mv_name][out_name], color="C3"
            )
            ax.axvline(
                nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                nom_val, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.plot(nom_mv, nom_val, "ko", markersize=4)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(
                    f"{out_info.get('name', out_name)} "
                    f"({out_info.get('symbol', out_name)})"
                    f"\n[{out_info.get('units', '')}]",
                    fontsize=8,
                )
            if row == n_reg_out - 1:
                ax.set_xlabel(
                    f"{feed_mv_info.get('name', feed_mv_name)} ({feed_mv_sym})\n"
                    f"[{feed_mv_unt}]",
                    fontsize=7,
                )

    # Enforce a minimum y-axis range on the composition rows so small curvature
    # is not exaggerated by tight auto-scaling.
    MIN_COMP_RANGE = 0.02
    for out_name in ("x40", "x0"):
        row = REG_OUTPUT_NAMES.index(out_name)
        ax = axs[
            row, 0
        ]  # sharey="row" — setting one propagates to the whole row
        ymin, ymax = ax.get_ylim()
        if ymax - ymin < MIN_COMP_RANGE:
            ymid = (ymin + ymax) / 2
            ax.set_ylim(ymid - MIN_COMP_RANGE / 2, ymid + MIN_COMP_RANGE / 2)

    fig.suptitle(
        "Column A Model – Regression-predicted $L_T$, $V_B$ and "
        "resulting compositions vs feed conditions (LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_regression.png", dpi=150)
    plt.show()


def plot_sensitivity(sens_results):
    print("\nPlotting sensitivity results …")
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(9, 7),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for col, feed_mv_name in enumerate(SENS_FEED_MVS):
        feed_mv_info = var_info.get(feed_mv_name, {})
        feed_mv_sym = feed_mv_info.get("symbol", feed_mv_name)
        feed_mv_unt = feed_mv_info.get("units", "")
        if feed_mv_unt == "dimensionless":
            feed_mv_unt = "—"
        mv_vals = MV_SWEEPS[feed_mv_name]
        nom_mv = U_NOM[MV_NAMES.index(feed_mv_name)]

        for row, cv_name in enumerate(SENS_CVS):
            ax = axs[row, col]
            cv_info = CV_INFO[cv_name]

            for scen_label, scale, color, ls in SCENARIOS:
                ax.plot(
                    mv_vals,
                    sens_results[scen_label][feed_mv_name][cv_name],
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                    label=scen_label,
                )

            ax.axvline(
                nom_mv, color="k", linewidth=0.6, linestyle="--", alpha=0.5
            )
            ax.axhline(
                CV_NOM[cv_name],
                color="k",
                linewidth=0.6,
                linestyle="--",
                alpha=0.5,
            )
            ax.plot(nom_mv, CV_NOM[cv_name], "ko", markersize=4)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(
                    f"{cv_info['name']} ({cv_info['symbol']})\n[{cv_info['units']}]",
                    fontsize=9,
                )
            if row == len(SENS_CVS) - 1:
                ax.set_xlabel(
                    f"{feed_mv_info.get('name', feed_mv_name)} ({feed_mv_sym})\n"
                    f"[{feed_mv_unt}]",
                    fontsize=9,
                )
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Column A Model – Composition sensitivity to feed conditions\n"
        "at nominal, +1% and −1% $L_T$ / $V_B$ (LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_sensitivity.png", dpi=150)
    plt.show()


def plot_total_impurity(sens_results):
    print("\nPlotting total impurity vs feed flow rate …")
    f_mv_vals = MV_SWEEPS["F"]
    f_nom = U_NOM[MV_NAMES.index("F")]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    for scen_label, scale, color, ls in SCENARIOS:
        xb = sens_results[scen_label]["F"]["x0"]
        xd = sens_results[scen_label]["F"]["x40"]
        ax.plot(
            f_mv_vals,
            xb + xd - 1.0,
            color=color,
            linestyle=ls,
            linewidth=1.6,
            label=scen_label,
        )

    total_imp_nom = CV_NOM["x0"] + CV_NOM["x40"] - 1.0
    ax.axvline(f_nom, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(
        total_imp_nom, color="k", linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax.plot(f_nom, total_imp_nom, "ko", markersize=5)
    ax.set_xlabel(
        f"{var_info['F']['name']} ({var_info['F']['symbol']})\n[{var_info['F']['units']}]",
        fontsize=10,
    )
    ax.set_ylabel(r"$x_B + x_D - 1$  [mol/mol]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        r"Column A Model – Combined purity metric $x_B + x_D - 1$ vs feed flow rate"
        "\nat nominal, +1% and −1% $L_T$ / $V_B$ (LV configuration)",
        fontsize=9,
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_ss_total_impurity.png", dpi=150)
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate steady-state I/O characteristic plots for the "
            "Column A LV model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "available plots:\n"
            + "".join(
                f"  {name:<16}  {PLOT_DESCRIPTIONS[name]}\n"
                for name in PLOT_DESCRIPTIONS.keys()
            )
        ),
    )
    parser.add_argument(
        "plots",
        nargs="*",
        metavar="PLOT",
        help=(
            "one or more plot names to generate (default: all);\n"
            "see available plots below"
        ),
    )
    args = parser.parse_args()

    unknown = set(args.plots) - set(PLOT_DESCRIPTIONS.keys())
    if unknown:
        parser.error(
            f"unknown plot name(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(PLOT_DESCRIPTIONS.keys())}"
        )

    selected = set(args.plots) if args.plots else set(PLOT_DESCRIPTIONS.keys())

    # ── Build model ──────────────────────────────────────────────────────
    print("Building model and compiling steady-state solver …")
    model = build_cola_lv_ct_model()
    param_vals = make_nominal_lv_param_values()
    ss_solver = make_steady_state_solver(model)
    print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")

    output_idx = {name: i for i, name in enumerate(model.output_names)}

    x_ss_nom, y_ss_nom = ss_solver(X_SS, U_NOM, param_vals)
    print(
        f"  Nominal SS: xB={y_ss_nom[output_idx['x0']]:.6f}, "
        f"xD={y_ss_nom[output_idx['x40']]:.6f}, "
        f"D={y_ss_nom[output_idx['D']]:.6f}, "
        f"B={y_ss_nom[output_idx['B']]:.6f}"
    )

    # ── SS sweeps: main, abstract, feed, feed_log ────────────────────────
    if selected & {"main", "abstract", "feed", "feed_log"}:
        results = compute_ss_sweeps(ss_solver, param_vals, output_idx)

        DATA_DIR = Path("data")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for mv_name in MV_NAMES:
            df = pd.DataFrame(
                {mv_name: MV_SWEEPS[mv_name], **results[mv_name]}
            )
            df.to_csv(
                DATA_DIR / f"ss_{mv_name}.csv",
                index=False,
                float_format="%.8g",
            )
        print(f"\nSaved CSVs to {DATA_DIR}/")

    if "main" in selected:
        plot_main(results)
    if "abstract" in selected:
        plot_abstract(results)

    if selected & {"feed", "feed_log"}:
        feed_data = compute_feed_data(results)
        if "feed" in selected:
            plot_feed(feed_data)
        if "feed_log" in selected:
            plot_feed_log(feed_data)

    # ── Joint LT/VB sweeps ───────────────────────────────────────────────
    if "joint" in selected:
        joint_sweeps, joint_results, t_vals = compute_joint_results(
            ss_solver, param_vals, output_idx
        )
        plot_joint(joint_sweeps, joint_results, t_vals)

    # ── Composition targeting and regression ─────────────────────────────
    if selected & {"targeting", "regression"}:
        targeting_results = compute_targeting_results(
            model, ss_solver, param_vals, output_idx
        )
        if "targeting" in selected:
            plot_targeting(targeting_results)
        if "regression" in selected:
            regression_results = compute_regression(
                targeting_results, ss_solver, param_vals, output_idx
            )
            plot_regression(regression_results)

    # ── Sensitivity ──────────────────────────────────────────────────────
    if selected & {"sensitivity", "total_impurity"}:
        sens_results = compute_sensitivity(ss_solver, param_vals, output_idx)
        if "sensitivity" in selected:
            plot_sensitivity(sens_results)
        if "total_impurity" in selected:
            plot_total_impurity(sens_results)

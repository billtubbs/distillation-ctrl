"""
dist_model_cola_lv_steps.py – Step responses for the LV closed-loop
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

Outputs (ny=84):
    x[0..40]  : liquid mole fraction of light component A  [mol/mol]
                (x[0] = reboiler/bottoms, x[40] = condenser/distillate)
    M[0..40]  : molar liquid holdup on each stage           [kmol]
                (M[0] = reboiler, M[40] = condenser)
    D         : distillate (top product) flow               [kmol/min]
    B         : bottoms (bottom product) flow               [kmol/min]

Monitored outputs (CVs):
    x[0]   xB  : bottoms composition     [mol/mol]  SS ~0.01
    x[40]  xD  : distillate composition  [mol/mol]  SS ~0.99
    D           : distillate flow        [kmol/min] SS = 0.5
    B           : bottoms flow           [kmol/min] SS = 0.5

For each non-zero MV step, one figure is produced with:
    upper subplots: CV deviations from steady state
    bottom subplot: MV step

Additional scenario: Table 11.1 step disturbances in feed rate F,
feed composition zF, reflux LT, and boilup VB (seven steps at
t=10, 100, 200, 300, 400, 500, 600 min), with product impurities
(xB and 1-xD), product flows, and all disturbance inputs shown.

Usage:
    python dist_model_cola_lv_steps.py                         # all plots
    python dist_model_cola_lv_steps.py steps scenario          # selected

Available plot names: steps, scenario, comp_profiles, holdup_profiles
"""

import argparse
from pathlib import Path

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
from cas_models.continuous_time.simulate import (
    make_n_step_simulation_function_from_model,
)
from dist_model_cola_cas.cola_lv_model import (
    BS_DEFAULT,
    DS_DEFAULT,
    X_SS,
    build_cola_lv_ct_model,
    make_nominal_lv_param_values,
)
from dist_model_cola_cas.var_info import var_info
from plot_utils import make_tsplots
from sim_utils import run_simulation

# ── Configuration ─────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ZF_NOM = 0.5
U_NOM = np.array([L0_DEFAULT, V0_DEFAULT, F0_DEFAULT, ZF_NOM, QF0_DEFAULT])

CV_OUTPUT_NAMES = ["x0", "x40", "D", "B"]
CV_NAMES = ["xB", "xD", "D", "B"]

MV_NAMES = ["LT", "VB", "F", "zF", "qF"]
MV_STEPS = [0.30, 0.30, 0.20, 0.10, 0.10]

DT = 1.0  # integration step [min]
T_HORIZON = 1000  # step-response simulation length [min]
NT_SIM = T_HORIZON
STEP_TIME = 20  # [min] time of initial step
RETURN_TIME = 400  # [min] time of return step (-2x original)

NONZERO_THR = 1e-5  # max deviation below which response is considered zero

CV_OUTPUT_REFS = {
    "x0": X_SS[0],
    "x40": X_SS[40],
    "D": DS_DEFAULT,
    "B": BS_DEFAULT,
}

# ── Extend var_info with per-stage state variables ─────────────────────────
_COMP_NAMES = {0: "Bottoms composition", NT - 1: "Distillate composition"}
for _i in range(NT):
    var_info[f"x{_i}"] = {
        "name": _COMP_NAMES.get(_i, f"Stage {_i} composition"),
        "symbol": rf"$x_{{{_i}}}$",
        "units": "mol/mol",
    }

_HOLDUP_NAMES = {0: "Reboiler holdup", NT - 1: "Condenser holdup"}
for _i in range(NT):
    var_info[f"M{_i}"] = {
        "name": _HOLDUP_NAMES.get(_i, f"Stage {_i} holdup"),
        "symbol": rf"$M_{{{_i}}}$",
        "units": "kmol",
    }

var_info["1-x40"] = {
    "name": "Distillate impurity",
    "symbol": r"$1-x_{40}$",
    "units": "mol/mol",
}

t_eval = pd.Series(
    np.linspace(0.0, NT_SIM * DT, NT_SIM + 1), name="Time [min]"
)

# ── Column profile plot constants ──────────────────────────────────────────
PROFILE_WIDTH = 8.0  # figure width [in]
TRAY_HEIGHT = 0.22  # height per tray subplot [in]
PROFILE_TITLE = (
    "F, zF, LT, VB step disturbances (t=10, 100, 200, 300, 400, 500, 600 min)"
)

PLOT_DESCRIPTIONS = {
    "steps": "Step responses for each MV (one figure per MV)",
    "step_abstract": "Step response matrix (compact qualitative version for publication)",
    "scenario": "Disturbance scenario (F, zF, LT, VB) I/O time series",
    "comp_profiles": "Column composition profiles from scenario simulation",
    "holdup_profiles": "Column holdup profiles from scenario simulation",
}


# ── Profile helper functions ───────────────────────────────────────────────
def _make_profile_axes(title):
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


# ── Computation functions ──────────────────────────────────────────────────
def compute_step_responses(model, sim_func, param_vals):
    """Run step response simulations for each MV.

    Returns a dict mapping mv_name → deviation DataFrame for each MV whose
    response exceeds NONZERO_THR.
    """
    print("\nComputing step responses …")
    step_data = {}
    for j, (mv_name, step_size) in enumerate(zip(MV_NAMES, MV_STEPS)):
        U_arr = np.tile(U_NOM, (NT_SIM, 1))
        U_arr[STEP_TIME:, j] += step_size
        U_arr[RETURN_TIME:, j] -= 2 * step_size
        U_df = pd.DataFrame(U_arr, columns=model.input_names)

        sim_results = run_simulation(
            t_eval, U_df, model, param_vals, X_SS, sim_func=sim_func
        )

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

        dev = sim_results.copy()
        for sn in CV_OUTPUT_NAMES:
            dev[("Outputs", sn)] -= CV_OUTPUT_REFS[sn]
        dev[("Inputs", mv_name)] -= U_NOM[j]

        step_data[mv_name] = dev
    return step_data


def compute_scenario(model, sim_func, param_vals):
    """Run disturbance scenario; return results with derived 1-x40 output."""
    print("\nRunning disturbance scenario (F, zF, LT, VB step disturbances) …")

    f_idx = model.input_names.index("F")
    zf_idx = model.input_names.index("zF")
    lt_idx = model.input_names.index("LT")
    vb_idx = model.input_names.index("VB")

    U_arr = np.tile(U_NOM, (NT_SIM, 1))
    U_arr[10:, f_idx] += 0.2  # t=10:  F  1.0 → 1.2
    U_arr[100:, f_idx] -= 0.4  # t=100: F  1.2 → 0.8
    U_arr[200:, f_idx] += 0.2  # t=200: F  0.8 → 1.0
    U_arr[300:, zf_idx] += 0.1  # t=300: zF 0.5 → 0.6
    U_arr[400:, zf_idx] -= 0.2  # t=400: zF 0.6 → 0.4
    U_arr[500:, lt_idx] += 0.3  # t=500: LT 2.706 → 3.006
    U_arr[600:, vb_idx] += 0.3  # t=600: VB 3.206 → 3.506
    U_scen = pd.DataFrame(U_arr, columns=model.input_names)

    sim_results_scen = run_simulation(
        t_eval, U_scen, model, param_vals, X_SS, sim_func=sim_func
    )
    sim_results_scen[("Outputs", "1-x40")] = (
        1.0 - sim_results_scen[("Outputs", "x40")]
    )
    return sim_results_scen


# ── Plot functions ─────────────────────────────────────────────────────────
def plot_steps(step_data):
    """One figure per MV: CV deviations (top panels) and MV step (bottom)."""
    print("\nPlotting step responses …")
    for mv_name, dev in step_data.items():
        mv_info = var_info.get(mv_name, {})
        plot_info = {
            "Deviation in compositions": {
                ("Outputs", "x40"): {"color": "C0"},
                ("Outputs", "x0"): {"color": "C1"},
                "ylabel": f"Δ [{var_info['x0']['units']}]",
            },
            "Deviation in product flows": {
                ("Outputs", "D"): {"color": "C0"},
                ("Outputs", "B"): {"color": "C1"},
                "ylabel": f"Δ [{var_info['D']['units']}]",
            },
            f"Step changes in {mv_info.get('name', mv_name)}": {
                ("Inputs", mv_name): {
                    "color": "C2",
                    "drawstyle": "steps-post",
                },
                "ylabel": f"Δ{mv_name} [{mv_info.get('units', '')}]",
            },
        }
        fig, axs = make_tsplots(
            dev, plot_info, time_label="Time [min]", var_info=var_info
        )
        for ax in axs:
            ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / f"dist_model_cola_lv_step_{mv_name}.png", dpi=300
        )
        plt.show()


def plot_scenario(sim_results_scen):
    """Disturbance scenario I/O time-series plot."""
    print("\nPlotting disturbance scenario …")
    plot_info_scen = {
        "Product impurities": {
            ("Outputs", "1-x40"): {"color": "C0"},
            ("Outputs", "x0"): {"color": "C1"},
        },
        "Product flows": {
            ("Outputs", "D"): {"color": "C2"},
            ("Outputs", "B"): {"color": "C3"},
        },
        "Feed rate": {
            ("Inputs", "F"): {"color": "C4", "drawstyle": "steps-post"},
        },
        "Feed composition": {
            ("Inputs", "zF"): {"color": "C5", "drawstyle": "steps-post"},
        },
        "Reflux": {
            ("Inputs", "LT"): {"color": "C6", "drawstyle": "steps-post"},
        },
        "Boilup": {
            ("Inputs", "VB"): {"color": "C7", "drawstyle": "steps-post"},
        },
    }
    fig, axs = make_tsplots(
        sim_results_scen,
        plot_info_scen,
        time_label="Time [min]",
        var_info=var_info,
        figsize=(8, 1 + 1.5 * len(plot_info_scen)),
    )
    axs[0].axhline(
        1.0 - CV_OUTPUT_REFS["x40"], color="k", linewidth=0.5, linestyle="--"
    )
    axs[0].axhline(
        CV_OUTPUT_REFS["x0"], color="k", linewidth=0.5, linestyle="--"
    )
    axs[1].axhline(
        CV_OUTPUT_REFS["D"],
        color="k",
        linewidth=0.5,
        linestyle="--",
        label="steady-state",
    )
    axs[1].axhline(
        CV_OUTPUT_REFS["B"], color="k", linewidth=0.5, linestyle="--"
    )
    axs[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_scenario.png", dpi=300)
    plt.show()


def plot_comp_profiles(sim_results_scen):
    """Tray composition profiles from the scenario simulation."""
    print("\nPlotting composition profiles …")
    x_scen = sim_results_scen["States"].values
    x_comp_scen = x_scen[:, :NT]
    t_arr = t_eval.values

    fig, axs = _make_profile_axes(
        f"Tray compositions (light=blue, heavy=orange) – {PROFILE_TITLE}"
    )
    for plot_row in range(NT):
        tray_idx = NT - 1 - plot_row
        ax = axs[plot_row]
        x_i = x_comp_scen[:, tray_idx]
        ax.fill_between(t_arr, 0, 1 - x_i, color="tab:orange", linewidth=0)
        ax.fill_between(t_arr, 1 - x_i, 1.0, color="tab:blue", linewidth=0)
    _style_profile_axes(axs, ylim=(0.0, 1.0))
    plt.savefig(
        PLOT_DIR / "dist_model_cola_lv_scenario_comp_profiles.png", dpi=150
    )
    plt.show()


def plot_holdup_profiles(sim_results_scen):
    """Tray holdup profiles from the scenario simulation."""
    print("\nPlotting holdup profiles …")
    x_scen = sim_results_scen["States"].values
    M_scen = x_scen[:, NT:]
    t_arr = t_eval.values
    M_lo = M_scen.min()
    M_hi = M_scen.max()
    pad = max(0.005 * (M_hi - M_lo), 1e-4)

    fig, axs = _make_profile_axes(
        f"Tray holdups [{var_info['M0']['units']}] – {PROFILE_TITLE}"
    )
    for plot_row in range(NT):
        tray_idx = NT - 1 - plot_row
        ax = axs[plot_row]
        ax.fill_between(
            t_arr,
            M_lo - pad,
            M_scen[:, tray_idx],
            color="tab:green",
            linewidth=0,
        )
    _style_profile_axes(axs, ylim=(M_lo - pad, M_hi + pad))
    plt.savefig(
        PLOT_DIR / "dist_model_cola_lv_scenario_holdup_profiles.png", dpi=150
    )
    plt.show()


def plot_abstract_steps(step_data, subplot_size=(1.75, 1.75)):
    """Step response matrix in the same compact abstract style as the SS I/O plot.

    Shows the forward step period only (t=0 to RETURN_TIME) for each
    (CV row, MV column) pair.  MVs with negligible response show a flat
    zero line.
    """
    print("\nPlotting abstract step response matrix …")
    n_cvs = len(CV_OUTPUT_NAMES)
    n_mvs = len(MV_NAMES)

    # Labels to match the SS abstract plot (symbol = physical variable, not stage index)
    cv_labels = {
        "x0": {"name": "Bottoms comp.", "symbol": r"$x_B$"},
        "x40": {"name": "Distillate comp.", "symbol": r"$x_D$"},
        "D": {"name": "Distillate flow", "symbol": r"$D$"},
        "B": {"name": "Bottoms flow", "symbol": r"$B$"},
    }

    t_slice = t_eval.iloc[: RETURN_TIME + 1].values

    fig, axs = plt.subplots(
        n_cvs,
        n_mvs,
        figsize=(subplot_size[0] * n_mvs, subplot_size[1] * n_cvs),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    for col, mv_name in enumerate(MV_NAMES):
        mv_info = var_info.get(mv_name, {})

        for row, cv_name in enumerate(CV_OUTPUT_NAMES):
            ax = axs[row, col]

            if mv_name in step_data:
                y = (
                    step_data[mv_name][("Outputs", cv_name)]
                    .iloc[: RETURN_TIME + 1]
                    .values
                )
            else:
                y = np.zeros(RETURN_TIME + 1)

            ax.plot(t_slice, y, color="C0")
            ax.axvline(
                STEP_TIME, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.axhline(
                0.0, color="k", linewidth=0.6, linestyle="--", alpha=0.6
            )
            ax.margins(0.12)
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            ax.grid(False)

            if col == 0:
                cv_info = cv_labels[cv_name]
                ax.set_ylabel(f"{cv_info['name']}\n({cv_info['symbol']})")
            if row == n_cvs - 1:
                mv_symbol = mv_info.get("symbol", mv_name)
                mv_label = mv_info.get("name", mv_name)
                ax.set_xlabel(f"{mv_label}\n({mv_symbol})")

    fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.94])
    fig.suptitle(
        "Column A Model – Step responses (LV configuration)",
    )
    plt.savefig(PLOT_DIR / "dist_model_cola_lv_step_abstract.png", dpi=150)
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate step response and scenario plots for the Column A LV model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "available plots:\n"
            + "".join(
                f"  {name:<18}  {PLOT_DESCRIPTIONS[name]}\n"
                for name in PLOT_DESCRIPTIONS
            )
        ),
    )
    parser.add_argument(
        "plots",
        nargs="*",
        metavar="PLOT",
        help="plot name(s) to generate (default: all); see available plots below",
    )
    args = parser.parse_args()

    unknown = set(args.plots) - set(PLOT_DESCRIPTIONS)
    if unknown:
        parser.error(
            f"unknown plot name(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(PLOT_DESCRIPTIONS)}"
        )

    selected = set(args.plots) if args.plots else set(PLOT_DESCRIPTIONS)

    # ── Build model ──────────────────────────────────────────────────────
    print("Building CasADi simulation function …")
    model = build_cola_lv_ct_model()
    sim_func = make_n_step_simulation_function_from_model(
        model, dt=DT, nT=NT_SIM
    )
    param_vals = make_nominal_lv_param_values()
    print(f"  Model: n={model.n}, nu={model.nu}, ny={model.ny}")
    print(f"  Inputs: {model.input_names}")

    # ── Step responses ────────────────────────────────────────────────────
    if selected & {"steps", "step_abstract"}:
        step_data = compute_step_responses(model, sim_func, param_vals)
        if "steps" in selected:
            plot_steps(step_data)
        if "step_abstract" in selected:
            plot_abstract_steps(step_data)

    # ── Scenario simulation ───────────────────────────────────────────────
    if selected & {"scenario", "comp_profiles", "holdup_profiles"}:
        sim_results_scen = compute_scenario(model, sim_func, param_vals)
        if "scenario" in selected:
            plot_scenario(sim_results_scen)
        if "comp_profiles" in selected:
            plot_comp_profiles(sim_results_scen)
        if "holdup_profiles" in selected:
            plot_holdup_profiles(sim_results_scen)

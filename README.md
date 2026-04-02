# distillation-ctrl

Python implementations of two distillation column process models, together
with utility libraries and a model-predictive control (MPC) simulation.

---

## Overview

This repository contains two independent distillation column models:

| Model | Description | Outputs | Source |
|-------|-------------|---------|--------|
| **Linear** (`dist_model_lin_*`) | MIMO transfer-function model with temperature outputs, discretised via ZOH | Overhead temperature, reflux rate, bottom temperature | M. Grieve (Simulink, 2020) |
| **Nonlinear** (`dist_model_cola_*`) | Skogestad–Morari Column A nonlinear state-space model with composition outputs | 41 stage compositions + 41 molar holdups (82 states) | Skogestad & Morari (1988) |

Both models are implemented in Python using standard scientific libraries
(NumPy, python-control, CasADi).

---

## Attribution

### Linear model

Original Simulink model and transfer-function parameters:

> Michael Grieve, August 2020.  Provided as a Simulink screenshot and
> spreadsheet.

### Nonlinear Column A model

Model equations, variable names, and parameter values follow the
Skogestad–Morari reference implementation:

> Skogestad, S. and Morari, M. (1988). "Understanding the dynamic behavior
> of distillation columns." *Industrial & Engineering Chemistry Research*,
> 27(10), 1848–1862.

MATLAB reference implementation (`colamod.m`):

> Skogestad, S. "colamod.m — Nonlinear model of distillation column A."
> Companion code to *Multivariable Feedback Design* (1996).
> https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html

Python/CasADi implementation by Bill Tubbs, using the
[casadi-models](https://github.com/billtubbs/casadi-models) framework
(`src/dist_model_cola_cas/sys_model.py`).

---

## Directory structure

```
distillation-ctrl/
│
├── dist_model_lin_ct.py        # Linear CT model – step responses
├── dist_model_lin_dt.py        # Linear DT model – discretisation,
│                               #   step responses, saves SS matrices
├── mpc_distillation.py         # Offset-free MPC + EKF simulation
│                               #   (uses the linear DT model)
│
├── dist_model_cola_lv.py       # Nonlinear Column A (LV config) –
│                               #   step responses & scenario simulation
│
├── images/
│   └── Simulink model diagram.png
│
├── plots/                      # Saved plot images (generated at runtime)
│
├── src/
│   ├── dist_model_lin_ctrl/    # Utility library for the linear model
│   │   ├── c2d_utils.py        #   CT→DT discretisation with delays
│   │   ├── delay_ct.py         #   Continuous-time Padé delay
│   │   ├── sim_utils.py        #   MIMO forced-response helper
│   │   ├── ss_utils.py         #   State-space augmentation utilities
│   │   └── variables.py        #   Variable metadata (names, units)
│   │
│   ├── dist_model_lin_dt/      # Saved DT state-space matrices (CSV)
│   │   ├── A.csv, B.csv,
│   │   │   C.csv, D.csv
│   │
│   ├── dist_model_cola_cas/    # Nonlinear Column A CasADi model
│   │   └── sys_model.py        #   build_cola_lv_ct_model,
│   │                           #   build_cola_lv_sim_function, ...
│   │
│   └── dist_model_cola_m/      # Reference MATLAB/Octave implementation
│       ├── colamod.m           #   Original Skogestad nonlinear model
│       ├── cola_lv.m           #   LV closed-loop wrapper
│       └── ...
│
├── tests/
│   ├── test_c2d_utils.py       # Tests for c2d_with_delay
│   └── test_cola_sys_model.py  # Tests for the nonlinear CasADi model
│
├── pyproject.toml
└── requirements.txt
```

---

## Linear model (`dist_model_lin_*`)

### Model description

A 2-input, 3-output MIMO transfer-function model of a distillation column
derived from a Simulink diagram (M. Grieve, 2020).

**Manipulated variables (MVs):**

| Symbol | Description | Units |
|--------|-------------|-------|
| V | Boilup rate | BPH |
| D | Distillate draw | BPH |

**Controlled variables (CVs):**

| Symbol | Description | Units |
|--------|-------------|-------|
| OHt | Overhead (tower top) temperature | °F |
| L | Reflux rate | BPH |
| BmT | Bottom temperature | °F |

DC gain matrix (each entry is the steady-state output change per unit
input change):

```
         V       D
OHt  [ -0.20  +0.30 ]
L    [ +1.00  -1.00 ]
BmT  [ +1.00  +1.20 ]
```

All transfer-function elements are first-order lags with time delays
(dead times range from 1 to 5 min, time constants from 4 to 8 min).

### Scripts

- **`dist_model_lin_ct.py`** – Builds the continuous-time MIMO transfer
  function, plots the 3×2 step-response grid, and simulates a two-step
  input scenario (V at t=5 min, D at t=30 min).

- **`dist_model_lin_dt.py`** – Discretises each SISO channel at Ts=1 min
  via ZOH and appends exact sample delays (z-shift), assembles the
  discrete-time MIMO system, plots the step-response grid, runs the same
  two-step scenario in discrete time, and saves the minimal state-space
  matrices (A, B, C, D) to `src/dist_model_lin_dt/*.csv` for use by the
  MPC script.

### MPC simulation (`mpc_distillation.py`)

An offset-free Model Predictive Controller (MPC) combined with an Extended
Kalman Filter (EKF) disturbance estimator, implemented using
[do-mpc](https://www.do-mpc.com/).

#### Formulation

Variables are expressed as deviations from a nominal operating point,
scaled so that typical moves span roughly ±100 units.

The MPC uses the **original** (non-augmented) linear DT plant model
(nx=15, nu=2, ny=3) loaded from `src/dist_model_lin_dt/*.csv`.  To achieve
zero steady-state offset in the presence of sustained input disturbances,
an *offset-free* formulation is used (Pannocchia & Rawlings, 2003): the EKF
estimates an augmented state `xa = [x; d_u]` where `d_u` (nu=2 integrating
disturbance states, one per input channel) captures unmeasured input
disturbances.  The disturbance estimate `d_hat` is passed to the MPC as a
time-varying parameter and fed forward over the prediction horizon:

```
x(k+j+1|k) = A x(k+j|k) + B u(k+j|k) + B d_hat(k)
```

This keeps the MPC optimisation problem at its original size (nx=15) while
eliminating steady-state offset.

#### Simulation events

| Time | Event |
|------|-------|
| t = 30 min | Setpoint change on OHt and BmT |
| t = 70 min | Unmeasured step disturbance on V |
| t = 100 min | Unmeasured step disturbance on D |

#### Tuning

| Parameter | Value |
|-----------|-------|
| Prediction horizon N | 20 steps |
| Output weights Q (OHt, L, BmT) | 10, 0, 10 |
| Input rate weights ΔU (V, D) | 1, 1 |
| Input hard constraints | ±100 units |
| Process noise std (plant states) | 0.01 |
| Process noise std (disturbance states) | 1.0 |

The output plot is saved to `plots/mpc_dist_sim.png`.

---

## Nonlinear Column A model (`dist_model_cola_*`)

### Model description

A first-principles nonlinear model of a binary distillation column
(Skogestad–Morari "Column A"), implemented in Python using CasADi symbolic
mathematics.  The column has NT=41 theoretical stages (reboiler + 39 trays
+ total condenser) with the feed at stage 21 from the bottom.

The model is provided in two configurations:

- **Open-loop** (`build_cola_ct_model`) – All seven inputs (LT, VB, D, B,
  F, zF, qF) are external.  Reboiler and condenser holdups are
  open-loop integrating (unstable).

- **LV closed-loop** (`build_cola_lv_ct_model`) – D and B are computed
  internally by proportional (P) level controllers from the condenser and
  reboiler holdup states, leaving five external inputs.  This is the
  configuration used in `dist_model_cola_lv.py`.

**Manipulated variables (MVs, LV configuration):**

| Symbol | Description | Units | Nominal |
|--------|-------------|-------|---------|
| LT | Reflux flow | kmol/min | 2.706 |
| VB | Boilup flow | kmol/min | 3.206 |
| F | Feed flow rate | kmol/min | 1.0 |
| zF | Feed composition | mol frac | 0.5 |
| qF | Feed liquid fraction | – | 1.0 |

**Controlled / monitored variables (82 states):**

| Symbol | Description |
|--------|-------------|
| x[0] = xB | Reboiler (bottoms) composition — SS ≈ 0.010 mol frac |
| x[1..39] | Tray compositions (bottom to top) |
| x[40] = xD | Condenser (distillate) composition — SS ≈ 0.990 mol frac |
| x[41..81] | Molar holdups on each stage — SS = 0.5 kmol |

### Script

**`dist_model_cola_lv.py`** – Builds the LV closed-loop Column A model
using `build_cola_lv_sim_function` (CasADi stiff ODE integrator, dt=1 min).
Starting from the known steady state, it:

1. Runs a step response for each MV (LT, VB, F, zF, qF) and plots the
   two key product composition deviations (ΔxB, ΔxD) and the MV step in a
   three-subplot figure.  Responses with maximum absolute deviation below
   1×10⁻⁵ are skipped.  Figures are saved to `plots/`.

2. Runs a 200-min scenario in which LT is stepped at t=30 min and VB is
   stepped at t=120 min, plotting xB and xD (deviations) and both MV
   trajectories.  Saved to `plots/dist_model_cola_lv_scenario.png`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Running the scripts

```bash
python dist_model_lin_ct.py        # linear CT step responses
python dist_model_lin_dt.py        # linear DT step responses + save matrices
python mpc_distillation.py         # MPC closed-loop simulation
python dist_model_cola_lv.py       # nonlinear LV step responses + scenario
```

Tests:

```bash
pytest tests/
```

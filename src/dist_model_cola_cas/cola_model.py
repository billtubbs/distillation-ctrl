"""
Nonlinear distillation column model (Column A) as a CasADi continuous-time
state-space model — open-loop (all seven inputs external).

Attribution
-----------
Original model equations:
    Skogestad, S. and Morari, M. (1988). "Understanding the dynamic behavior
    of distillation columns." Industrial & Engineering Chemistry Research,
    27(10), 1848-1862.

Column A data and MATLAB reference implementation:
    Skogestad, S. "colamod.m — Nonlinear model of distillation column A."
    Available at: https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html
    (Companion code to Skogestad & Postlethwaite, "Multivariable Feedback
    Design", 1996.)

This Python/CasADi implementation replicates colamod.m using the
casadi-models framework (https://github.com/billtubbs/casadi-models). The
model equations, variable names, and parameter values follow colamod.m
exactly. Deviations from the MATLAB source:

  * All arrays are 0-indexed (MATLAB uses 1-based indexing).
  * ``lambda`` is renamed ``lam`` (reserved word in Python).
  * The internal VLE vapour composition array is named ``y_vle`` to avoid
    collision with the casadi-models output variable convention (``y``).
  * The CasADi state vector is named ``x`` (casadi-models convention);
    the composition slice of the state is named ``x_comp`` to distinguish
    it from the full state.

Variable naming otherwise follows the Skogestad notation throughout. These
are established symbols in the distillation / process control literature
and map directly to the source equations.

Model description
-----------------
Binary distillation column with NT - 1 = 40 theoretical stages (trays) plus:

  * Stage 0     : reboiler (equilibrium stage)
  * Stage NT-1  : total condenser (no equilibrium)
  * Feed at stage NF-1 = 20 (Python 0-indexed from bottom)

Model assumptions: binary separation; constant relative volatility; no vapour
holdup; one feed and two products; constant molar flows; total condenser.

States  n = 82
--------------
x_comp[0  .. NT-1] : liquid mole fraction of light component A on each stage
                     (x_comp[0] = reboiler/bottoms, x_comp[NT-1] = condenser)
                     Units: dimensionless (mole fraction)

M[0  .. NT-1]      : molar liquid holdup on each stage
                     Units: kmol

State vector ordering: compositions first, then holdups.
  x[0]   = x_comp[0]  (reboiler composition)
  x[40]  = x_comp[40] (condenser composition)
  x[41]  = M[0]       (reboiler holdup)
  x[81]  = M[40]      (condenser holdup)

Inputs  nu = 7
--------------
u[0]  LT  : reflux flow (top liquid)                [kmol/min]
u[1]  VB  : boilup flow (bottom vapour)              [kmol/min]
u[2]  D   : distillate (top product) flow            [kmol/min]
u[3]  B   : bottoms (bottom product) flow            [kmol/min]
u[4]  F   : feed flow rate                           [kmol/min]
u[5]  zF  : feed composition (mole fraction of light component A)
u[6]  qF  : feed liquid fraction (q-value; 1 = saturated liquid feed)

Outputs  ny = 82
----------------
y[0..81] : all 82 states (compositions then holdups), i.e. H = I.

Parameters
----------
alpha : relative volatility                          (Column A: 1.5)
taul  : liquid flow time constant                    [min]   (0.063)
F0    : nominal feed rate                            [kmol/min]  (1.0)
qF0   : nominal feed liquid fraction                 (1.0)
L0    : nominal reflux / liquid flow above feed      [kmol/min]  (2.70629)
L0b   : nominal liquid flow below feed               [kmol/min]  (3.70629)
lam   : K2-effect coefficient (vapour effect on
        liquid flow; 0 = no K2 effect)               (0.0)
V0    : nominal boilup flow                          [kmol/min]  (3.20629)
V0t   : nominal vapour flow above feed               [kmol/min]  (3.20629)
M0    : nominal molar holdup on each stage           [kmol]
        (vector, length NT=41; all 0.5 for Column A)

Column constants (structural — not symbolic parameters)
-------------------------------------------------------
NT = 41   : total number of stages (reboiler + 39 trays + condenser)
NF = 21   : feed stage, 1-indexed from bottom (Python 0-indexed: NF-1 = 20)
"""

import numpy as np
import casadi as cas

from cas_models.continuous_time.models import StateSpaceModelCT

# ---------------------------------------------------------------------------
# Column A structural constants
# ---------------------------------------------------------------------------
NT = 41  # total stages: reboiler (0) + 39 trays + condenser (40)
NF = 21  # feed stage, 1-indexed (Python 0-indexed: NF-1 = 20)

# ---------------------------------------------------------------------------
# Default nominal parameter values for Column A
# ---------------------------------------------------------------------------
ALPHA_DEFAULT = 1.5
TAUL_DEFAULT = 0.063
F0_DEFAULT = 1.0
QF0_DEFAULT = 1.0
L0_DEFAULT = 2.70629
L0B_DEFAULT = L0_DEFAULT + QF0_DEFAULT * F0_DEFAULT  # 3.70629
LAM_DEFAULT = 0.0
V0_DEFAULT = 3.20629
V0T_DEFAULT = V0_DEFAULT + (1 - QF0_DEFAULT) * F0_DEFAULT  # 3.20629
M0_DEFAULT = 0.5  # same nominal holdup on all NT stages


def make_nominal_param_values() -> dict:
    """Return Column A nominal parameter values as a dict of CasADi DM.

    Returns
    -------
    dict
        Keys match the ``params`` dict of ``build_cola_ct_model()``.
    """
    return {
        "alpha": cas.DM(ALPHA_DEFAULT),
        "taul": cas.DM(TAUL_DEFAULT),
        "F0": cas.DM(F0_DEFAULT),
        "qF0": cas.DM(QF0_DEFAULT),
        "L0": cas.DM(L0_DEFAULT),
        "L0b": cas.DM(L0B_DEFAULT),
        "lam": cas.DM(LAM_DEFAULT),
        "V0": cas.DM(V0_DEFAULT),
        "V0t": cas.DM(V0T_DEFAULT),
        "M0": cas.DM(M0_DEFAULT * np.ones(NT)),
    }


def _build_cola_rhs(
    x_comp, M, LT, VB, D, B, F, zF, qF, alpha, taul, L0, L0b, lam, V0, V0t, M0
):
    """Build the symbolic RHS for Column A from individual symbolic arguments.

    Shared by both the open-loop and LV closed-loop model builders.

    Parameters
    ----------
    x_comp : cas.SX  shape (NT,)  — liquid compositions
    M      : cas.SX  shape (NT,)  — molar holdups
    LT, VB, D, B, F, zF, qF : cas.SX scalars — column inputs
    alpha, taul, L0, L0b, lam, V0, V0t : cas.SX scalars — column parameters
    M0     : cas.SX  shape (NT,)  — nominal holdups

    Returns
    -------
    cas.SX  shape (2*NT,)
        RHS vector [dxdt; dMdt].
    """
    # ---- Vapour-liquid equilibria (stages 0..NT-2) ---------------------
    x_c = x_comp[: NT - 1]
    y_vle = alpha * x_c / (1 + (alpha - 1) * x_c)

    # ---- Vapour flows (NT-1 values, stages 0..NT-2) --------------------
    #   Below feed (0..NF-2): VB;  at/above feed (NF-1..NT-2): VB + (1-qF)*F
    V_flow = cas.vertcat(
        cas.repmat(VB, NF - 1, 1),
        cas.repmat(VB + (1 - qF) * F, NT - NF, 1),
    )

    # ---- Liquid flows (NT values, stages 0..NT-1) ----------------------
    #   index 0        : placeholder (reboiler has no liquid inflow from below)
    #   index 1..NF-1  : below-feed stages, linearised around L0b
    #   index NF..NT-2 : above-feed stages, linearised around L0
    #   index NT-1     : condenser reflux = LT
    L_below = L0b + (M[1:NF] - M0[1:NF]) / taul + lam * (V_flow[: NF - 1] - V0)
    L_above = (
        L0
        + (M[NF : NT - 1] - M0[NF : NT - 1]) / taul
        + lam * (V_flow[NF - 1 : NT - 2] - V0t)
    )
    L_flow = cas.vertcat(cas.SX(0), L_below, L_above, LT)

    # ---- Material balances: total holdup and component holdup ----------
    # Vectorised over internal stages (i = 1..NT-2)
    dMdt_int = (
        L_flow[2:NT]
        - L_flow[1 : NT - 1]
        + V_flow[: NT - 2]
        - V_flow[1 : NT - 1]
    )
    dMxdt_int = (
        L_flow[2:NT] * x_comp[2:NT]
        - L_flow[1 : NT - 1] * x_comp[1 : NT - 1]
        + V_flow[: NT - 2] * y_vle[: NT - 2]
        - V_flow[1 : NT - 1] * y_vle[1 : NT - 1]
    )

    # Feed correction at stage NF-1 (position NF-2 within the internal array)
    feed_dM = cas.vertcat(
        cas.SX.zeros(NF - 2, 1), F, cas.SX.zeros(NT - 1 - NF, 1)
    )
    feed_dMx = cas.vertcat(
        cas.SX.zeros(NF - 2, 1), F * zF, cas.SX.zeros(NT - 1 - NF, 1)
    )

    # Assemble full dMdt and dMxdt (NT elements each)
    dMdt = cas.vertcat(
        L_flow[1] - V_flow[0] - B,  # reboiler (i=0)
        dMdt_int + feed_dM,  # internal (i=1..NT-2)
        V_flow[NT - 2] - LT - D,  # condenser (i=NT-1)
    )
    dMxdt = cas.vertcat(
        L_flow[1] * x_comp[1]
        - V_flow[0] * y_vle[0]  # reboiler
        - B * x_comp[0],
        dMxdt_int + feed_dMx,  # internal
        V_flow[NT - 2] * y_vle[NT - 2]  # condenser
        - LT * x_comp[NT - 1]
        - D * x_comp[NT - 1],
    )

    # ---- Composition derivatives: dx/dt = (d(Mx)/dt - x·dM/dt) / M ---
    dxdt = (dMxdt - x_comp * dMdt) / M

    return cas.vertcat(dxdt, dMdt)


def build_cola_ct_model(name: str = "cola") -> StateSpaceModelCT:
    """Build the Skogestad-Morari Column A nonlinear distillation model
    as a CasADi continuous-time state-space model (open-loop).

    All seven column inputs (including D and B) are external. This model
    is open-loop unstable: without level control the reboiler and condenser
    holdups integrate freely.

    Parameters
    ----------
    name : str
        Name tag for the returned model object.

    Returns
    -------
    StateSpaceModelCT
        Model with n=82 states, nu=7 inputs, ny=82 outputs (H = I).

    Examples
    --------
    >>> model = build_cola_ct_model()
    >>> model.n, model.nu, model.ny
    (82, 7, 82)
    """
    n = 2 * NT  # 82 states
    nu = 7
    ny = n

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    alpha = cas.SX.sym("alpha")
    taul = cas.SX.sym("taul")
    F0 = cas.SX.sym("F0")
    qF0 = cas.SX.sym("qF0")
    L0 = cas.SX.sym("L0")
    L0b = cas.SX.sym("L0b")
    lam = cas.SX.sym("lam")
    V0 = cas.SX.sym("V0")
    V0t = cas.SX.sym("V0t")
    M0 = cas.SX.sym("M0", NT)

    params = {
        "alpha": alpha,
        "taul": taul,
        "F0": F0,
        "qF0": qF0,
        "L0": L0,
        "L0b": L0b,
        "lam": lam,
        "V0": V0,
        "V0t": V0t,
        "M0": M0,
    }

    x_comp = x[:NT]
    M = x[NT:]
    LT, VB, D, B, F, zF, qF = (u[i] for i in range(7))

    rhs = _build_cola_rhs(
        x_comp,
        M,
        LT,
        VB,
        D,
        B,
        F,
        zF,
        qF,
        alpha,
        taul,
        L0,
        L0b,
        lam,
        V0,
        V0t,
        M0,
    )

    f = cas.Function(
        "f",
        [t, x, u, *params.values()],
        [rhs],
        ["t", "x", "u", *params.keys()],
        ["rhs"],
    )
    h = cas.Function(
        "h",
        [t, x, u, *params.values()],
        [x],
        ["t", "x", "u", *params.keys()],
        ["y"],
    )

    state_names = [f"x{i}" for i in range(NT)] + [f"M{i}" for i in range(NT)]
    input_names = ["LT", "VB", "D", "B", "F", "zF", "qF"]
    output_names = state_names[:]

    return StateSpaceModelCT(
        f,
        h,
        n=n,
        nu=nu,
        ny=ny,
        params=params,
        name=name,
        state_names=state_names,
        input_names=input_names,
        output_names=output_names,
    )

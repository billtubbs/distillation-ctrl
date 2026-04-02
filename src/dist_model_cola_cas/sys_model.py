"""
Nonlinear distillation column model (Column A) as a CasADi continuous-time
state-space model.

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

Inputs  nu = 7  (open-loop model, build_cola_ct_model)
------------------------------------------------------
u[0]  LT  : reflux flow (top liquid)                [kmol/min]
u[1]  VB  : boilup flow (bottom vapour)              [kmol/min]
u[2]  D   : distillate (top product) flow            [kmol/min]
u[3]  B   : bottoms (bottom product) flow            [kmol/min]
u[4]  F   : feed flow rate                           [kmol/min]
u[5]  zF  : feed composition (mole fraction of light component A)
u[6]  qF  : feed liquid fraction (q-value; 1 = saturated liquid feed)

Inputs  nu = 5  (LV closed-loop model, build_cola_lv_ct_model)
--------------------------------------------------------------
u[0]  LT  : reflux flow (top liquid)                [kmol/min]
u[1]  VB  : boilup flow (bottom vapour)              [kmol/min]
u[2]  F   : feed flow rate                           [kmol/min]
u[3]  zF  : feed composition (mole fraction of light component A)
u[4]  qF  : feed liquid fraction (q-value; 1 = saturated liquid feed)
D and B are computed internally by P-controllers from the holdup states.

Outputs  ny = 82
----------------
y[0..81] : all 82 states (compositions then holdups), i.e. H = I.

Parameters  (open-loop model)
------------------------------
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

Additional parameters  (LV closed-loop model)
---------------------------------------------
KcD   : P-controller gain for condenser level        (10.0)
KcB   : P-controller gain for reboiler level         (10.0)
Ds    : nominal distillate flow setpoint             [kmol/min]  (0.5)
Bs    : nominal bottoms flow setpoint                [kmol/min]  (0.5)

Column constants (structural — not symbolic parameters)
-------------------------------------------------------
NT = 41   : total number of stages (reboiler + 39 trays + condenser)
NF = 21   : feed stage, 1-indexed from bottom (Python 0-indexed: NF-1 = 20)
"""

import numpy as np
import casadi as cas

from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator_fixed_dt,
)
from cas_models.discrete_time.simulate import make_n_step_simulation_function

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

# Default P-controller gains for the LV closed-loop model
KCD_DEFAULT = 10.0  # condenser level controller gain
KCB_DEFAULT = 10.0  # reboiler level controller gain
DS_DEFAULT = 0.5  # nominal distillate flow setpoint [kmol/min]
BS_DEFAULT = 0.5  # nominal bottoms flow setpoint    [kmol/min]

# ---------------------------------------------------------------------------
# Steady-state reference values for Column A (LV configuration)
# Computed by Octave/cola_lv.m: ode15s from x0=0.5*ones(82), t=2000 min,
# nominal inputs LT=2.70629, VB=3.20629, F=1, zF=0.5, qF=1.
# norm(xprime) at t=2000: 2.92e-9
# ---------------------------------------------------------------------------
X_SS_COMP = np.array(
    [
        0.01000004946,
        0.01426098197,
        0.01972368383,
        0.02669338974,
        0.03553129,
        0.04665111403,
        0.06050561756,
        0.07755811532,
        0.09823455073,
        0.1228542705,
        0.1515438077,
        0.1841475776,
        0.2201598312,
        0.2587075543,
        0.2986072091,
        0.3384967484,
        0.3770154943,
        0.4129836365,
        0.4455332825,
        0.4741640485,
        0.4987250205,
        0.5264947888,
        0.5577638685,
        0.5921605039,
        0.6290390119,
        0.6675065276,
        0.7064981526,
        0.7448898952,
        0.7816253042,
        0.8158264802,
        0.846866089,
        0.8743908164,
        0.8983013842,
        0.9187037365,
        0.9358482989,
        0.9500710134,
        0.9617443802,
        0.971241585,
        0.9789132562,
        0.9850745778,
        0.989999967,
    ]
)
X_SS_HOLDUPS = 0.5 * np.ones(NT)
X_SS = np.concatenate([X_SS_COMP, X_SS_HOLDUPS])


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


def make_nominal_lv_param_values() -> dict:
    """Return nominal parameter values for the LV closed-loop model.

    Extends ``make_nominal_param_values()`` with controller gains.

    Returns
    -------
    dict
        Keys match the ``params`` dict of ``build_cola_lv_ct_model()``.
    """
    p = make_nominal_param_values()
    p["KcD"] = cas.DM(KCD_DEFAULT)
    p["KcB"] = cas.DM(KCB_DEFAULT)
    p["Ds"] = cas.DM(DS_DEFAULT)
    p["Bs"] = cas.DM(BS_DEFAULT)
    return p


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
    x_c = x_comp[:NT - 1]
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
    L_below = L0b + (M[1:NF]      - M0[1:NF])      / taul + lam * (V_flow[:NF - 1]        - V0)
    L_above = L0  + (M[NF:NT - 1] - M0[NF:NT - 1]) / taul + lam * (V_flow[NF - 1:NT - 2] - V0t)
    L_flow = cas.vertcat(cas.SX(0), L_below, L_above, LT)

    # ---- Material balances: total holdup and component holdup ----------
    # Vectorised over internal stages (i = 1..NT-2)
    dMdt_int = L_flow[2:NT] - L_flow[1:NT - 1] + V_flow[:NT - 2] - V_flow[1:NT - 1]
    dMxdt_int = (
        L_flow[2:NT] * x_comp[2:NT]
        - L_flow[1:NT - 1] * x_comp[1:NT - 1]
        + V_flow[:NT - 2] * y_vle[:NT - 2]
        - V_flow[1:NT - 1] * y_vle[1:NT - 1]
    )

    # Feed correction at stage NF-1 (position NF-2 within the internal array)
    feed_dM  = cas.vertcat(cas.SX.zeros(NF - 2, 1), F,      cas.SX.zeros(NT - 1 - NF, 1))
    feed_dMx = cas.vertcat(cas.SX.zeros(NF - 2, 1), F * zF, cas.SX.zeros(NT - 1 - NF, 1))

    # Assemble full dMdt and dMxdt (NT elements each)
    dMdt = cas.vertcat(
        L_flow[1] - V_flow[0] - B,                                                    # reboiler (i=0)
        dMdt_int + feed_dM,                                                            # internal (i=1..NT-2)
        V_flow[NT - 2] - LT - D,                                                      # condenser (i=NT-1)
    )
    dMxdt = cas.vertcat(
        L_flow[1] * x_comp[1] - V_flow[0] * y_vle[0] - B * x_comp[0],               # reboiler
        dMxdt_int + feed_dMx,                                                          # internal
        V_flow[NT - 2] * y_vle[NT - 2] - LT * x_comp[NT - 1] - D * x_comp[NT - 1],  # condenser
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


def build_cola_lv_ct_model(name: str = "cola_lv") -> StateSpaceModelCT:
    """Build the Column A model with built-in P-controllers for level
    stabilisation (replicates cola_lv.m).

    D and B are no longer external inputs; instead they are computed
    internally from the reboiler and condenser holdup states:

        D = Ds + KcD * (M[NT-1] - M0[NT-1])   (condenser level control)
        B = Bs + KcB * (M[0]    - M0[0]   )   (reboiler  level control)

    This closed-loop structure makes the holdup states self-regulating and
    allows the column to reach a well-defined steady state from arbitrary
    initial conditions.

    Parameters
    ----------
    name : str
        Name tag for the returned model object.

    Returns
    -------
    StateSpaceModelCT
        Model with n=82 states, nu=5 inputs, ny=82 outputs (H = I).
        Symbolic parameters include column params plus KcD, KcB, Ds, Bs.

    Examples
    --------
    >>> model = build_cola_lv_ct_model()
    >>> model.n, model.nu, model.ny
    (82, 5, 82)
    >>> model.input_names
    ['LT', 'VB', 'F', 'zF', 'qF']
    """
    n = 2 * NT
    nu = 5  # [LT, VB, F, zF, qF]
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
    KcD = cas.SX.sym("KcD")
    KcB = cas.SX.sym("KcB")
    Ds = cas.SX.sym("Ds")
    Bs = cas.SX.sym("Bs")

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
        "KcD": KcD,
        "KcB": KcB,
        "Ds": Ds,
        "Bs": Bs,
    }

    x_comp = x[:NT]
    M = x[NT:]
    LT, VB, F, zF, qF = (u[i] for i in range(5))

    # P-controllers: D and B computed from holdup states (replicates cola_lv.m)
    D = Ds + KcD * (M[NT - 1] - M0[NT - 1])
    B = Bs + KcB * (M[0] - M0[0])

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
    input_names = ["LT", "VB", "F", "zF", "qF"]
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


def make_nominal_sim_param_values() -> dict:
    """Return parameter values for calling a simulation function built by
    ``build_cola_lv_sim_function``.

    The simulation function has M0 baked in at its nominal value (see note
    in ``build_cola_lv_sim_function``), so M0 is absent from this dict.

    Returns
    -------
    dict
        All scalar parameters for the LV closed-loop model at Column A
        nominal values, excluding M0.
    """
    p = make_nominal_lv_param_values()
    del p["M0"]
    return p


def build_cola_lv_sim_function(
    dt: float, nT: int, m0_val=None, name: str = "cola_lv_sim"
):
    """Build an nT-step simulation function for the LV closed-loop Column A
    model.

    Each time step uses CasADi's cvodes stiff integrator to advance the
    state by ``dt`` minutes, making this suitable for the stiff composition
    and hydraulic dynamics of the column.

    .. note::
        The vector-valued parameter M0 (nominal stage holdups, shape NT=41)
        is substituted with a fixed numeric value at build time.  This is
        required because the ``cas_models`` integrator backend only supports
        scalar symbolic parameters.  Use ``make_nominal_sim_param_values()``
        to obtain the matching scalar-only parameter dict for calling the
        returned function.

    Parameters
    ----------
    dt : float
        Integration step size [min].
    nT : int
        Number of time steps.
    m0_val : float or array-like, optional
        Nominal holdup value(s) to bake into the model.  A scalar is
        broadcast to all NT stages.  Defaults to ``M0_DEFAULT`` (0.5).
    name : str
        Name tag for the compiled CasADi simulation function.

    Returns
    -------
    sim_func : cas.Function
        CasADi Function with signature::

            (t_eval, U, x0, *scalar_params) -> (X, Y)

        where ``t_eval`` has shape ``(nT+1,)``, ``U`` has shape ``(nT, 5)``,
        ``X`` has shape ``(nT+1, 82)``, ``Y`` has shape ``(nT+1, 82)``.
        ``scalar_params`` are all LV model params except M0 (13 values).
        Use ``make_nominal_sim_param_values()`` to get nominal values.
    model : StateSpaceModelCT
        The underlying closed-loop model (full symbolic params including M0).
    scalar_params : dict
        The symbolic parameter dict used inside ``sim_func`` (no M0).

    Examples
    --------
    >>> sim_func, model, params = build_cola_lv_sim_function(dt=1.0, nT=100)
    """
    model = build_cola_lv_ct_model()
    n, nu, ny = model.n, model.nu, model.ny

    if m0_val is None:
        m0_val = M0_DEFAULT * np.ones(NT)
    elif np.isscalar(m0_val):
        m0_val = m0_val * np.ones(NT)
    m0_dm = cas.DM(m0_val)

    # Build scalar-only symbolic variables for the baked function.
    # M0 is substituted with the numeric m0_dm; all other params stay symbolic.
    t_s = cas.SX.sym("t")
    x_s = cas.SX.sym("x", n)
    u_s = cas.SX.sym("u", nu)

    scalar_params = {k: cas.SX.sym(k) for k in model.params if k != "M0"}

    param_args = [
        m0_dm if k == "M0" else scalar_params[k] for k in model.params
    ]

    f_baked = cas.Function(
        "f",
        [t_s, x_s, u_s, *scalar_params.values()],
        [model.f(t_s, x_s, u_s, *param_args)],
        ["t", "x", "u", *scalar_params.keys()],
        ["rhs"],
    )

    h_baked = cas.Function(
        "h",
        [t_s, x_s, u_s, *scalar_params.values()],
        [x_s],
        ["t", "x", "u", *scalar_params.keys()],
        ["y"],
    )

    F_step = make_sim_step_function_integrator_fixed_dt(
        f_baked,
        n,
        nu,
        dt,
        params=scalar_params,
        name="F",
    )

    sim_func = make_n_step_simulation_function(
        F_step,
        h_baked,
        n,
        nu,
        ny,
        nT,
        params=scalar_params,
        name=name,
    )

    return sim_func, model, scalar_params

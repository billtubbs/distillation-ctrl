"""
Column A distillation model in LV configuration as a CasADi continuous-time
state-space model.

In the LV configuration, condenser and reboiler levels are stabilised by
internal P-controllers that manipulate the distillate flow D and bottoms
flow B respectively.  The five remaining inputs (LT, VB, F, zF, qF) are
external.  This replicates the MATLAB reference cola_lv.m.

Builds on the open-loop Column A model in cola_model.py.

Inputs  nu = 5
--------------
u[0]  LT  : reflux flow (top liquid)                [kmol/min]
u[1]  VB  : boilup flow (bottom vapour)              [kmol/min]
u[2]  F   : feed flow rate                           [kmol/min]
u[3]  zF  : feed composition (mole fraction of light component A)
u[4]  qF  : feed liquid fraction (q-value; 1 = saturated liquid feed)

D and B are computed internally:
    D = Ds + KcD * (M[NT-1] - M0[NT-1])   (condenser level control)
    B = Bs + KcB * (M[0]    - M0[0]   )   (reboiler  level control)

Additional parameters  (beyond the open-loop column parameters)
---------------------------------------------------------------
KcD   : P-controller gain for condenser level        [1/min]  (10.0)
KcB   : P-controller gain for reboiler level         [1/min]  (10.0)
Ds    : nominal distillate flow setpoint             [kmol/min]  (0.5)
Bs    : nominal bottoms flow setpoint                [kmol/min]  (0.5)

Steady-state reference (LV configuration)
------------------------------------------
X_SS is the Column A steady state computed by Octave/cola_lv.m using ode15s
from x0=0.5*ones(82) over 2000 min at nominal inputs
(LT=2.70629, VB=3.20629, F=1, zF=0.5, qF=1).
norm(xprime) at t=2000: 2.92e-9.
"""

import numpy as np
import casadi as cas

from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator_fixed_dt,
)
from cas_models.discrete_time.simulate import make_n_step_simulation_function

from dist_model_cola_cas.cola_model import (
    NT,
    M0_DEFAULT,
    make_nominal_param_values,
    _build_cola_rhs,
)

# ---------------------------------------------------------------------------
# Default P-controller and setpoint values for the LV model
# ---------------------------------------------------------------------------
KCD_DEFAULT = 10.0  # condenser level controller gain [1/min]
KCB_DEFAULT = 10.0  # reboiler level controller gain  [1/min]
DS_DEFAULT = 0.5    # nominal distillate flow setpoint [kmol/min]
BS_DEFAULT = 0.5    # nominal bottoms flow setpoint    [kmol/min]

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
        x_comp, M, LT, VB, D, B, F, zF, qF,
        alpha, taul, L0, L0b, lam, V0, V0t, M0,
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

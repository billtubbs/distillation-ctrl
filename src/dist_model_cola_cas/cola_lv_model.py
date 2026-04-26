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
    make_n_step_simulation_function_from_model,
)

from dist_model_cola_cas.cola_model import (
    NT,
    make_nominal_param_values,
    _build_cola_rhs,
)

# ---------------------------------------------------------------------------
# Default P-controller and setpoint values for the LV model
# ---------------------------------------------------------------------------
KCD_DEFAULT = 10.0  # condenser level controller gain [1/min]
KCB_DEFAULT = 10.0  # reboiler level controller gain  [1/min]
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
        Model with n=82 states, nu=5 inputs, ny=84 outputs (82 states + D, B).
        Symbolic parameters include column params plus KcD, KcB, Ds, Bs.

    Examples
    --------
    >>> model = build_cola_lv_ct_model()
    >>> model.n, model.nu, model.ny
    (82, 5, 84)
    >>> model.input_names
    ['LT', 'VB', 'F', 'zF', 'qF']
    >>> model.output_names[-2:]
    ['D', 'B']
    """
    n = 2 * NT
    nu = 5  # [LT, VB, F, zF, qF]
    ny = n + 2  # 82 states + D + B

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
        [cas.vertcat(x, D, B)],
        ["t", "x", "u", *params.keys()],
        ["y"],
    )

    state_names = [f"x{i}" for i in range(NT)] + [f"M{i}" for i in range(NT)]
    input_names = ["LT", "VB", "F", "zF", "qF"]
    output_names = state_names[:] + ["D", "B"]

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


def build_cola_lv_sim_function(dt: float, nT: int, name: str = "cola_lv_sim"):
    """Build an nT-step simulation function for the LV closed-loop Column A
    model.

    Each time step uses CasADi's cvodes stiff integrator to advance the
    state by ``dt`` minutes, making this suitable for the stiff composition
    and hydraulic dynamics of the column.

    Parameters
    ----------
    dt : float
        Integration step size [min].
    nT : int
        Number of time steps.
    name : str
        Name tag for the compiled CasADi simulation function.

    Returns
    -------
    sim_func : cas.Function
        CasADi Function with signature::

            (t_eval, U, x0, *params) -> (X, Y)

        where ``t_eval`` has shape ``(nT+1,)``, ``U`` has shape ``(nT, 5)``,
        ``X`` has shape ``(nT+1, 82)``, ``Y`` has shape ``(nT+1, 84)``.
        Use ``make_nominal_lv_param_values()`` to get nominal parameter values.
    model : StateSpaceModelCT
        The underlying closed-loop model.

    Examples
    --------
    >>> sim_func, model = build_cola_lv_sim_function(dt=1.0, nT=100)
    """
    model = build_cola_lv_ct_model()
    sim_func = make_n_step_simulation_function_from_model(
        model, dt=dt, nT=nT, name=name
    )
    return sim_func, model

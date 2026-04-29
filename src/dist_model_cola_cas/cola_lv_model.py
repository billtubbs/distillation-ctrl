"""
Column A distillation model in LV configuration as a CasADi continuous-time
state-space model.

In the LV configuration, condenser and reboiler levels are stabilised by
internal P-controllers that manipulate the distillate flow D and bottoms
flow B respectively.  The five remaining inputs (LT, VB, F, zF, qF) are
external.  This replicates the MATLAB reference cola_lv.m.

Builds on the open-loop Column A model in cola_model.py.

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

Outputs  ny = 84
----------------
y[0..81] : all 82 states (compositions then holdups), i.e. H = I for
           the first 82 outputs.
y[82]  D : distillate (top product) flow             [kmol/min]
y[83]  B : bottoms (bottom product) flow             [kmol/min]

Monitored outputs (CVs)
-----------------------
y[0]   x[0]  : bottoms composition    [mol/mol]  SS ≈ 0.01
y[40]  x[40] : distillate composition [mol/mol]  SS ≈ 0.99
y[82]  D     : distillate flow        [kmol/min] SS = 0.5
y[83]  B     : bottoms flow           [kmol/min] SS = 0.5

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
from cas_models.param_utils import make_symbolic_vars_from_kwargs

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


def build_cola_lv_ct_model(
    name: str = "cola_lv",
    *,
    alpha=None,
    taul=None,
    F0=None,
    qF0=None,
    L0=None,
    L0b=None,
    lam=None,
    V0=None,
    V0t=None,
    M0=(NT, 1),
    KcD=None,
    KcB=None,
    Ds=None,
    Bs=None,
) -> StateSpaceModelCT:
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
    alpha, taul, F0, qF0, L0, L0b, lam, V0, V0t : optional
        Column physics parameters.  Each may be:

        - ``None`` (default) — a fresh CasADi symbolic scalar is created.
        - A numeric constant (int, float, ``cas.DM``) — folded into the
          model expressions; the parameter is excluded from ``model.params``
          and the CasADi function signatures.
        - An existing ``cas.SX`` symbolic — reused as-is (useful for sharing
          a symbol between models).
    M0 : array-like or tuple, optional
        Nominal tray holdups, length NT.  Default ``(NT, 1)`` creates a fresh
        symbolic vector ``cas.SX.sym("M0", NT)``.  Pass a numpy array or DM
        to bake in a fixed holdup profile.
    KcD, KcB : optional
        P-controller gains for condenser and reboiler level (default: symbolic).
    Ds, Bs : optional
        Nominal distillate and bottoms flow setpoints (default: symbolic).

    Returns
    -------
    StateSpaceModelCT
        Model with n=82 states, nu=5 inputs, ny=84 outputs (82 states + D, B).
        ``model.params`` contains only the parameters that remain symbolic;
        any parameter supplied as a numeric constant is excluded.

    Examples
    --------
    >>> model = build_cola_lv_ct_model()          # all params symbolic
    >>> model.n, model.nu, model.ny
    (82, 5, 84)
    >>> model.input_names
    ['LT', 'VB', 'F', 'zF', 'qF']
    >>> model.output_names[-2:]
    ['D', 'B']
    >>> model_no_k2 = build_cola_lv_ct_model(lam=0)   # K2-effect disabled
    >>> "lam" in model_no_k2.params
    False
    """
    n = 2 * NT
    nu = 5  # [LT, VB, F, zF, qF]
    ny = n + 2  # 82 states + D + B

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # Resolve each parameter: None → new symbolic scalar, tuple → shaped
    # symbolic array, numeric constant or existing SX → used as-is.
    all_params = make_symbolic_vars_from_kwargs(
        alpha=alpha,
        taul=taul,
        F0=F0,
        qF0=qF0,
        L0=L0,
        L0b=L0b,
        lam=lam,
        V0=V0,
        V0t=V0t,
        M0=M0,
        KcD=KcD,
        KcB=KcB,
        Ds=Ds,
        Bs=Bs,
    )
    alpha = all_params["alpha"]
    taul = all_params["taul"]
    F0 = all_params["F0"]
    qF0 = all_params["qF0"]
    L0 = all_params["L0"]
    L0b = all_params["L0b"]
    lam = all_params["lam"]
    V0 = all_params["V0"]
    V0t = all_params["V0t"]
    M0 = all_params["M0"]
    KcD = all_params["KcD"]
    KcB = all_params["KcB"]
    Ds = all_params["Ds"]
    Bs = all_params["Bs"]

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

    # Retain only the parameters that are symbolic SX variables; numeric
    # constants are already folded into the CasADi expressions.
    params = {
        k: v
        for k, v in all_params.items()
        if isinstance(v, cas.SX) and len(cas.symvar(v)) > 0
    }

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

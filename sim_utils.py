"""Simulation utility functions for distillation column models."""

import casadi as cas
import numpy as np
import pandas as pd


def run_simulation(
    t: pd.Series,
    U: pd.DataFrame,
    model,
    param_vals: dict,
    x0,
    sim_func=None,
) -> pd.DataFrame:
    """Run a simulation and return states, outputs and inputs as a DataFrame.

    Parameters
    ----------
    t : pd.Series
        Time points, length nT+1.  t[0] is the initial time (x0 applies);
        subsequent entries are output sample times.  The series name, if set,
        is used as the DataFrame index name.
    U : pd.DataFrame
        Input sequence, shape (nT, nu).  Column names must match
        model.input_names.  U.iloc[k] is applied during [t[k], t[k+1]].
    model : StateSpaceModelCT
        Model object supplying input_names, state_names, and output_names
        for the DataFrame column labels.
    param_vals : dict
        Parameter values passed to sim_func as positional arguments.
    x0 : array-like, shape (n,)
        Initial state vector.
    sim_func : cas.Function, optional
        Pre-built CasADi simulation function with signature::

            (t_eval, U, x0, *params) -> (X, Y)

        If None, a new function is built from the model using
        ``build_cola_lv_ct_model`` and
        ``make_n_step_simulation_function_from_model``.  Note: building
        compiles CasADi code and is slow; pass a pre-built sim_func for
        repeated calls.

    Returns
    -------
    pd.DataFrame
        MultiIndex-column DataFrame indexed by t with three column groups:

        ``"Inputs"``  — input sequence, forward-filled to align with all
                        time points (U.iloc[k] held constant after t[k])
        ``"States"``  — state trajectory X, shape (nT+1, n)
        ``"Outputs"`` — output trajectory Y, shape (nT+1, ny); for the LV
                        model this is the 82 states followed by D and B
    """
    if sim_func is None:
        from cas_models.continuous_time.simulate import (
            make_n_step_simulation_function_from_model,
        )
        from dist_model_cola_cas.cola_lv_model import build_cola_lv_ct_model

        dt = float(t.iloc[1] - t.iloc[0])
        nT = len(t) - 1
        _model = build_cola_lv_ct_model()
        sim_func = make_n_step_simulation_function_from_model(
            _model, dt=dt, nT=nT
        )

    x_traj, y_traj = sim_func(
        cas.DM(t.values),
        cas.DM(U.values),
        cas.DM(x0),
        *[param_vals[k] for k in model.params],
    )
    x_arr = np.array(x_traj)  # (nT+1, n)
    y_arr = np.array(y_traj)  # (nT+1, ny)

    # Forward-fill inputs: repeat last row so inputs align with all t points.
    # Convention: U.iloc[k] is the input applied from t[k] onwards.
    U_full = pd.concat([U, U.iloc[[-1]]], ignore_index=True).values

    cols = pd.MultiIndex.from_tuples(
        [("Inputs", name) for name in model.input_names]
        + [("States", name) for name in model.state_names]
        + [("Outputs", name) for name in model.output_names]
    )
    return pd.DataFrame(
        np.hstack([U_full, x_arr, y_arr]),
        index=t.values,
        columns=cols,
    )


def make_steady_state_solver(
    model, name="ss_solver", method="newton", opts=None
):
    """Build a steady-state solver for a continuous-time state-space model.

    Compiles a CasADi rootfinder that solves ``f(0, x, u, *params) = 0``
    for ``x``.  The solver is compiled once; the returned callable may then
    be called repeatedly with different inputs, parameter values, or initial
    guesses.

    Parameters
    ----------
    model : StateSpaceModelCT
        Continuous-time model with attributes ``f``, ``h``, ``n``, ``nu``,
        ``ny``, and ``params``.
    name : str, optional
        Name for the compiled CasADi rootfinder function.
    method : str, optional
        CasADi rootfinder algorithm.  ``"newton"`` (default) is fast with
        a good initial guess; ``"fast_newton"`` trades robustness for speed;
        ``"nlpsol"`` (wraps an NLP solver) is the most robust option for
        difficult problems.
    opts : dict, optional
        Options forwarded verbatim to ``cas.rootfinder``.

    Returns
    -------
    callable
        A function ``(x0, u, param_vals) -> (x_ss, y_ss)`` where:

        - ``x0`` : array-like, shape ``(n,)`` — initial state guess
        - ``u``  : array-like, shape ``(nu,)`` — input vector
        - ``param_vals`` : dict — values keyed by ``model.params``;
          extra keys are silently ignored
        - ``x_ss`` : ``np.ndarray``, shape ``(n,)`` — steady-state state
        - ``y_ss`` : ``np.ndarray``, shape ``(ny,)`` — steady-state output

    Notes
    -----
    For sweeps over an input range, pass the previous solution as ``x0``
    at each step (warm-starting).  Converging from a nearby point typically
    takes only a handful of Newton iterations.

    Examples
    --------
    >>> model = build_cola_lv_ct_model()
    >>> param_vals = make_nominal_lv_param_values()
    >>> solve_ss = make_steady_state_solver(model)
    >>> x_ss, y_ss = solve_ss(X_SS, U_NOM, param_vals)
    """
    if opts is None:
        opts = {}

    # Build SX variables matching model.params shapes.
    x_sx = cas.SX.sym("x", model.n)
    u_sx = cas.SX.sym("u", model.nu)
    p_sxs = [cas.SX.sym(k, *v.shape) for k, v in model.params.items()]

    # Residual: dx/dt at t=0 (steady state is time-invariant).
    rhs = model.f(cas.SX(0), x_sx, u_sx, *p_sxs)

    # Pack u and all params into a single parameter vector for the rootfinder.
    p_sx = cas.vertcat(u_sx, *p_sxs)
    g = cas.Function("g", [x_sx, p_sx], [rhs])
    rf = cas.rootfinder(name, method, g, opts)

    def solve(x0, u, param_vals):
        p_val = cas.vertcat(cas.DM(u), *[param_vals[k] for k in model.params])
        x_ss = np.array(rf(cas.DM(x0), p_val)).flatten()
        y_ss = np.array(
            model.h(
                cas.DM(0),
                cas.DM(x_ss),
                cas.DM(u),
                *[param_vals[k] for k in model.params],
            )
        ).flatten()
        return x_ss, y_ss

    return solve


def make_composition_targeting_solver(
    model,
    free_input_indices,
    target_state_indices,
    name="comp_target_solver",
    method="newton",
    opts=None,
):
    """Build a solver that finds free inputs to achieve target state values at
    SS.

    Extends the steady-state rootfinding approach by promoting a subset of
    the model inputs to decision variables and adding state-targeting
    residuals. The resulting system has ``n + n_free`` equations and unknowns:

        r[0 : n]          = f(0, x, u_full, *params)       (ODE residuals)
        r[n : n+n_free]   = x[target_state_indices] − targets

    Parameters
    ----------
    model : StateSpaceModelCT
        Model with attributes ``f``, ``h``, ``n``, ``nu``, ``ny``, ``params``.
    free_input_indices : sequence of int
        Indices into ``u`` that are free (solved for).  Length must equal
        ``len(target_state_indices)``.
    target_state_indices : sequence of int
        Indices into ``x`` whose steady-state values are targeted.
    name : str, optional
        Name for the compiled CasADi rootfinder.
    method : str, optional
        CasADi rootfinder algorithm (default ``"newton"``).
    opts : dict, optional
        Options forwarded to ``cas.rootfinder``.

    Returns
    -------
    callable
        ``solve(z0, u_fixed, targets, param_vals) -> (x_ss, u_free_ss, y_ss)``

        - ``z0`` : array-like, shape ``(n + n_free,)`` — initial guess
          ``[x_init, free_input_init]``
        - ``u_fixed`` : array-like, shape ``(nu − n_free,)`` — values of the
          fixed inputs, ordered by their position in ``u``
        - ``targets`` : array-like, shape ``(n_free,)`` — target state values
        - ``param_vals`` : dict — model parameter values
        - ``x_ss`` : ``np.ndarray``, shape ``(n,)``
        - ``u_free_ss`` : ``np.ndarray``, shape ``(n_free,)`` — solved inputs
        - ``y_ss`` : ``np.ndarray``, shape ``(ny,)``

    Examples
    --------
    >>> model = build_cola_lv_ct_model()
    >>> param_vals = make_nominal_lv_param_values()
    >>> # Find LT, VB (indices 0, 1) to hit target xB=x[0], xD=x[40]
    >>> solver = make_composition_targeting_solver(
    ...     model, free_input_indices=[0, 1], target_state_indices=[0, 40]
    ... )
    >>> z0 = np.concatenate([X_SS, [L0_DEFAULT, V0_DEFAULT]])
    >>> u_fixed = [F_val, zF_val, qF_val]   # in order of fixed indices (2, 3, 4)
    >>> x_ss, u_free_ss, y_ss = solver(z0, u_fixed, [xB_target, xD_target], param_vals)
    """
    if opts is None:
        opts = {}

    n_free = len(free_input_indices)
    assert n_free == len(target_state_indices), (
        "len(free_input_indices) must equal len(target_state_indices)"
    )
    fixed_input_indices = [
        i for i in range(model.nu) if i not in free_input_indices
    ]

    # ── Symbolic variables ────────────────────────────────────────────────
    # Decision vector: z = [x (n states), free inputs (n_free)]
    z_sx = cas.SX.sym("z", model.n + n_free)
    x_sx = z_sx[: model.n]
    u_free_sx = [z_sx[model.n + i] for i in range(n_free)]

    # Parameters: [fixed inputs, targets, model params]
    u_fixed_sx = [
        cas.SX.sym(f"u_fixed_{i}") for i in range(len(fixed_input_indices))
    ]
    targets_sx = [cas.SX.sym(f"target_{i}") for i in range(n_free)]
    p_sxs = [cas.SX.sym(k, *v.shape) for k, v in model.params.items()]

    # Reconstruct full u in the original input order
    u_full_list = [None] * model.nu
    for i, idx in enumerate(free_input_indices):
        u_full_list[idx] = u_free_sx[i]
    for i, idx in enumerate(fixed_input_indices):
        u_full_list[idx] = u_fixed_sx[i]
    u_sx = cas.vertcat(*u_full_list)

    # ── Residuals ─────────────────────────────────────────────────────────
    rhs = model.f(cas.SX(0), x_sx, u_sx, *p_sxs)
    comp_res = cas.vertcat(
        *[
            x_sx[idx] - targets_sx[i]
            for i, idx in enumerate(target_state_indices)
        ]
    )
    r = cas.vertcat(rhs, comp_res)

    p_sx = cas.vertcat(*u_fixed_sx, *targets_sx, *p_sxs)
    g = cas.Function("g", [z_sx, p_sx], [r])
    rf = cas.rootfinder(name, method, g, opts)

    def solve(z0, u_fixed, targets, param_vals):
        p_val = cas.vertcat(
            cas.DM(u_fixed),
            cas.DM(targets),
            *[param_vals[k] for k in model.params],
        )
        z_ss = np.array(rf(cas.DM(z0), p_val)).flatten()
        x_ss = z_ss[: model.n]
        u_free_ss = z_ss[model.n :]

        u_full = np.zeros(model.nu)
        for i, idx in enumerate(free_input_indices):
            u_full[idx] = u_free_ss[i]
        for i, idx in enumerate(fixed_input_indices):
            u_full[idx] = float(np.array(cas.DM(u_fixed)).flatten()[i])

        y_ss = np.array(
            model.h(
                cas.DM(0),
                cas.DM(x_ss),
                cas.DM(u_full),
                *[param_vals[k] for k in model.params],
            )
        ).flatten()
        return x_ss, u_free_ss, y_ss

    return solve

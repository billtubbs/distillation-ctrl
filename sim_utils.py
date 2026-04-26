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

        If None, a new function is built by calling
        ``build_cola_lv_sim_function(dt, nT)`` where dt and nT are derived
        from t.  Note: building compiles CasADi code and is slow; pass a
        pre-built sim_func for repeated calls.

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
        from dist_model_cola_cas.cola_lv_model import build_cola_lv_sim_function
        dt = float(t.iloc[1] - t.iloc[0])
        nT = len(t) - 1
        sim_func, _ = build_cola_lv_sim_function(dt=dt, nT=nT)

    x_traj, y_traj = sim_func(
        cas.DM(t.values),
        cas.DM(U.values),
        cas.DM(x0),
        *param_vals.values(),
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

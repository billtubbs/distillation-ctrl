import control as con
import numpy as np


def _forced_response_safe(sys, t, u):
    """Wrap control.forced_response for both 2- and 3-tuple return APIs."""
    out = con.forced_response(sys, T=t, U=u)
    if len(out) == 3:
        return out
    if len(out) == 2:
        return out[0], out[1], None
    raise ValueError(f"unexpected forced_response return length {len(out)}")


def mimo_forced_response(T, t, u):
    """Compute MIMO forced response without requiring Slycot."""
    try:
        _, yout, _ = _forced_response_safe(T, t, u)
        return yout
    except con.exception.ControlMIMONotImplemented:
        yout = np.zeros((T.noutputs, len(t)))
        for i in range(T.noutputs):
            for j in range(T.ninputs):
                _, yij, _ = _forced_response_safe(T[i, j], t, u[j])
                yout[i] += yij
        return yout

import control as con

Ts = 1.0

def c2d_with_delay(ct_tf, delay_steps, Ts=Ts, method='zoh'):
    """Discretize a continuous-time transfer function and append integer delay.

    - ct_tf: continuous-time control.TransferFunction
    - delay_steps: non-negative integer, number of samples of delay (z^-N)
    - Ts: sampling time in same units as model (minutes)
    - method: discretization method passed to con.c2d

    Returns a discrete-time transfer function with dt = Ts.
    """
    if delay_steps < 0:
        raise ValueError('delay_steps must be >= 0')
    dt_tf = con.c2d(ct_tf, Ts, method=method)
    if delay_steps == 0:
        return dt_tf
    # For discrete-time systems in python-control, a pure z^-N delay is represented
    # by numerator [1] and denominator [1, 0, 0, ..., 0] (N zeros).
    # This yields y[k] = u[k-N] exactly.
    delay_tf = con.TransferFunction([1], [1] + [0] * delay_steps, Ts)
    return dt_tf * delay_tf

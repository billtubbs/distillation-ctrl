import control as con

# default sample interval for deadtime in minutes (as used in the CT model)
Ts = 1.0


def delay_tf(N, pade_order=3):
    """Return a continuous-time delay approximation transfer function.

    Parameters
    ----------
    N : float
        Deadtime value in minutes (same units as Ts).
    pade_order : int, optional
        Timing accuracy for Pade approximation. Default is 3.

    Returns
    -------
    control.TransferFunction
        Continuous-time transfer function approximating the delay.
    """
    if N <= 0:
        return con.TransferFunction([1], [1])
    num, den = con.pade(N * Ts, pade_order)
    return con.TransferFunction(num, den)

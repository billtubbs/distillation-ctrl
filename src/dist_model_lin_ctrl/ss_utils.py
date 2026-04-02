"""State-space model utilities."""

import numpy as np


def augment_input_disturbances(A, B, C, D=None):
    """Augment a discrete-time state-space model with input disturbance
    integrators.

    Each input channel gains one integrating disturbance state d_u[j] modelled
    as a random walk:

        d_u(k+1) = d_u(k)                          (one state per input)
        x(k+1)   = A x(k) + B (u(k) + d_u(k))
        y(k)     = C x(k) + D u(k)

    The augmented state is  xa = [x; d_u]  (shape nx+nu), giving:

        xa(k+1) = Aa xa(k) + Ba u(k)
        y(k)    = Ca xa(k) + Da u(k)

    with

        Aa = [[A,  B ],     Ba = [[B],     Ca = [C  0],   Da = D
              [0,  I ]]           [0]]

    Parameters
    ----------
    A : (nx, nx) array
    B : (nx, nu) array
    C : (ny, nx) array
    D : (ny, nu) array or None
        Direct feedthrough matrix.  Defaults to zeros.

    Returns
    -------
    Aa : (nx+nu, nx+nu) array
    Ba : (nx+nu, nu) array
    Ca : (ny, nx+nu) array
    Da : (ny, nu) array
    """
    nx, nu = B.shape
    ny = C.shape[0]
    nd = nu  # one disturbance state per input

    if D is None:
        D = np.zeros((ny, nu))

    Aa = np.block(
        [
            [A, B],
            [np.zeros((nd, nx)), np.eye(nd)],
        ]
    )
    Ba = np.block(
        [
            [B],
            [np.zeros((nd, nu))],
        ]
    )
    Ca = np.block([[C, np.zeros((ny, nd))]])
    Da = D.copy()

    return Aa, Ba, Ca, Da

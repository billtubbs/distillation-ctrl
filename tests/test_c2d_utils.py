import pytest
import control as con
import numpy as np
from dist_ctrl.c2d_utils import c2d_with_delay


def test_c2d_with_delay_no_delay():
    s = con.TransferFunction.s
    ct = 1 / (s + 1)
    dt = c2d_with_delay(ct, 0, Ts=1.0, method="zoh")
    # check dt values and basic poles
    assert abs(dt.dt - 1.0) < 1e-9
    assert np.allclose(dt.poles(), [0.3678794412], atol=1e-5)


@pytest.mark.parametrize("delay_steps", [0, 1, 3])
def test_c2d_with_delay_n_steps(delay_steps):
    s = con.TransferFunction.s
    ct = 1 / (s + 1)
    dt = c2d_with_delay(ct, delay_steps, Ts=1.0, method="zoh")

    t = np.arange(0, 10)
    _, y = con.step_response(dt, t)

    # Normalize y into numpy array for indexing.
    y = np.asarray(y).flatten()

    # With ZOH, the first sample for first-order system is at t=0 and is zero.
    assert y[0] == 0

    # For delay_steps==0, response should start rising at sample 1.
    # For delay_steps>0, the step is delayed by delay_steps samples.
    expected_nonzero = delay_steps + 1
    assert all(np.isclose(y[:expected_nonzero], 0.0, atol=1e-9))
    assert y[expected_nonzero] > 0


def test_c2d_with_delay_negative_error():
    s = con.TransferFunction.s
    ct = 1 / (s + 1)
    try:
        c2d_with_delay(ct, -1)
        assert False, "Expected ValueError for negative delay_steps"
    except ValueError:
        assert True


def test_c2d_with_delay_applied_on_discretized_first_order():
    s = con.TransferFunction.s
    ct = 1 / (s + 1)
    dt_base = con.c2d(ct, 1.0, method="zoh")
    delay = con.TransferFunction([1], [1, 0], 1.0)  # z^-1
    combined = dt_base * delay

    t = np.arange(0, 10)
    _, y_combined = con.step_response(combined, t)
    y_combined = np.asarray(y_combined).flatten()

    # first sample should be zero from delay, second should be same as base[0]
    _, y_base = con.step_response(dt_base, t)
    y_base = np.asarray(y_base).flatten()

    assert y_combined[0] == 0.0
    assert np.isclose(y_combined[1], y_base[0], atol=1e-9)
    assert y_combined[2] > y_combined[1]

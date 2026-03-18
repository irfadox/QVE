"""
gradients.py
------------
Analytical and numerical gradients for VQE circuits.
"""

import numpy as np


def parameter_shift_gradient(cost_fn, x: np.ndarray,
                             shift: float = np.pi / 2) -> np.ndarray:
    """
    Compute the gradient of *cost_fn* at point *x* via the parameter-shift rule.

    Parameters
    ----------
    cost_fn : callable
        f(x) → float.  The VQE cost function.
    x : np.ndarray
        Current parameter values (shape: (n_params,)).
    shift : float
        Shift amount (default π/2 for standard gates like RY, RX, RZ).

    Returns
    -------
    np.ndarray
        Gradient vector of shape (n_params,).
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += shift
        x_minus[i] -= shift
        grad[i] = (cost_fn(x_plus) - cost_fn(x_minus)) / (2 * np.sin(shift))

    return grad


def finite_difference_gradient(cost_fn, x: np.ndarray,
                               epsilon: float = 1e-4) -> np.ndarray:
    """
    Fallback: central finite-difference gradient.

    Cheaper approximation when parameter-shift isn't needed.
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (cost_fn(x_plus) - cost_fn(x_minus)) / (2 * epsilon)

    return grad


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test with a simple quadratic
    def simple_cost(x):
        return np.sum((x - 1.0) ** 2)

    x0 = np.array([0.0, 0.5, 2.0])
    grad = finite_difference_gradient(simple_cost, x0)
    print(f"x = {x0}")
    print(f"Gradient (finite diff) = {grad}")
    print(f"Expected = {2 * (x0 - 1.0)}")

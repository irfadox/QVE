"""
optimizers.py
-------------
Unified interface for classical optimizers in VQE.
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from src.gradients import parameter_shift_gradient



AVAILABLE_OPTIMIZERS = ["COBYLA", "SPSA", "Adam", "L-BFGS-B", "QNSPSA"]


def optimize(cost_fn, x0: np.ndarray, method: str = "COBYLA",
             maxiter: int = 200, **kwargs) -> dict:
    """
    Run the chosen optimizer to minimize *cost_fn* starting from *x0*.

    Parameters
    ----------
    cost_fn : callable
        f(x) → float.  The VQE expectation-value function.
    x0 : np.ndarray
        Initial parameter vector.
    method : str
        One of AVAILABLE_OPTIMIZERS.
    maxiter : int
        Maximum optimizer iterations / updates.
    **kwargs
        Method-specific hyper-parameters (see individual implementations).

    Returns
    -------
    dict
        "optimal_energy"        : float
        "optimal_params"        : np.ndarray
        "convergence_history"   : list[float]  — energy after each eval / step
        "num_evals"             : int           — total cost-function evaluations
        "gradient_norms"        : list[float]  — ‖∇E‖ per step (if applicable)
    """
    method_upper = method.upper()
    if method_upper == "COBYLA":
        return _optimize_cobyla(cost_fn, x0, maxiter, **kwargs)
    elif method_upper == "SPSA":
        return _optimize_spsa(cost_fn, x0, maxiter, **kwargs)
    elif method_upper == "ADAM":
        return _optimize_adam(cost_fn, x0, maxiter, **kwargs)
    elif method_upper in ("L-BFGS-B", "LBFGSB"):
        return _optimize_lbfgsb(cost_fn, x0, maxiter, **kwargs)
    elif method_upper == "QNSPSA":
        return _optimize_qnspsa(cost_fn, x0, maxiter, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer '{method}'. "
                         f"Choose from {AVAILABLE_OPTIMIZERS}")



def _optimize_cobyla(cost_fn, x0, maxiter, **kw):
    """Constrained Optimization BY Linear Approximation."""
    history = []
    eval_count = {"n": 0}

    def tracked_cost(x):
        eval_count["n"] += 1
        e = cost_fn(x)
        history.append(e)
        return e

    result = scipy_minimize(tracked_cost, x0, method="COBYLA",
                            options={"maxiter": maxiter, "rhobeg": 0.5})
    return {
        "optimal_energy": result.fun,
        "optimal_params": result.x,
        "convergence_history": history,
        "num_evals": eval_count["n"],
        "gradient_norms": [],       # no gradient info for COBYLA
    }



def _optimize_spsa(cost_fn, x0, maxiter, *,
                   a0=0.2, c0=0.1, alpha=0.602, gamma=0.101, **kw):
    """
    Simultaneous Perturbation Stochastic Approximation.

    Uses only **2 function evaluations per step** regardless of the number
    of parameters (unlike parameter-shift which needs 2p).

    Hyper-parameters follow the standard SPSA schedule:
        a_k = a0 / (k+1)^alpha       (step size)
        c_k = c0 / (k+1)^gamma       (perturbation size)
    """
    x = x0.copy().astype(float)
    n_params = len(x)
    history = []
    gradient_norms = []
    eval_count = 0
    rng = np.random.default_rng(kw.get("spsa_seed", 0))

    best_energy = float("inf")
    best_x = x.copy()

    for k in range(maxiter):
        a_k = a0 / (k + 1) ** alpha
        c_k = c0 / (k + 1) ** gamma

        # Random perturbation: ±1 with equal probability (Bernoulli).
        delta = rng.choice([-1.0, 1.0], size=n_params)

        # Two function evaluations
        e_plus = cost_fn(x + c_k * delta)
        e_minus = cost_fn(x - c_k * delta)
        eval_count += 2

        # Gradient estimate
        g_hat = (e_plus - e_minus) / (2 * c_k * delta)
        gradient_norms.append(float(np.linalg.norm(g_hat)))

        # Update parameters
        x = x - a_k * g_hat

        # Track best
        current_e = (e_plus + e_minus) / 2  # midpoint estimate
        history.append(current_e)
        if current_e < best_energy:
            best_energy = current_e
            best_x = x.copy()

    # Final evaluation
    final_e = cost_fn(best_x)
    eval_count += 1

    return {
        "optimal_energy": final_e,
        "optimal_params": best_x,
        "convergence_history": history,
        "num_evals": eval_count,
        "gradient_norms": gradient_norms,
    }



def _optimize_adam(cost_fn, x0, maxiter, *,
                   lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, **kw):
    """
    Adam optimizer with parameter-shift gradients.

    Adam maintains per-parameter learning rates via exponential moving
    averages of the gradient (m) and squared gradient (v).
    """
    x = x0.copy().astype(float)
    n_params = len(x)
    history = []
    gradient_norms = []
    eval_count = 0

    # Adam state
    m = np.zeros(n_params)  # first moment
    v = np.zeros(n_params)  # second moment

    best_energy = float("inf")
    best_x = x.copy()

    for t in range(1, maxiter + 1):
        # Compute gradient via parameter-shift rule (2 * n_params evals)
        grad = parameter_shift_gradient(cost_fn, x)
        eval_count += 2 * n_params

        gradient_norms.append(float(np.linalg.norm(grad)))

        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Parameter update
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        # Evaluate current energy (1 eval)
        e = cost_fn(x)
        eval_count += 1
        history.append(e)

        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    return {
        "optimal_energy": best_energy,
        "optimal_params": best_x,
        "convergence_history": history,
        "num_evals": eval_count,
        "gradient_norms": gradient_norms,
    }



def _optimize_lbfgsb(cost_fn, x0, maxiter, **kw):
    """
    Limited-memory BFGS with box constraints.

    Uses parameter-shift gradients. L-BFGS-B builds an approximate Hessian
    from recent gradient history — very efficient for smooth landscapes,
    but sensitive to noise.
    """
    history = []
    gradient_norms = []
    eval_count = {"n": 0}

    def tracked_cost(x):
        eval_count["n"] += 1
        e = cost_fn(x)
        history.append(e)
        return e

    def grad_fn(x):
        g = parameter_shift_gradient(cost_fn, x)
        eval_count["n"] += 2 * len(x)
        gradient_norms.append(float(np.linalg.norm(g)))
        return g

    result = scipy_minimize(tracked_cost, x0, method="L-BFGS-B",
                            jac=grad_fn,
                            options={"maxiter": maxiter, "maxfun": maxiter * 20})
    return {
        "optimal_energy": result.fun,
        "optimal_params": result.x,
        "convergence_history": history,
        "num_evals": eval_count["n"],
        "gradient_norms": gradient_norms,
    }



def _optimize_qnspsa(cost_fn, x0, maxiter, *,
                     a0=0.2, c0=0.1, alpha=0.602, gamma=0.101,
                     metric_period=5, lam=0.01, **kw):
    """
    Quantum Natural SPSA — SPSA with a regularised quantum metric tensor.

    Every *metric_period* steps, a rank-1 SPSA estimate of the Fubini-Study
    metric tensor G is accumulated.  The parameter update becomes:

        x_{k+1} = x_k  −  a_k * G_k^{-1} * g_k

    This "rotates" the gradient into the natural parameter space, greatly
    improving convergence on barren landscapes.

    The metric is regularised as  G + λI  to stay invertible.
    """
    x = x0.copy().astype(float)
    n = len(x)
    history = []
    gradient_norms = []
    eval_count = 0
    rng = np.random.default_rng(kw.get("qnspsa_seed", 0))

    # Running average of the metric tensor (start as identity)
    G_avg = np.eye(n)
    beta_metric = 0.9  # EMA smoothing for metric

    best_energy = float("inf")
    best_x = x.copy()

    for k in range(maxiter):
        a_k = a0 / (k + 1) ** alpha
        c_k = c0 / (k + 1) ** gamma

        delta = rng.choice([-1.0, 1.0], size=n)

        # --- SPSA gradient estimate (2 evals) ---
        e_plus = cost_fn(x + c_k * delta)
        e_minus = cost_fn(x - c_k * delta)
        eval_count += 2

        g_hat = (e_plus - e_minus) / (2 * c_k * delta)
        gradient_norms.append(float(np.linalg.norm(g_hat)))

        # --- Metric tensor estimate (every metric_period steps, +2 evals) ---
        if k % metric_period == 0:
            delta2 = rng.choice([-1.0, 1.0], size=n)
            # Rank-1 SPSA estimate of the metric:
            # G ≈ (∂f/∂δ1)(∂f/∂δ2)^T  (outer product of SPSA gradient estimates)
            e_plus2 = cost_fn(x + c_k * delta2)
            e_minus2 = cost_fn(x - c_k * delta2)
            eval_count += 2

            g_hat2 = (e_plus2 - e_minus2) / (2 * c_k * delta2)
            G_sample = np.outer(g_hat, g_hat2)
            # Symmetrise
            G_sample = (G_sample + G_sample.T) / 2

            G_avg = beta_metric * G_avg + (1 - beta_metric) * G_sample

        # Regularise and invert
        G_reg = G_avg + lam * np.eye(n)
        try:
            G_inv = np.linalg.inv(G_reg)
        except np.linalg.LinAlgError:
            G_inv = np.eye(n)  # fallback to vanilla SPSA

        # Natural gradient update
        nat_grad = G_inv @ g_hat
        x = x - a_k * nat_grad

        current_e = (e_plus + e_minus) / 2
        history.append(current_e)
        if current_e < best_energy:
            best_energy = current_e
            best_x = x.copy()

    final_e = cost_fn(best_x)
    eval_count += 1

    return {
        "optimal_energy": final_e,
        "optimal_params": best_x,
        "convergence_history": history,
        "num_evals": eval_count,
        "gradient_norms": gradient_norms,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test: minimize a 3-D quadratic
    def rosenbrock(x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                   for i in range(len(x) - 1))

    x0 = np.array([0.0, 0.0, 0.0])
    for name in AVAILABLE_OPTIMIZERS:
        result = optimize(rosenbrock, x0, method=name, maxiter=100)
        print(f"{name:10s}: energy = {result['optimal_energy']:.4f}, "
              f"evals = {result['num_evals']}")

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cvxpy"])
    import cvxpy as cp

# ==========================================================================
# CELL 7: Almgren-Chriss Optimal Liquidation Solver (QP via cvxpy)
# ==========================================================================
# QP:
#   min   sum_m  v_m' H v_m   +   gamma_star * sum_m  x_m' Sigma x_m
#   s.t.  sum_m v_m = Q_trigger        (liquidate everything)
#         x_m = Q_trigger - sum_{j<=m} v_j

def solve_almgren_chriss(Q_trigger, N_liq, H_matrix, Sigma, gamma_star, tau=1.0):
    N = len(Q_trigger)

    # Nothing to liquidate
    if np.linalg.norm(Q_trigger) < 1e-8:
        return np.zeros((N_liq, N)), np.zeros((N_liq + 1, N)), 0.0

    v = cp.Variable((N_liq, N))

    impact_cost  = 0
    holding_risk = 0
    for m in range(N_liq):
        impact_cost  += cp.quad_form(v[m, :], cp.psd_wrap(H_matrix))
        x_m           = Q_trigger - cp.sum(v[:m+1, :], axis=0)
        holding_risk += gamma_star * cp.quad_form(x_m, cp.psd_wrap(Sigma))

    objective   = cp.Minimize(impact_cost + holding_risk)
    constraints = [cp.sum(v, axis=0) == Q_trigger]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            v_opt = v.value
            x_path = np.zeros((N_liq + 1, N))
            x_path[0] = Q_trigger
            for m in range(N_liq):
                x_path[m + 1] = x_path[m] - v_opt[m]
            return v_opt, x_path, float(prob.value)
    except Exception:
        pass

    # Fallback: TWAP
    v_twap = np.tile(Q_trigger / N_liq, (N_liq, 1))
    x_path = np.zeros((N_liq + 1, N))
    x_path[0] = Q_trigger
    for m in range(N_liq):
        x_path[m + 1] = x_path[m] - v_twap[m]
    return v_twap, x_path, float("inf")



def compute_twap_cost(Q_trigger, N_liq, H_matrix, Sigma, gamma_star):
    """Compute TWAP baseline cost for comparison."""
    v_twap = np.tile(Q_trigger / N_liq, (N_liq, 1))
    x      = np.zeros((N_liq + 1, len(Q_trigger)))
    x[0]   = Q_trigger
    cost   = 0.0
    for m in range(N_liq):
        x[m + 1] = x[m] - v_twap[m]
        cost     += v_twap[m] @ H_matrix @ v_twap[m]
        cost     += gamma_star * x[m + 1] @ Sigma @ x[m + 1]
    return cost
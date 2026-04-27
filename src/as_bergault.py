# =============================================================================
# avellaneda_stoikov.py —  Multi-asset Avellaneda-Stoikov (Bergault)
# =============================================================================

import numpy as np
import pandas as pd


def compute_half_spreads(gamma, kvec, Gamma, N):
    """
    Bergault asymptotic half-spread for each asset.
    """
    hs = np.zeros(N)
    for i in range(N):
        liq  = (1 / gamma) * np.log(1 + gamma / kvec[i])
        risk = (np.sqrt(gamma) / 2) * np.sqrt(np.dot(Gamma[i, :], Gamma[:, i])) / kvec[i]
        hs[i] = liq + risk
    return hs


def simulate(gamma, day_indices, prices_all, Avec, kvec, Gamma, N,
             lot_size, fee_per_share, max_inv):
    """
    Single-gamma AS simulation


    Parameters
    ----------
    gamma       : float
    day_indices : list of index arrays
    prices_all  : ndarray (T, N)
    Avec, kvec  : ndarray (N,)
    Gamma       : ndarray (N, N)
    N           : int
    lot_size    : int
    fee_per_share : float
    max_inv     : int  

    Returns
    -------
    pnl : ndarray (D,)
    hs  : ndarray (N,)
    """
    rng = np.random.RandomState(42)
    hs  = compute_half_spreads(gamma, kvec, Gamma, N)

    pnl = []
    for idx in day_indices:
        Q    = np.zeros(N)
        cash = 0

        for t in idx:
            S    = prices_all[t]
            skew = -np.sqrt(gamma) * (Gamma @ Q)
            asks = S + hs + skew
            bids = S - hs + skew

            for i in range(N):
                da = max(asks[i] - S[i], 0)
                db = max(S[i] - bids[i], 0)

                lam_a = Avec[i] * np.exp(-kvec[i] * da)
                lam_b = Avec[i] * np.exp(-kvec[i] * db)

                p_a = 1 - np.exp(-lam_a)
                p_b = 1 - np.exp(-lam_b)

                u = rng.rand()

                if u < p_a and Q[i] > -max_inv:
                    cash += (asks[i] - fee_per_share) * lot_size
                    Q[i] -= lot_size
                elif u > 1 - p_b and Q[i] < max_inv:
                    cash -= (bids[i] + fee_per_share) * lot_size
                    Q[i] += lot_size

        pnl.append(cash + Q @ prices_all[idx[-1]])

    return np.array(pnl), hs


def gamma_sharpe_sweep(gamma_grid, Avec, kvec, Sigma, Gamma,
                       prices_all, train_day_indices, test_day_indices,
                       N, lot_size, fee_per_share, max_inv):
    """
    Grid-search over gamma to maximise Markowitz Sharpe on train data.

    Returns
    -------
    results : list of dicts
    best    : dict — row with highest sharpe_train
    """
    def sharpe(x):
        return x.mean() / x.std() if x.std() > 1e-12 else 0

    results = []
    for g in gamma_grid:
        pnl_tr, hs_tr = simulate(g, train_day_indices, prices_all,
                                  Avec, kvec, Gamma, N,
                                  lot_size, fee_per_share, max_inv)
        pnl_te, hs_te = simulate(g, test_day_indices, prices_all,
                                  Avec, kvec, Gamma, N,
                                  lot_size, fee_per_share, max_inv)
        results.append({
            "gamma":        g,
            "sharpe_train": sharpe(pnl_tr),
            "sharpe_test":  sharpe(pnl_te),
            "mean_train":   pnl_tr.mean(),
            "mean_test":    pnl_te.mean(),
            "hs_tr":        hs_tr,
            "hs_te":        hs_te,
        })

    import pandas as pd
    res_df = pd.DataFrame(results)
    best   = results[int(res_df["sharpe_train"].idxmax())]
    print(f"\nBest gamma: {best['gamma']:.6e}  "
          f"(train Sharpe: {best['sharpe_train']:.3f}, "
          f"test Sharpe:  {best['sharpe_test']:.3f})")
    return results, best
# =============================================================================
# risk_matrix.py — Build Σ and Γ (liquidity-adjusted risk matrix)
# =============================================================================
# Γ = D+^{-1/2} · sqrtm(D+^{1/2} · Σ · D+^{1/2}) · D+^{-1/2}
# D+ = diag(A_i · k_i)  — Bergault et al. (2018), Section 3.

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm


def build_covariance(
    prices_df: pd.DataFrame,
    timestamps,
    train_days_set: set,
) -> np.ndarray:
    """
    Compute the N×N log-return covariance matrix on training bars only.

    Returns
    -------
    Sigma : ndarray (N, N), positive definite
    """
    train_mask = pd.Series(timestamps).dt.date.isin(train_days_set).values
    log_ret    = np.log(
        prices_df.loc[timestamps[train_mask]] /
        prices_df.loc[timestamps[train_mask]].shift(1)
    ).dropna()
    Sigma    = log_ret.cov().values
    pd_check = np.all(np.linalg.eigvalsh(Sigma) > 0)
    print(f"Σ shape: {Sigma.shape},  PD: {pd_check}")
    return Sigma


def build_sigma(prices_all: np.ndarray,
                train_day_indices: list) -> np.ndarray:
    """
    Compute the (N×N) sample covariance matrix of log-returns
    using TRAIN bars only.
    """
    # Build a boolean mask over all bars
    T = prices_all.shape[0]
    train_mask = np.zeros(T, dtype=bool)
    for idx in train_day_indices:
        train_mask[idx] = True

    # Log-returns (length T-1); shift the mask by 1 to align
    log_rets = np.diff(np.log(prices_all), axis=0)   # (T-1, N)
    log_rets_train = log_rets[train_mask[1:]]

    return np.cov(log_rets_train.T)

def build_Gamma(
    Avec: np.ndarray,
    kvec: np.ndarray,
    Sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute the liquidity-adjusted risk matrix Γ.

    Returns
    -------
    Gamma : ndarray (N, N)
    """
    D          = Avec * kvec
    D_sqrt     = np.diag(np.sqrt(D))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    M          = D_sqrt @ Sigma @ D_sqrt
    M_sqrt     = np.real(sqrtm(M))
    Gamma      = D_inv_sqrt @ M_sqrt @ D_inv_sqrt
    pd_check   = np.all(np.linalg.eigvalsh(Gamma) > 0)
    print(f"Γ shape: {Gamma.shape},  PD: {pd_check}")
    return Gamma
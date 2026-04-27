# =============================================================================
# calibration.py — Calibration of (A, k) and η per asset
# =============================================================================
# λ(δ) = A · exp(−k·δ).  We estimate (A, k) by regressing log(λ) on δ.
# η_i  = median_CS_spread_i / (2 · sqrt(ADV_i))  [Almgren et al. 2005]

import numpy as np
import pandas as pd
from src.config import TAU


def estimate_k_A(df: pd.DataFrame, delta_grid: np.ndarray) -> tuple[float, float]:
    """
    Estimate Poisson arrival parameters (k, A) for one asset via log-linear OLS.

    For each δ the empirical fill probability is the fraction of bars where
    the price touched mid ± δ. Then λ = −log(1−p), and
    log λ = log A − k·δ is fit by OLS.
    """
    mid = (df["high"] + df["low"]) / 2.0

    lambdas, deltas_valid = [], []
    for delta in delta_grid:
        p_ask = (df["high"] >= mid + delta).mean()
        p_bid = (df["low"]  <= mid - delta).mean()
        p     = float(np.clip((p_ask + p_bid) / 2.0, 1e-6, 0.999))
        lam   = -np.log(1.0 - p)
        if lam > 0:
            lambdas.append(lam)
            deltas_valid.append(delta)

    if len(deltas_valid) < 2:
        return 1.0, 1.0

    deltas_valid = np.array(deltas_valid)
    lambdas      = np.array(lambdas)
    slope, intercept = np.polyfit(deltas_valid, np.log(lambdas), 1)
    return float(-slope), float(np.exp(intercept))


def calibrate_all_assets(
    asset_data: dict,
    asset_names: list,
    train_date_set: set,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Calibration for every asset independently on train data only.

    Returns
    -------
    Avec : ndarray (N,)   baseline intensities
    kvec : ndarray (N,)   decay rates
    """
    Avec_list, kvec_list = [], []
    print(f"\n{'Asset':>8} {'A':>10} {'k':>10}")
    print("-" * 32)

    for name in asset_names:
        df       = asset_data[name]
        train_df = df[df["datetime"].dt.date.isin(train_date_set)]

        spread_proxy = (train_df["high"] - train_df["low"]).median()
        delta_grid   = np.linspace(0.5 * spread_proxy, 2.0 * spread_proxy, 20)

        k, A = estimate_k_A(train_df, delta_grid)
        kvec_list.append(k)
        Avec_list.append(A)
        print(f"{name:>8} {A:>10.4f} {k:>10.4f}")

    return np.array(Avec_list), np.array(kvec_list)


def calibrate_eta(
    all_sheets: dict,
    asset_names: list,
    train_days_set: set,
    tau: float = TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrate temporary-impact coefficients η_i from train data only.
    H_ii = η_i / τ  (diagonal temporary-impact cost matrix).

    Returns: eta_vec (N,), H_matrix (N, N)
    """
    N       = len(asset_names)
    eta_vec = np.zeros(N)

    print("Temporary Impact Calibration (train-only CS spread + ADV):")
    print("=" * 75)
    print(f"{'Asset':<8} {'Med CS($)':>10} {'Med DailyVol':>14} "
          f"{'eta_i':>12} {'eta_i/tau':>12}")
    print("-" * 60)

    for i, name in enumerate(asset_names):
        df             = all_sheets[name].copy()
        df["datetime"] = pd.to_datetime(df["Date/Time"])
        df["date"]     = df["datetime"].dt.date
        df             = df[df["date"].isin(train_days_set)]

        cs_spread = pd.to_numeric(df["CS Spread $"], errors="coerce")
        volume    = pd.to_numeric(df["Volume"],      errors="coerce")

        med_cs     = cs_spread[cs_spread > 0].median()
        adv        = volume.groupby(df["date"]).sum().median()
        eta_vec[i] = med_cs / (2.0 * np.sqrt(adv))
        print(f"{name:<8} {med_cs:>10.4f} {adv:>14,.0f} "
              f"{eta_vec[i]:>12.6f} {eta_vec[i]/tau:>12.6f}")

    H_matrix = np.diag(eta_vec / tau)
    print(f"\nH matrix (diag): min={np.diag(H_matrix).min():.2e}, "
          f"max={np.diag(H_matrix).max():.2e}")
    return eta_vec, H_matrix
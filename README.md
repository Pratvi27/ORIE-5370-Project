# Multi-Asset Market Making & Liquidation Framework

**ORIE 5370 — Optimization Modelling in Finance**

A unified, state-driven market making engine that synthesizes continuous-time stochastic control (Avellaneda-Stoikov) with discrete-time convex optimization (Almgren-Chriss) across 19 correlated equity assets.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Theoretical Framework](#theoretical-framework)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Quickstart](#quickstart)
- [Pipeline Phases](#pipeline-phases)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Module Reference](#module-reference)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

This framework solves the multi-asset market making problem in four sequential phases:

1. **Parameter Calibration** — fit order arrival intensities `λ_i(δ) = A_i · exp(−k_i · δ)` using OLS
2. **Avellaneda-Stoikov (AS) Passive Quoting** — quote optimal bid/ask prices using the Bergault et al. (2018) closed-form approximation of the Hamilton-Jacobi value function
3. **Markowitz / Sharpe Gamma Sweep** — find the risk-aversion parameter `γ*` that maximises the Sharpe ratio
4. **Almgren-Chriss (AC) Liquidation** — upon breaching an inventory norm threshold or entering the EOD flattening window, solve a convex QP to optimally unwind positions

---

## Project Structure

```
market_making_project/
│
├── data/
│   └── Cleaned Data.xlsx          # Input: 1-min OHLCV + CS spread, 19 assets
│
├── src/
│   ├── config.py                   # All hyperparameters
│   ├── data_loader.py              # Load, preprocess, align, split train/test
│   ├── risk_matrix.py              # Build Σ, Γ matrix from train bars only
│   ├── calibration.py              # OLS: fit A_i, k_i 
│   ├── simulation.py               # simulation driver code
│   ├── as_bergault.py              # AS Multi-asset driver
│   └── almgren_chriss.py           # AC QP solver 
│
├── run_pipeline.ipynb            # End-to-end runner — single entry point
│
├── outputs/                       
│   ├── phase1_A_k.csv
│   ├── phase3_gamma_star.csv
│   ├── phase4_daily_pnl.csv
|   ├── phase4_dashboard.png
│
└── README.md
```

---

## Theoretical Framework

### Phase 1 — Parameter Calibration

Market order arrivals at spread distance `δ` are modelled as a Poisson process:

```
λ_i(δ) = A_i · exp(−k_i · δ)
```

Parameters `(A_i, k_i)` are estimated via log-linear OLS on empirical fill probabilities across a grid of spread levels, using **train data only** to prevent look-ahead leakage.

### Phase 2 — Bergault Multi-Asset AS Regime

Following Bergault, Evangelista, Guéant, and Vieira (2018), the Hamilton-Jacobi equation is solved in closed form by replacing Hamiltonian functions with their second-order Taylor expansions. The central object is the **liquidity-adjusted risk matrix**:

```
Γ = D₊^{−1/2} (D₊^{1/2} Σ D₊^{1/2})^{1/2} D₊^{−1/2}
```

where `D₊ = diag(A_i · k_i)` encodes per-asset liquidity curvature and `Σ` is the empirical return covariance matrix. In the asymptotic regime the optimal half-spread and inventory skew for asset `i` are:

```
Δ_i  = (1/γz_i) ln(1 + γz_i/k_i)  +  γ^{1/2} z_i e_i^T Γ e_i / 2
skew_i = −γ^{1/2} Q_t^T Γ e_i
```

The reservation price and quoted prices follow:

```
R(s,t) = S_t − γ(T−t)Σ Q_t
Ask_t  = R(s,t) + Δ
Bid_t  = R(s,t) − Δ
```

### Phase 3 — Markowitz / Sharpe Optimisation

The risk-aversion parameter `γ` is selected by grid search to maximise the **train-set Sharpe ratio** of terminal wealth:

```
γ* = argmax_γ  E[W_T(γ)] / √(Var(W_T(γ)))
```

Terminal wealth is `W_T = X_T + Q_T^T S_T`, where `X_T` is cash and `Q_T` is end-of-day inventory marked to mid minus a half-spread liquidation charge.

### Phase 4 — Almgren-Chriss Dual-Trigger Liquidation

The AS regime is interrupted when either trigger fires:

| Trigger | Condition |
|---------|-----------|
| Intraday risk limit | `‖Q_t‖ > Q_max` |
| EOD flattening | `t ≥ T − T_close` and `‖Q_t‖ > 1` |

Upon triggering, the following convex QP is solved (Almgren & Chriss, 2001):

```
min_{v_1,...,v_{N_liq}}  Σ_m v_m^T H v_m  +  γ* Σ_m x_m^T Σ x_m

s.t.  x_0 = Q_trigger,  x_{N_liq} = 0,  x_m = x_{m-1} − v_m
```

where `H = diag(η_i / τ)` is the temporary impact cost matrix. A TWAP schedule is used as a fallback if the QP solver fails to converge.

---

> **Note:** `cvxpy` is required for the Phase 4 Almgren-Chriss QP solver. 

---

## Data Requirements

Place your data file at:

```
data/Cleaned Data.xlsx
```

The workbook must contain:
- One sheet per asset ticker (e.g., `AAPL`, `NVDA`, …)
- A sheet named `Guide` (excluded automatically) with metadata/documentation
- Each asset sheet must have the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `DateTime` | datetime | Timestamp of the 1-minute bar |
| `Open` | float | Open price |
| `High` | float | High price |
| `Low` | float | Low price |
| `Close` | float | Close price |
| `Volume` | float | Bar volume (shares) |
| `CS Spread` | float | Quoted bid-ask spread at bar close |

The pipeline was developed on 19 S&P 500 large-cap equities with 1-minute bars from September 3, 2025 to approximately March 2026 (~133 trading days). It is asset-count agnostic — add or remove sheets freely.

---

## Pipeline Phases in run_pipeline.ipynb

### Phase 1 — Market Calibration (`phase1_calibration.py`)

Estimates `(A_i, k_i)` for each asset using log-linear regression on empirical fill probabilities across a grid of `DELTA_GRID_POINTS` spread levels. Only train-set bars are used.

**Outputs:** `outputs/phase1_A_k.csv`

---

### Phase 2 — Γ Matrix and AS Regime (`market_making.py`)

Builds the liquidity-adjusted risk matrix `Γ` from `(A_vec, k_vec, Σ)` and implements the per-day AS simulation loop. Used internally by Phase 3.

---

### Phase 3 — Gamma Sweep (`phase3_gamma_optimization.py`)

Runs the AS simulator over `GAMMA_GRID_POINTS` values of `γ` on both train and test splits. Selects `γ*` as the value maximising the **train** Sharpe ratio (no test-set peeking).

**Outputs:** `outputs/phase3_gamma_star.csv`

---

### Phase 4 — Integrated Simulation (`phase4_liquidation.py`)

Runs the full AS + AC dual-trigger engine. Reports daily PnL, trigger counts, AC costs, fill counts, and inventory norms for four conditions: `{train, test} × {with AC, without AC}`.

**Outputs:**
- `outputs/phase4_daily_pnl.csv`

---

## Configuration

All hyperparameters are in `src/config.py`. No other file contains magic numbers.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_PATH` | `data/Cleaned Data.xlsx` | Path to input workbook |
| `GUIDE_SHEET` | `Guide` | Sheet name to exclude from asset list |
| `N_TRAIN_DAYS` | `93` | Number of trading days in the train set |
| `MIN_BARS_PER_DAY` | `10` | Minimum bars for a day to be included |
| `LOT_SIZE` | `100` | Shares per fill event |
| `FEE_PER_SHARE` | `0.001` | Transaction cost per share ($) |
| `MAX_INV_SHARES` | `1000` | Per-asset inventory cap (shares) |
| `MAX_SPREAD_PCT` | `0.005` | Cap on CS spread as fraction of mid |
| `DELTA_GRID_POINTS` | `20` | Spread levels for MLE estimation |
| `GAMMA_LOG_MIN` | `-3` | Lower bound of γ grid (`10^GAMMA_LOG_MIN`) |
| `GAMMA_LOG_MAX` | `2` | Upper bound of γ grid (`10^GAMMA_LOG_MAX`) |
| `GAMMA_GRID_POINTS` | `25` | Number of γ candidates |
| `Q_MAX_TRIGGER` | `2000` | Inventory norm threshold for AC trigger |
| `T_CLOSE` | `30` | Minutes before EOD to force flattening |
| `MAX_AC_STEPS` | `30` | Maximum liquidation steps in AC regime |
| `MIN_AC_STEPS` | `5` | Minimum liquidation steps in AC regime |
| `TAU` | `1.0` | Bar duration (minutes) |
| `FILL_PROB_CAP` | `0.1` | Per-bar fill probability ceiling |
| `RANDOM_SEED` | `42` | Global RNG seed for reproducibility |

---

## Output Files

### `phase1_A_k.csv`

| Column | Description |
|--------|-------------|
| `ticker` | Asset name |
| `A` | Baseline arrival intensity |
| `k` | Spread sensitivity |

### `phase3_gamma_star.csv`

| Column | Description |
|--------|-------------|
| `gamma_star` | Optimal risk-aversion parameter |
| `sharpe_train` | Annualised Sharpe on train set |
| `sharpe_test` | Annualised Sharpe on test set |
| `mean_train` | Mean train pnl|
| `mean_test` | Mean test pnl|

### `phase4_daily_pnl.csv`

| Column | Description |
|--------|-------------|
| `day` | Day index within split |
| `pnl` | Terminal wealth for that day ($) |
| `split` | `train` or `test` |
| `regime` | `ASAC` (with liquidation) or `AS-only` |

---

## Module Reference

### `data_loader.py`

```python
load_all_sheets(path)
    # Returns dict[ticker -> raw DataFrame], excluding GUIDE_SHEET

preprocess_train_data(df)
    # Parses DateTime, coerces OHLCV, computes mid and log-return, drops NaN rows

build_calibration_date_split(asset_data, n_train_days)
    # Returns (train_date_set, test_date_set) from the UNION of all asset dates
    # Critical: must NOT use the inner-join aligned index to avoid cutoff drift

build_price_matrices(asset_data, asset_names, train_date_set, test_date_set)
    # Inner-joins all assets on a common timestamp index
    # Returns (prices_all, highs_all, lows_all, timestamps,
    #          train_day_indices, test_day_indices)
```

### `calibration.py`

```python
estimate_k_A(df, delta_grid)       # Single-asset OLS
calibrate_all(asset_data, asset_names, train_date_set)
    # Returns (A_vec, k_vec, stats_dict)
calibrate_eta(all_sheets, asset_names, train_days_set, tau)
    # Calibrate temporary-impact coefficients η_i
```

### `risk_matrix.py`

```python
build_Gamma(A_vec, k_vec, Sigma)   # Returns Γ matrix (N×N)
build_covariance(prices_all, train_day_indices) # Computes Σ from log-returns on train bars only
```

### `as_bergault.py`

```python
compute_half_spreads(gamma, k_vec, Gamma)  # Returns hs (N,)
run_gamma_sweep(...)               # Returns results DataFrame
simulate(...)  # Returns (v_opt, x_path, cost)
```

### `almgren_chriss.py`
```python
solve_almgren_chriss(Q_trigger, N_liq, H_matrix, Sigma, gamma_star, tau)
    # Returns (v_opt, x_path, cost); TWAP fallback on solver failure
compute_twap_cost(Q_trigger, N_liq, H_matrix, Sigma, gamma_star)
     # Compute TWAP baseline cost for comparison
```

### `simulation.py`
```python
simulate_integrated(...)
    # Full integrated simulation: asymptotic-regime AS MM + AC liquidation
compute_stats(pnl) # Standard performance stats for a daily PnL array
```
---

## Key Design Decisions

**Train/test date split uses the union of all asset dates.**
`build_calibration_date_split` collects dates from every per-asset DataFrame before alignment. This guarantees that `N_TRAIN_DAYS` always refers to the same 93 calendar trading days regardless of which assets have sparse data on any given day. Using the inner-join aligned index would silently shift the cutoff.

**Σ uses `train_days_set` directly, not a secondary filter.**
This ensures the covariance matrix and the simulation are based on exactly the same set of bars, with no risk of divergence.

**`γ*` is selected on train Sharpe only.**
The gamma sweep evaluates both train and test Sharpe for diagnostic purposes, but `find_best_gamma` always returns the train-maximising value. Test performance is reported purely as an out-of-sample check — it does not influence model selection.

**AC solver falls back to TWAP.**
If `cvxpy` fails to converge (e.g., due to a near-singular covariance matrix at extreme inventory levels), `solve_almgren_chriss` returns a uniform TWAP schedule rather than crashing the simulation. The fallback cost is recorded as `inf` for post-hoc identification.

---

## References

1. Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3):217–224.
2. Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2):5–39.
3. Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1):77–91.
4. Bergault, P., Evangelista, D., Guéant, O., and Vieira, D. (2018). Closed-form approximations in multi-asset market making. *arXiv preprint arXiv:1810.04383*.
5. Guéant, O., Lehalle, C.-A., and Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics*, 7(4):477–507.
6. Guéant, O. (2017). *The Financial Mathematics of Market Liquidity: From Optimal Execution to Market Making*. Chapman & Hall/CRC.

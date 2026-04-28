# =============================================================================
# config.py — Central configuration for all simulation constants
# =============================================================================

# ── Data ─────────────────────────────────────────────────────────────────────
OHLCV_PATH   = "data/Cleaned Data.xlsx"
PARAMS_PATH  = "outputs/phase1_params.csv"
GAMMA_PATH   = "outputs/phase3_gamma_star.csv"

# ── Train / Test split ────────────────────────────────────────────────────────
N_TRAIN_DAYS = 93          # Sep 3 2025 → Jan 14 2026
RANDOM_SEED  = 42
GUIDE_SHEET = "Guide"
MIN_BARS_PER_DAY = 10
# ── Market-making constants ───────────────────────────────────────────────────
LOT_SIZE       = 100
FEE_PER_SHARE  = 0.001
MAX_INV_SHARES = 1000
MAX_SPREAD_PCT = 0.005     # 0.5% cap on CS spread used for A-proxy

# ── Phase 3: Sharpe grid search ───────────────────────────────────────────────
GAMMA_GRID_LOG_LO  = -3
GAMMA_GRID_LOG_HI  =  2
GAMMA_GRID_POINTS  =  25

# ── Phase 4: AC liquidation triggers ─────────────────────────────────────────
Q_MAX_TRIGGER  = 2000      # ||Q|| inventory-breach threshold
T_CLOSE        = 30        # minutes before session close for EOD flattening
AC_N_LIQ_MAX   = 30
AC_N_LIQ_MIN   = 5
TAU            = 1.0       # bar duration (minutes)

# ── Hard-coded γ* (used when CSV not available) ───────────────────────────────
GAMMA_STAR_DEFAULT = 0.825404185268019

# ── Robustness analysis configs ───────────────────────────────────────────
CONFIGS = [
    ("NoAC",            dict(ac_mode="off",      liq_solver="ac")),
    ("EOD_AC",          dict(ac_mode="eod_only", liq_solver="ac")),
    ("Inv_AC",          dict(ac_mode="inv_only", liq_solver="ac")),
    ("Dual_AC",         dict(ac_mode="dual",     liq_solver="ac")),
    ("Dual_TWAP",       dict(ac_mode="dual",     liq_solver="twap")),
    ("EOD_TWAP",        dict(ac_mode="eod_only", liq_solver="twap")),
]
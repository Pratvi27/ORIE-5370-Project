"""
data_loader.py — Load and preprocess OHLCV data from the Excel workbook.

"""

import numpy as np
import pandas as pd

from src.config import OHLCV_PATH, GUIDE_SHEET, N_TRAIN_DAYS, MIN_BARS_PER_DAY


# ── Raw loading ───────────────────────────────────────────────────────────────

def load_all_sheets(path: str = OHLCV_PATH) -> dict:
    all_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return {k: v for k, v in all_sheets.items() if k != GUIDE_SHEET}


# ── Per-asset preprocessing ───────────────────────────────────────────────────

def preprocess_train_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame()
    clean["datetime"] = pd.to_datetime(df["Date/Time"])
    clean["open"]     = pd.to_numeric(df["Open"],   errors="coerce")
    clean["high"]     = pd.to_numeric(df["High"],   errors="coerce")
    clean["low"]      = pd.to_numeric(df["Low"],    errors="coerce")
    clean["close"]    = pd.to_numeric(df["Close"],  errors="coerce")
    clean["volume"]   = pd.to_numeric(df["Volume"], errors="coerce")

    raw_spread = df['CS Spread $']
    if raw_spread.dtype == object:
        clean["csspread"] = pd.to_numeric(
            raw_spread.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    else:
        clean["csspread"] = pd.to_numeric(raw_spread, errors="coerce")

    clean["mid"]       = (clean["high"] + clean["low"]) / 2.0
    clean["logreturn"] = np.log(clean["close"] / clean["close"].shift(1))

    clean = clean.dropna(subset=["close", "high", "low", "volume", "csspread"])
    return clean.reset_index(drop=True)


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame()
    clean["datetime"] = pd.to_datetime(df["Date/Time"])
    clean["open"]     = pd.to_numeric(df["Open"],   errors="coerce")
    clean["high"]     = pd.to_numeric(df["High"],   errors="coerce")
    clean["low"]      = pd.to_numeric(df["Low"],    errors="coerce")
    clean["close"]    = pd.to_numeric(df["Close"],  errors="coerce")
    clean["volume"]   = pd.to_numeric(df["Volume"], errors="coerce")

    raw_spread = df['CS Spread $']
    if raw_spread.dtype == object:
        clean["csspread"] = pd.to_numeric(
            raw_spread.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    else:
        clean["csspread"] = pd.to_numeric(raw_spread, errors="coerce")

    clean["mid"]       = (clean["high"] + clean["low"]) / 2.0
    clean["logreturn"] = np.log(clean["close"] / clean["close"].shift(1))

    clean = clean.dropna(subset=["close", "high", "low"])
    return clean.reset_index(drop=True)

# ── Date split — MUST use union of all asset dates ────────────────────────────

def build_calibration_date_split(asset_data: dict,
                                  n_train_days: int = N_TRAIN_DAYS):
    """
    Build train/test date sets from the UNION of all per-asset dates.
    """
    all_dates_combined = sorted(set(
        d.date()
        for df in asset_data.values()
        for d in pd.to_datetime(df["datetime"])
    ))
    train_date_set = set(all_dates_combined[:n_train_days])
    test_date_set  = set(all_dates_combined[n_train_days:])
    return train_date_set, test_date_set


# ── Aligned price matrices ────────────────────────────────────────────────────

def build_price_matrices(asset_data: dict,
                          asset_names: list,
                          train_date_set: set,
                          test_date_set: set,
                          min_bars: int = MIN_BARS_PER_DAY):
    """
    Align all assets on a common timestamp index (inner join), then build
    train/test day-index lists by filtering the ALIGNED unique_days against
    the pre-computed train_date_set / test_date_set.

    Returns
    -------
    prices_all        : np.ndarray  (T, N)
    highs_all         : np.ndarray  (T, N)
    lows_all          : np.ndarray  (T, N)
    timestamps        : DatetimeIndex
    train_day_indices : list[np.ndarray]
    test_day_indices  : list[np.ndarray]
    """
    price_frames, high_frames, low_frames = {}, {}, {}

    for name in asset_names:
        df = asset_data[name].copy()
        df = df.set_index("datetime")
        price_frames[name] = df["close"].rename(name)
        high_frames[name]  = df["high"].rename(name)
        low_frames[name]   = df["low"].rename(name)

    prices_df = pd.concat(price_frames.values(), axis=1, join="inner").dropna().sort_index()
    highs_df  = pd.concat(high_frames.values(),  axis=1, join="inner")
    lows_df   = pd.concat(low_frames.values(),   axis=1, join="inner")

    common_idx = (
        prices_df.index
        .intersection(highs_df.index)
        .intersection(lows_df.index)
    )

    prices_all = prices_df.loc[common_idx].values
    highs_all  = highs_df.loc[common_idx].values
    lows_all   = lows_df.loc[common_idx].values
    timestamps = common_idx

    # day_indices: iterate unique_days from the ALIGNED matrix,
    # but filter membership against the union-derived date sets.
    dates       = pd.Series(timestamps).dt.date.values
    unique_days = sorted(set(dates))   

    def _day_indices(day_set: set):
        out = []
        for d in unique_days:
            if d not in day_set:
                continue
            mask = np.where(dates == d)[0]
            if len(mask) >= min_bars:
                out.append(mask)
        return out

    train_day_indices = _day_indices(train_date_set)
    test_day_indices  = _day_indices(test_date_set)

    return prices_all, highs_all, lows_all, timestamps, train_day_indices, test_day_indices
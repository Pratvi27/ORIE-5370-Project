# =============================================================================
# simulation.py — Integrated AS + AC simulation engine
# =============================================================================

import numpy as np
from src.config import LOT_SIZE, FEE_PER_SHARE, MAX_INV_SHARES
from src.as_bergault import compute_half_spreads, quote_bar
from src.almgren_chriss import solve_almgren_chriss

def simulate_ablation(gamma_star, A_vec, k_vec, Sigma, Gamma, H_matrix, eta_vec,
                      prices_all, day_indices, N,
                      Q_max=2000, T_close=30,
                      lot_size=None, fee_per_share=None,
                      max_inv_shares=None, seed=42,
                      ac_mode="dual",
                      liq_solver="ac"):
    """Integrated AS market maker + AC liquidation overlay.

    Parameters
    ----------
    ac_mode : {"off", "eod_only", "inv_only", "dual"}
        Which liquidation triggers are active.
    liq_solver : {"ac", "twap"}
        Schedule solver used when a trigger fires.
    """
    if lot_size       is None: lot_size       = LOT_SIZE
    if fee_per_share  is None: fee_per_share  = FEE_PER_SHARE
    if max_inv_shares is None: max_inv_shares = MAX_INV_SHARES

    rng          = np.random.RandomState(seed)
    sqrt_gamma   = np.sqrt(gamma_star)
    half_spreads = compute_half_spreads(gamma_star, k_vec, Gamma, N)
    tau          = 1.0   # 1-minute bar

    daily_pnl              = []
    daily_ac_triggers      = []
    daily_ac_costs         = []
    daily_fill_counts      = []
    daily_inventory_norms  = []
    sample_ac_trajectories = []

    for day_num, day_idx in enumerate(day_indices):
        day_prices = prices_all[day_idx]
        T_day      = len(day_prices)

        Q             = np.zeros(N)
        cash          = 0.0
        mode          = "AS"
        ac_triggers   = 0
        ac_total_cost = 0.0
        n_fills       = 0
        max_q_norm    = 0.0

        ac_schedule = None
        ac_step     = 0

        eod_fired      = False   # EOD triggers at most once per day
        inv_breach_arm = True    # rearms after ||Q|| < Q_max/2

        for t in range(T_day):
            S      = day_prices[t]
            q_norm = float(np.linalg.norm(Q))
            max_q_norm = max(max_q_norm, q_norm)

            # ------------------------------------------------------------------
            # TRIGGER CHECK
            # ------------------------------------------------------------------
            if ac_mode != "off" and mode == "AS" and q_norm > 0.5:
                trigger_reason = None
                allow_inv = ac_mode in ("inv_only", "dual")
                allow_eod = ac_mode in ("eod_only", "dual")

                if allow_inv and inv_breach_arm and q_norm > Q_max:
                    trigger_reason = "INVENTORY"
                    inv_breach_arm = False

                elif allow_eod and (not eod_fired) and (t >= T_day - T_close) and (q_norm > 1):
                    trigger_reason = "EOD"
                    eod_fired      = True

                if trigger_reason is not None:
                    mode        = "AC"
                    ac_triggers += 1
                    N_liq       = max(min(T_day - t, 30), 5)

                    if liq_solver == "twap":
                        ac_schedule = np.tile(Q / N_liq, (N_liq, 1))
                        ac_x_path   = np.zeros((N_liq + 1, N))
                        ac_x_path[0] = Q.copy()
                        for m_ in range(N_liq):
                            ac_x_path[m_ + 1] = ac_x_path[m_] - ac_schedule[m_]
                        ac_cost = 0.0
                        for m_ in range(N_liq):
                            ac_cost += ac_schedule[m_] @ H_matrix @ ac_schedule[m_]
                            ac_cost += gamma_star * ac_x_path[m_ + 1] @ Sigma @ ac_x_path[m_ + 1]
                    else:
                        ac_schedule, ac_x_path, ac_cost = solve_almgren_chriss(
                            Q, N_liq, H_matrix, Sigma, gamma_star)
                    ac_step = 0
                    if ac_cost < float("inf"):
                        ac_total_cost += ac_cost

                    if len(sample_ac_trajectories) < 5:
                        sample_ac_trajectories.append({
                            "day": day_num, "trigger_time": t,
                            "reason": trigger_reason,
                            "Q_trigger": Q.copy(),
                            "x_path": ac_x_path,
                            "v_schedule": ac_schedule,
                            "N_liq": N_liq,
                        })

            if (not inv_breach_arm) and q_norm < 0.5 * Q_max:
                inv_breach_arm = True

            # ------------------------------------------------------------------
            # EXECUTE: AC bar
            # ------------------------------------------------------------------
            if mode == "AC" and ac_schedule is not None and ac_step < len(ac_schedule):
                v_m = ac_schedule[ac_step]
                for i in range(N):
                    if abs(v_m[i]) > 0.01:
                        impact     = eta_vec[i] * abs(v_m[i]) / tau
                        exec_price = S[i] - impact if v_m[i] > 0 else S[i] + impact
                        cash      += exec_price * v_m[i]
                        Q[i]      -= v_m[i]
                ac_step += 1

                if ac_step >= len(ac_schedule) or np.linalg.norm(Q) < 0.5:
                    mode        = "AS"
                    ac_schedule = None

            # ------------------------------------------------------------------
            # EXECUTE: AS bar
            # ------------------------------------------------------------------
            else:
                if mode == "AC":          # schedule exhausted mid-day
                    mode        = "AS"
                    ac_schedule = None

                delta_cash, fills = quote_bar(
                    S, Q, half_spreads, sqrt_gamma, Gamma,
                    A_vec, k_vec, rng,
                    lot_size, fee_per_share, max_inv_shares,
                    p_cap=0.1)
                cash    += delta_cash
                n_fills += fills

        eod_liq_cost    = np.sum(np.abs(Q) * half_spreads)
        terminal_wealth = cash + Q @ day_prices[-1] - eod_liq_cost

        daily_pnl.append(terminal_wealth)
        daily_ac_triggers.append(ac_triggers)
        daily_ac_costs.append(ac_total_cost)
        daily_fill_counts.append(n_fills)
        daily_inventory_norms.append(max_q_norm)

        if (day_num + 1) % 20 == 0:
            print(f"  Day {day_num+1}/{len(day_indices)}: "
                  f"PnL=${terminal_wealth:,.0f}, AC triggers={ac_triggers}, "
                  f"fills={n_fills}, max||Q||={max_q_norm:.0f}")

    return {
        "daily_pnl":              np.array(daily_pnl),
        "daily_ac_triggers":      np.array(daily_ac_triggers),
        "daily_ac_costs":         np.array(daily_ac_costs),
        "daily_fill_counts":      np.array(daily_fill_counts),
        "daily_inventory_norms":  np.array(daily_inventory_norms),
        "sample_ac_trajectories": sample_ac_trajectories,
    }

def simulate_integrated(gamma_star, A_vec, k_vec, Sigma, Gamma, H_matrix, eta_vec,
                        prices_all, day_indices, N,
                        Q_max=2000, T_close=30,
                        lot_size=None, fee_per_share=None,
                        max_inv_shares=None, seed=42,
                        use_ac=True):
    """
    Full integrated simulation: asymptotic-regime AS market-making + AC liquidation.
    """
    if lot_size       is None: lot_size       = LOT_SIZE
    if fee_per_share  is None: fee_per_share  = FEE_PER_SHARE
    if max_inv_shares is None: max_inv_shares = MAX_INV_SHARES

    rng          = np.random.RandomState(seed)
    sqrt_gamma   = np.sqrt(gamma_star)
    half_spreads = compute_half_spreads(gamma_star, k_vec, Gamma, N)
    tau          = 1.0   # 1-minute bar

    daily_pnl              = []
    daily_ac_triggers      = []
    daily_ac_costs         = []
    daily_fill_counts      = []
    daily_inventory_norms  = []
    sample_ac_trajectories = []

    for day_num, day_idx in enumerate(day_indices):
        day_prices = prices_all[day_idx]
        T_day      = len(day_prices)

        Q             = np.zeros(N)
        cash          = 0.0
        mode          = "AS"
        ac_triggers   = 0
        ac_total_cost = 0.0
        n_fills       = 0
        max_q_norm    = 0.0

        ac_schedule = None
        ac_step     = 0

        eod_fired      = False  # EOD triggers at most once per day
        inv_breach_arm = True   # rearms after ||Q|| < Q_max/2

        for t in range(T_day):
            S      = day_prices[t]
            q_norm = float(np.linalg.norm(Q))
            max_q_norm = max(max_q_norm, q_norm)

            # ------------------------------------------------------------------
            # TRIGGER CHECK
            # ------------------------------------------------------------------
            if use_ac and mode == "AS" and q_norm > 0.5:
                trigger_reason = None

                if inv_breach_arm and q_norm > Q_max:
                    trigger_reason = "INVENTORY"
                    inv_breach_arm = False
                elif (not eod_fired) and (t >= T_day - T_close) and (q_norm > 1):
                    trigger_reason = "EOD"
                    eod_fired      = True

                if trigger_reason is not None:
                    mode        = "AC"
                    ac_triggers += 1
                    N_liq       = max(min(T_day - t, 30), 5)

                    ac_schedule, ac_x_path, ac_cost = solve_almgren_chriss(
                        Q, N_liq, H_matrix, Sigma, gamma_star)
                    ac_step = 0
                    if ac_cost < float("inf"):
                        ac_total_cost += ac_cost

                    if len(sample_ac_trajectories) < 5:
                        sample_ac_trajectories.append({
                            "day": day_num, "trigger_time": t,
                            "reason": trigger_reason,
                            "Q_trigger": Q.copy(),
                            "x_path": ac_x_path,
                            "v_schedule": ac_schedule,
                            "N_liq": N_liq,
                        })

            if (not inv_breach_arm) and q_norm < 0.5 * Q_max:
                inv_breach_arm = True

            # ------------------------------------------------------------------
            # EXECUTE: AC bar
            # ------------------------------------------------------------------
            if mode == "AC" and ac_schedule is not None and ac_step < len(ac_schedule):
                v_m = ac_schedule[ac_step]
                for i in range(N):
                    if abs(v_m[i]) > 0.01:
                        impact     = eta_vec[i] * abs(v_m[i]) / tau
                        exec_price = S[i] - impact if v_m[i] > 0 else S[i] + impact
                        cash      += exec_price * v_m[i]
                        Q[i]      -= v_m[i]
                ac_step += 1

                if ac_step >= len(ac_schedule) or np.linalg.norm(Q) < 0.5:
                    mode        = "AS"
                    ac_schedule = None

            # ------------------------------------------------------------------
            # EXECUTE: AS bar
            # ------------------------------------------------------------------
            else:
                if mode == "AC":          # schedule exhausted mid-day
                    mode        = "AS"
                    ac_schedule = None

                delta_cash, fills = quote_bar(
                    S, Q, half_spreads, sqrt_gamma, Gamma,
                    A_vec, k_vec, rng,
                    lot_size, fee_per_share, max_inv_shares,
                    p_cap=0.1)
                cash    += delta_cash
                n_fills += fills

        eod_liq_cost    = np.sum(np.abs(Q) * half_spreads)
        terminal_wealth = cash + Q @ day_prices[-1] - eod_liq_cost

        daily_pnl.append(terminal_wealth)
        daily_ac_triggers.append(ac_triggers)
        daily_ac_costs.append(ac_total_cost)
        daily_fill_counts.append(n_fills)
        daily_inventory_norms.append(max_q_norm)

        if (day_num + 1) % 20 == 0:
            print(f" Day {day_num+1}/{len(day_indices)}: "
                  f"PnL=${terminal_wealth:,.0f}, AC triggers={ac_triggers}, "
                  f"fills={n_fills}, max||Q||={max_q_norm:.0f}")

    return {
        "daily_pnl":              np.array(daily_pnl),
        "daily_ac_triggers":      np.array(daily_ac_triggers),
        "daily_ac_costs":         np.array(daily_ac_costs),
        "daily_fill_counts":      np.array(daily_fill_counts),
        "daily_inventory_norms":  np.array(daily_inventory_norms),
        "sample_ac_trajectories": sample_ac_trajectories,
    }


def compute_stats(pnl: np.ndarray) -> dict:
    """Standard performance statistics for a daily PnL array."""
    m, s = float(pnl.mean()), float(pnl.std())
    sh_d = m / s if s > 1e-12 else 0.0
    return dict(
        mean     = m,
        std      = s,
        sharpe_d = sh_d,
        sharpe_a = sh_d * np.sqrt(252),
        winrate  = float((pnl > 0).mean()) * 100,
        min_pnl  = float(pnl.min()),
        max_pnl  = float(pnl.max()),
    )
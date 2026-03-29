"""
QuantToolset — Black Swan Adversarial Stress-Testing Agent

Production-ready, headless quantitative finance diagnostic toolset.
Returns deterministic, structured JSON. No visual rendering libraries.
Strictly enforces t+1 execution to prevent lookahead bias.
Incorporates Optuna-based Walk-Forward Optimization (WFO).
Numba-accelerated backtester with fees, slippage & per-trade tax.
"""

import asyncio
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import optuna
from numba import njit

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ======================================================================
#  TAX REGIMES — country-specific capital gains rules
# ======================================================================
TAX_REGIMES = {
    "india": {
        "label": "India (Section 111A / 112A)",
        "stcg_threshold_days": 365,
        "stcg_rate": 0.20,    # 20% STCG on equity (post-2025 budget)
        "ltcg_rate": 0.125,   # 12.5% LTCG on equity
    },
    "us": {
        "label": "United States (Federal Max Bracket)",
        "stcg_threshold_days": 365,
        "stcg_rate": 0.37,    # Ordinary income max bracket
        "ltcg_rate": 0.20,    # Long-term max bracket
    },
    "uk": {
        "label": "United Kingdom (Higher Rate CGT)",
        "stcg_threshold_days": 365,
        "stcg_rate": 0.20,    # No ST/LT distinction in UK, but same rate
        "ltcg_rate": 0.20,
    },
    "none": {
        "label": "No Tax",
        "stcg_threshold_days": 0,
        "stcg_rate": 0.0,
        "ltcg_rate": 0.0,
    },
}


# ======================================================================
#  NUMBA KERNELS — compiled once, cached to disk
# ======================================================================
@njit(cache=True)
def _nb_backtest(close, signals, fees, slippage):
    """Numba-compiled backtest with t+1 execution, fees & slippage.

    Args:
        close:    1D float64 array of close prices.
        signals:  1D int64 array of +1/0/-1 signals.
        fees:     fractional fee per trade (e.g. 0.001 = 0.1%).
        slippage: fractional slippage per trade (e.g. 0.0005 = 0.05%).

    Returns:
        daily_returns, equity, positions, total_fees, total_slippage
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    daily_returns = np.zeros(n, dtype=np.float64)
    total_fees = 0.0
    total_slippage = 0.0

    # t+1 shift: signal at t → position at t+1
    for i in range(1, n):
        positions[i] = float(signals[i - 1])

    prev_pos = 0.0
    for i in range(1, n):
        price_ret = (close[i] - close[i - 1]) / close[i - 1]

        # Detect position change → apply costs
        pos_change = abs(positions[i] - prev_pos)
        if pos_change > 0.0:
            fee_cost = pos_change * fees
            slip_cost = pos_change * slippage
            total_fees += fee_cost
            total_slippage += slip_cost
            daily_returns[i] = positions[i] * price_ret - fee_cost - slip_cost
        else:
            daily_returns[i] = positions[i] * price_ret

        prev_pos = positions[i]

    # Build equity curve
    equity = np.empty(n, dtype=np.float64)
    equity[0] = 1.0
    for i in range(1, n):
        equity[i] = equity[i - 1] * (1.0 + daily_returns[i])

    return daily_returns, equity, positions, total_fees, total_slippage


@njit(cache=True)
def _nb_extract_trades(positions, close):
    """Extract discrete trades from position array.

    A trade opens when position changes from 0/opposite and closes when
    it returns to 0 or flips direction.

    Returns:
        entry_indices:  1D int64 array
        exit_indices:   1D int64 array
        pnl_per_trade:  1D float64 array (fractional PnL per trade)
        num_trades:     int64 (actual count — arrays are pre-allocated to max)
    """
    n = len(positions)
    # Pre-allocate to max possible trades
    max_trades = n // 2 + 1
    entry_indices = np.empty(max_trades, dtype=np.int64)
    exit_indices = np.empty(max_trades, dtype=np.int64)
    pnl_per_trade = np.empty(max_trades, dtype=np.float64)

    num_trades = 0
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    trade_dir = 0.0  # +1 long, -1 short

    for i in range(n):
        if not in_trade:
            if positions[i] != 0.0:
                in_trade = True
                entry_idx = i
                entry_price = close[i]
                trade_dir = positions[i]
        else:
            # Trade closes if position goes to 0 or flips
            if positions[i] == 0.0 or (positions[i] != 0.0 and positions[i] != trade_dir):
                exit_price = close[i]
                pnl = trade_dir * (exit_price - entry_price) / entry_price
                entry_indices[num_trades] = entry_idx
                exit_indices[num_trades] = i
                pnl_per_trade[num_trades] = pnl
                num_trades += 1

                # If position flipped (not zero), start new trade immediately
                if positions[i] != 0.0:
                    entry_idx = i
                    entry_price = close[i]
                    trade_dir = positions[i]
                else:
                    in_trade = False

    # Close any open trade at end of series
    if in_trade and n > 0:
        exit_price = close[n - 1]
        pnl = trade_dir * (exit_price - entry_price) / entry_price
        entry_indices[num_trades] = entry_idx
        exit_indices[num_trades] = n - 1
        pnl_per_trade[num_trades] = pnl
        num_trades += 1

    return entry_indices[:num_trades], exit_indices[:num_trades], pnl_per_trade[:num_trades], num_trades


class QuantToolset:
    """Headless adversarial stress-testing toolset for algorithmic trading strategies."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    #  PUBLIC TOOL — single entry point for the LLM
    # ------------------------------------------------------------------
    async def run_robustness_suite(
        self,
        ticker: str,
        strategy_type: str = "sma",
        strategy_params: str = "{}",
        period: str = "2y",
        iterations: int = 2000,
        drop_pct: float = 0.20,
        fees: float = 0.001,
        slippage: float = 0.0005,
        tax_regime: str = "none",
    ) -> str:
        """Run a full adversarial robustness suite on a trading strategy.

        Fetches OHLCV data, runs a Walk-Forward Optimization (WFO) backtest using Optuna
        and pandas-ta, computes baseline metrics on the Out-of-Sample (OOS) curve,
        profiles the data (Hurst + tail risk), and executes Monte Carlo stress tests.
        Applies transaction costs (fees + slippage) and per-trade holding-period tax.
        Returns a single structured JSON string.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA').
            strategy_type: pandas-ta supported strategy type ('sma', 'rsi', 'macd', 'bbands', etc).
            strategy_params: JSON string of param ranges. e.g. {"fast_period": {"low": 10, "high": 50}}.
            period: Data lookback period for yfinance (e.g. '1y', '2y', '5y').
            iterations: Number of Monte Carlo iterations (default 2000).
            drop_pct: Fraction of trades to drop in the Connectivity MC test (default 0.20).
            fees: Fractional fee per trade (default 0.001 = 0.1%).
            slippage: Fractional slippage per trade (default 0.0005 = 0.05%).
            tax_regime: Tax regime to apply ('india', 'us', 'uk', 'none').

        Returns:
            str: JSON string with WFO metrics, costs, tax, and adversarial diagnostics.
        """
        try:
            # Parse strategy_params from JSON string
            if isinstance(strategy_params, str):
                try:
                    params = json.loads(strategy_params) if strategy_params else {}
                except json.JSONDecodeError:
                    import ast
                    try:
                        params = ast.literal_eval(strategy_params) if strategy_params else {}
                    except (ValueError, SyntaxError):
                        params = {}
            else:
                params = strategy_params if strategy_params else {}

            # Validate tax regime
            regime_key = tax_regime.lower().strip() if isinstance(tax_regime, str) else "none"
            if regime_key not in TAX_REGIMES:
                regime_key = "none"

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                self._execute_suite,
                ticker,
                strategy_type,
                params,
                period,
                iterations,
                drop_pct,
                fees,
                slippage,
                regime_key,
            )
            return json.dumps(result, indent=2)

        except Exception as exc:
            logger.error("run_robustness_suite failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    #  PRIVATE — orchestrator (runs in thread pool to keep async happy)
    # ------------------------------------------------------------------
    def _execute_suite(
        self,
        ticker: str,
        strategy_type: str,
        params: dict,
        period: str,
        iterations: int,
        drop_pct: float,
        fees: float,
        slippage: float,
        tax_regime: str,
    ) -> dict:
        # 1. Fetch & clean data
        df = self._fetch_data(ticker, period)
        if df.empty or len(df) < 100:
            raise ValueError(
                f"Insufficient data for {ticker} with period={period}. "
                f"Got {len(df)} rows; need at least 100 for WFO."
            )

        # 2. Walk-Forward Optimization (WFO)
        wfo_result = self._run_wfo(
            df, strategy_type, params,
            wfo_train_days=180, wfo_test_days=30, optuna_trials=30,
            penalty_factor=2.0, fees=fees, slippage=slippage, tax_regime=tax_regime,
        )
        wfo_daily_returns = wfo_result["daily_returns"]
        optimal_params_list = wfo_result["optimal_params"]
        cost_summary = wfo_result["cost_summary"]

        # Build portfolio object for baseline metrics using OOS curve
        oos_equity = np.cumprod(1 + wfo_daily_returns)
        portfolio = {
            "daily_returns": wfo_daily_returns,
            "equity_curve": oos_equity,
            "positions": np.sign(wfo_daily_returns),
        }

        # 3. Baseline metrics
        baseline = self._compute_baseline_metrics(portfolio)

        # 4. Data profiling & News
        close_series = np.asarray(df["Close"])
        daily_returns = portfolio["daily_returns"]
        hurst = self._compute_hurst_exponent(close_series)
        tail = self._compute_tail_risk(daily_returns)

        # Fetch latest news
        try:
            news_items = yf.Ticker(ticker).news
            recent_news = [item.get("title", "") for item in news_items[:5]] if news_items else []
        except Exception:
            recent_news = []

        # 5. Adversarial lab
        trades_arr = daily_returns[daily_returns != 0.0]
        if len(trades_arr) < 10:
            trades_arr = daily_returns

        daily_sequences = self._build_daily_sequences(daily_returns)

        connectivity = self._stress_connectivity_mc(trades_arr, iterations, drop_pct)
        trade_seq = self._stress_shuffle_trades_mc(trades_arr, iterations)
        day_seq = self._stress_shuffle_days_mc(
            daily_sequences, baseline["max_drawdown_pct"], iterations
        )
        gbm = self._stress_gaussian_gbm_mc(
            daily_returns,
            baseline["max_drawdown_pct"],
            baseline["net_return_pct"] / 100.0,
            iterations,
        )

        # 6. Build execution costs block
        regime_info = TAX_REGIMES[tax_regime]
        execution_costs = {
            "fees_per_trade_pct": round(fees * 100, 4),
            "slippage_pct": round(slippage * 100, 4),
            "tax_regime": tax_regime,
            "tax_regime_label": regime_info["label"],
            "tax_rules_applied": {
                "stcg_threshold_days": regime_info["stcg_threshold_days"],
                "stcg_rate_pct": round(regime_info["stcg_rate"] * 100, 2),
                "ltcg_rate_pct": round(regime_info["ltcg_rate"] * 100, 2),
            },
            "total_fees_deducted": round(float(cost_summary["total_fees"]), 6),
            "total_slippage_cost": round(float(cost_summary["total_slippage"]), 6),
            "total_trades": int(cost_summary["total_trades"]),
            "short_term_trades": int(cost_summary["short_term_trades"]),
            "long_term_trades": int(cost_summary["long_term_trades"]),
            "total_tax_deducted": round(float(cost_summary["total_tax"]), 6),
        }

        return {
            "ticker": ticker.upper(),
            "strategy": {"type": strategy_type, "params": params},
            "period": period,
            "execution_costs": execution_costs,
            "data_profile": {
                "hurst_exponent": round(hurst, 4),
                "tail_risk": tail,
                "recent_news": recent_news,
            },
            "baseline_backtest": baseline,
            "wfo_optimal_params_history": optimal_params_list,
            "adversarial_lab": {
                "connectivity_mc": connectivity,
                "trade_sequence_mc": trade_seq,
                "day_sequence_mc": day_seq,
                "gaussian_gbm_mc": gbm,
            },
        }

    # ------------------------------------------------------------------
    #  DATA INGESTION
    # ------------------------------------------------------------------
    def _fetch_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch OHLCV data via yfinance, forward-fill gaps."""
        # Clean and map natural language periods to yfinance codes
        p = str(period).lower().strip()
        mapper = {
            "1 year": "1y", "2 years": "2y", "3 years": "3y", "5 years": "5y",
            "10 years": "10y", "1 month": "1mo", "3 months": "3mo", 
            "6 months": "6mo", "year to date": "ytd", "max": "max"
        }
        # Handle plurals and simple abbreviations
        p = p.replace("years", "y").replace("year", "y")
        p = p.replace("months", "mo").replace("month", "mo")
        p = p.replace("days", "d").replace("day", "d")
        p = p.replace(" ", "")

        data = yf.download(ticker, period=p, auto_adjust=True, progress=False)
        if data is None or data.empty:
            # Try one fallback with original string just in case
            if p != period:
                data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            
            if data is None or data.empty:
                return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.ffill().dropna()
        return data

    # ------------------------------------------------------------------
    #  WALK-FORWARD OPTIMIZATION (WFO)
    # ------------------------------------------------------------------
    def _run_wfo(self, df: pd.DataFrame, strategy_type: str, param_ranges: dict,
                 wfo_train_days=180, wfo_test_days=30, optuna_trials=30,
                 penalty_factor=2.0, fees=0.001, slippage=0.0005, tax_regime="none"):
        """Runs Walk-Forward Optimization across the dataset using Optuna."""
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        dates = df.index
        start_date = dates[0]
        end_date = dates[-1]

        current_train_start = start_date

        oos_returns_list = []
        optimal_params_list = []

        # Aggregate cost tracking across all WFO windows
        agg_total_fees = 0.0
        agg_total_slippage = 0.0
        agg_total_tax = 0.0
        agg_total_trades = 0
        agg_st_trades = 0
        agg_lt_trades = 0

        while True:
            train_end = current_train_start + pd.Timedelta(days=wfo_train_days)
            test_end = train_end + pd.Timedelta(days=wfo_test_days)

            if test_end > end_date:
                if train_end < end_date and (end_date - train_end).days > 5:
                    test_end = end_date
                else:
                    break

            train_df = df[(df.index >= current_train_start) & (df.index < train_end)]
            test_df = df[(df.index >= train_end) & (df.index < test_end)]

            if len(train_df) < 30 or len(test_df) < 5:
                current_train_start += pd.Timedelta(days=wfo_test_days)
                continue

            def objective(trial):
                opt_params = {}
                for k, v in param_ranges.items():
                    if isinstance(v, dict) and "low" in v and "high" in v:
                        if isinstance(v["low"], int) and isinstance(v["high"], int):
                            opt_params[k] = trial.suggest_int(k, v["low"], v["high"])
                        else:
                            opt_params[k] = trial.suggest_float(k, float(v["low"]), float(v["high"]))
                    elif isinstance(v, list):
                        opt_params[k] = trial.suggest_categorical(k, v)
                    else:
                        opt_params[k] = v

                try:
                    signals = self._generate_signals_dynamic(train_df, strategy_type, opt_params)
                    res = self._run_backtest(train_df, signals, fees=fees, slippage=slippage, tax_regime="none")
                    daily = res['daily_returns']
                    safe_daily = np.maximum(daily, -0.9999)
                    log_rets = np.log1p(safe_daily)
                    mean_lr = np.mean(log_rets)
                    std_lr = np.std(log_rets)
                    sharpe = (mean_lr / std_lr) * np.sqrt(252) if std_lr != 0 else 0.0
                    max_dd = self._get_max_drawdown(daily)

                    score = sharpe - (max_dd * penalty_factor)
                    return score
                except Exception:
                    return -999.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=optuna_trials)

            best_params = study.best_params

            for k, v in param_ranges.items():
                if k not in best_params:
                    if isinstance(v, dict) and "low" in v and "high" in v:
                        best_params[k] = v["low"]
                    elif isinstance(v, list):
                        best_params[k] = v[0] if len(v) > 0 else v
                    else:
                        best_params[k] = v

            combined_df = df[(df.index >= current_train_start) & (df.index < test_end)]
            oos_signals = self._generate_signals_dynamic(combined_df, strategy_type, best_params)

            res_comb = self._run_backtest(combined_df, oos_signals, fees=fees, slippage=slippage, tax_regime=tax_regime)
            daily_returns_comb = res_comb['daily_returns']

            test_mask = np.asarray(combined_df.index >= train_end)
            oos_daily = daily_returns_comb[test_mask]

            oos_returns_list.append(oos_daily)

            # Accumulate costs from this window
            agg_total_fees += res_comb["total_fees_paid"]
            agg_total_slippage += res_comb["total_slippage_cost"]
            agg_total_tax += res_comb["total_tax_deducted"]
            agg_total_trades += res_comb["total_trades"]
            agg_st_trades += res_comb["short_term_trades"]
            agg_lt_trades += res_comb["long_term_trades"]

            optimal_params_list.append({
                "window_start": current_train_start.strftime('%Y-%m-%d'),
                "train_end": train_end.strftime('%Y-%m-%d'),
                "test_end": test_end.strftime('%Y-%m-%d'),
                "params": best_params
            })

            current_train_start += pd.Timedelta(days=wfo_test_days)

        if not oos_returns_list:
            logger.warning("Data too short for WFO splits. Executing default backtest.")
            default_params = {}
            for k, v in param_ranges.items():
                if isinstance(v, dict) and "low" in v and "high" in v:
                    default_params[k] = v["low"]
                elif isinstance(v, list):
                    default_params[k] = v[0] if len(v) > 0 else None
                else:
                    default_params[k] = v
            sig = self._generate_signals_dynamic(df, strategy_type, default_params)
            res = self._run_backtest(df, sig, fees=fees, slippage=slippage, tax_regime=tax_regime)
            return {
                "daily_returns": res['daily_returns'],
                "optimal_params": [{"fallback": "Not enough data for WFO, static backtest executed."}],
                "cost_summary": {
                    "total_fees": res["total_fees_paid"],
                    "total_slippage": res["total_slippage_cost"],
                    "total_tax": res["total_tax_deducted"],
                    "total_trades": res["total_trades"],
                    "short_term_trades": res["short_term_trades"],
                    "long_term_trades": res["long_term_trades"],
                },
            }

        full_oos_returns = np.concatenate(oos_returns_list)
        return {
            "daily_returns": full_oos_returns,
            "optimal_params": optimal_params_list,
            "cost_summary": {
                "total_fees": agg_total_fees,
                "total_slippage": agg_total_slippage,
                "total_tax": agg_total_tax,
                "total_trades": agg_total_trades,
                "short_term_trades": agg_st_trades,
                "long_term_trades": agg_lt_trades,
            },
        }

    # ------------------------------------------------------------------
    #  SIGNAL GENERATION
    # ------------------------------------------------------------------
    def _generate_signals_dynamic(
        self, df: pd.DataFrame, strategy_type: str, params: dict
    ) -> pd.Series:
        """Generate +1 (long) / 0 (flat) / -1 (short) signals dynamically using pandas-ta."""
        close = df["Close"]
        strategy_type = strategy_type.lower().strip()

        # Moving Averages
        if strategy_type in ["sma", "ema", "wma", "vwma", "hma", "rma", "dema", "tema"]:
            fast = int(params.get("fast_period", 10))
            slow = int(params.get("slow_period", 50))
            if fast >= slow:
                return pd.Series(0, index=df.index)
            fast_ma = getattr(ta, strategy_type)(close, length=fast)
            slow_ma = getattr(ta, strategy_type)(close, length=slow)
            signal = pd.Series(0, index=df.index)
            signal[fast_ma > slow_ma] = 1
            signal[fast_ma <= slow_ma] = -1

        # Oscillators
        elif strategy_type in ["rsi", "cci", "mfi", "stoch"]:
            per = int(params.get("length", 14))
            ob = float(params.get("overbought", 70))
            os_val = float(params.get("oversold", 30))
            ind_val = getattr(ta, strategy_type)(df["High"], df["Low"], close, length=per) if strategy_type in ["cci", "mfi"] else getattr(ta, strategy_type)(close, length=per)
            if isinstance(ind_val, pd.DataFrame):
                ind_val = ind_val.iloc[:, 0]
            signal = pd.Series(0, index=df.index)
            signal[ind_val < os_val] = 1   # oversold → buy
            signal[ind_val > ob] = -1      # overbought → sell

        # MACD
        elif strategy_type in ["macd", "ppo", "apo"]:
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            sig_p = int(params.get("signal", 9))
            if fast >= slow:
                return pd.Series(0, index=df.index)
            ind_df = getattr(ta, strategy_type)(close, fast=fast, slow=slow, signal=sig_p)
            if ind_df is None or len(ind_df.columns) < 3:
                return pd.Series(0, index=df.index)
            macd_col = next((c for c in ind_df.columns if c.startswith(("MACD_", "PPO_", "APO_"))), None)
            sig_col = next((c for c in ind_df.columns if c.startswith(("MACDs_", "PPOs_", "APOs_"))), None)
            if macd_col is None or sig_col is None:
                return pd.Series(0, index=df.index)
            macd_line = ind_df[macd_col]
            signal_line = ind_df[sig_col]
            signal = pd.Series(0, index=df.index)
            signal[macd_line > signal_line] = 1
            signal[macd_line <= signal_line] = -1

        # Bollinger Bands
        elif strategy_type in ["bbands", "kc", "dc"]:
            per = int(params.get("length", 20))
            std_dev = float(params.get("std", 2.0))
            ind_df = getattr(ta, strategy_type)(close, length=per, std=std_dev)
            if ind_df is None or len(ind_df.columns) < 3:
                return pd.Series(0, index=df.index)
            lower_col = next((c for c in ind_df.columns if "L" in c.split("_")[0]), None)
            upper_col = next((c for c in ind_df.columns if "U" in c.split("_")[0]), None)
            if lower_col is None or upper_col is None:
                return pd.Series(0, index=df.index)
            lower = ind_df[lower_col]
            upper = ind_df[upper_col]
            signal = pd.Series(0, index=df.index)
            signal[close < lower] = 1   # price below lower band → buy
            signal[close > upper] = -1  # price above upper band → sell

        else:
            raise ValueError(
                f"Unknown strategy_type '{strategy_type}'. "
                "Supported: sma, ema, rsi, macd, bbands, etc."
            )

        signal = signal.fillna(0).astype(int)
        return signal

    # ------------------------------------------------------------------
    #  BACKTEST — Numba-accelerated with fees, slippage & tax
    # ------------------------------------------------------------------
    def _run_backtest(self, df: pd.DataFrame, signals: pd.Series,
                      fees: float = 0.001, slippage: float = 0.0005,
                      tax_regime: str = "none") -> dict:
        """Numba-accelerated backtest with t+1 execution, fees, slippage & per-trade tax."""
        close = np.asarray(df["Close"], dtype=np.float64)
        sig_arr = np.asarray(signals, dtype=np.int64)

        # Run Numba-compiled backtest core
        daily_returns, equity, positions, total_fees, total_slippage = \
            _nb_backtest(close, sig_arr, fees, slippage)

        # Apply per-trade holding-period tax
        tax_result = self._apply_holding_period_tax(
            daily_returns, positions, df.index, close, tax_regime
        )

        post_tax_returns = tax_result["post_tax_returns"]

        # Rebuild equity from post-tax returns
        post_tax_equity = np.empty(len(post_tax_returns), dtype=np.float64)
        post_tax_equity[0] = 1.0
        for i in range(1, len(post_tax_returns)):
            post_tax_equity[i] = post_tax_equity[i - 1] * (1.0 + post_tax_returns[i])

        return {
            "daily_returns": post_tax_returns,
            "equity_curve": post_tax_equity,
            "positions": positions,
            "total_fees_paid": float(total_fees),
            "total_slippage_cost": float(total_slippage),
            "total_tax_deducted": float(tax_result["total_tax"]),
            "total_trades": int(tax_result["total_trades"]),
            "short_term_trades": int(tax_result["short_term_trades"]),
            "long_term_trades": int(tax_result["long_term_trades"]),
        }

    # ------------------------------------------------------------------
    #  HOLDING-PERIOD TAX ENGINE
    # ------------------------------------------------------------------
    def _apply_holding_period_tax(
        self, daily_returns: np.ndarray, positions: np.ndarray,
        dates: pd.DatetimeIndex, close: np.ndarray, tax_regime: str
    ) -> dict:
        """Apply per-trade tax based on holding period and country-specific regime.

        Extracts trades via Numba, computes holding days from the date index,
        applies STCG or LTCG rate on winning trades, distributes tax deduction
        onto the exit day's return.
        """
        regime = TAX_REGIMES.get(tax_regime, TAX_REGIMES["none"])
        stcg_threshold = regime["stcg_threshold_days"]
        stcg_rate = regime["stcg_rate"]
        ltcg_rate = regime["ltcg_rate"]

        # Extract trades using Numba kernel
        entry_indices, exit_indices, pnl_per_trade, num_trades = \
            _nb_extract_trades(positions, close)

        post_tax_returns = daily_returns.copy()
        total_tax = 0.0
        short_term_count = 0
        long_term_count = 0

        if num_trades > 0 and (stcg_rate > 0 or ltcg_rate > 0):
            # Convert dates to numpy datetime64 for fast day computation
            dates_arr = np.asarray(dates)

            for t in range(num_trades):
                pnl = pnl_per_trade[t]
                if pnl <= 0:
                    # No tax on losing trades
                    continue

                entry_dt = dates_arr[entry_indices[t]]
                exit_dt = dates_arr[exit_indices[t]]
                holding_days = int((exit_dt - entry_dt) / np.timedelta64(1, 'D'))

                if holding_days <= stcg_threshold:
                    tax_amount = pnl * stcg_rate
                    short_term_count += 1
                else:
                    tax_amount = pnl * ltcg_rate
                    long_term_count += 1

                total_tax += tax_amount

                # Distribute tax deduction onto the exit day return
                exit_idx = exit_indices[t]
                post_tax_returns[exit_idx] -= tax_amount
        else:
            # Count trades even if no tax
            for t in range(num_trades):
                entry_dt = np.asarray(dates)[entry_indices[t]]
                exit_dt = np.asarray(dates)[exit_indices[t]]
                holding_days = int((exit_dt - entry_dt) / np.timedelta64(1, 'D'))
                if holding_days <= stcg_threshold:
                    short_term_count += 1
                else:
                    long_term_count += 1

        return {
            "post_tax_returns": post_tax_returns,
            "total_tax": total_tax,
            "total_trades": int(num_trades),
            "short_term_trades": short_term_count,
            "long_term_trades": long_term_count,
        }

    # ------------------------------------------------------------------
    #  BASELINE METRICS
    # ------------------------------------------------------------------
    def _compute_baseline_metrics(self, portfolio: dict) -> dict:
        daily = portfolio["daily_returns"]
        equity = portfolio["equity_curve"]

        total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0
        n_days = len(daily)
        ann_return = (equity[-1] ** (252 / max(n_days, 1))) - 1.0 if n_days > 0 else 0.0

        max_dd = self._get_max_drawdown(daily)

        safe_daily = np.maximum(daily, -0.9999)
        log_rets = np.log1p(safe_daily)
        mean_lr = np.mean(log_rets)
        std_lr = np.std(log_rets)
        sharpe = (mean_lr / std_lr) * np.sqrt(252) if std_lr != 0 else 0.0

        downside = log_rets[log_rets < 0]
        if len(downside) > 1:
            down_std = np.std(downside, ddof=1)
            sortino = (mean_lr / down_std) * np.sqrt(252) if down_std != 0 else 0.0
        else:
            sortino = 0.0

        calmar = (ann_return / max_dd) if max_dd > 0 else 0.0

        trades = daily[daily != 0]
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        return {
            "net_return_pct": round(total_return * 100, 2),
            "annualised_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
        }

    # ------------------------------------------------------------------
    #  HURST EXPONENT  (R/S analysis)
    # ------------------------------------------------------------------
    def _compute_hurst_exponent(self, series: np.ndarray) -> float:
        """Compute Hurst exponent using rescaled range (R/S) analysis."""
        if len(series) < 20:
            return 0.5

        returns = np.diff(np.log(series))
        n = len(returns)

        max_k = min(n // 2, 512)
        sizes = []
        k = 8
        while k <= max_k:
            sizes.append(k)
            k = int(k * 1.5)
        if len(sizes) < 2:
            return 0.5

        rs_values = []
        for s in sizes:
            n_subs = n // s
            rs_list = []
            for i in range(n_subs):
                sub = returns[i * s : (i + 1) * s]
                mean_sub = np.mean(sub)
                devs = np.cumsum(sub - mean_sub)
                r = np.max(devs) - np.min(devs)
                std = np.std(sub, ddof=1)
                if std > 0:
                    rs_list.append(r / std)
            if rs_list:
                rs_values.append((np.log(s), np.log(np.mean(rs_list))))

        if len(rs_values) < 2:
            return 0.5

        log_n = np.array([v[0] for v in rs_values])
        log_rs = np.array([v[1] for v in rs_values])
        hurst = np.polyfit(log_n, log_rs, 1)[0]
        return float(np.clip(hurst, 0.0, 1.0))

    # ------------------------------------------------------------------
    #  TAIL RISK
    # ------------------------------------------------------------------
    def _compute_tail_risk(self, returns: np.ndarray) -> dict:
        from scipy import stats as sp_stats

        if len(returns) < 5:
            return {"skewness": 0.0, "kurtosis": 3.0, "var_95": 0.0, "cvar_95": 0.0}

        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns, fisher=False))
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= var_95])) if np.any(returns <= var_95) else var_95

        return {
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
        }

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _get_max_drawdown(returns_array: np.ndarray) -> float:
        if len(returns_array) == 0:
            return 0.0
        equity = np.cumprod(1 + returns_array)
        hwm = np.maximum.accumulate(equity)
        dd = (hwm - equity) / hwm
        return float(np.max(dd))

    @staticmethod
    def _build_daily_sequences(daily_returns: np.ndarray):
        return [np.array([r]) for r in daily_returns]

    # ------------------------------------------------------------------
    #  MONTE CARLO — Connectivity
    # ------------------------------------------------------------------
    def _stress_connectivity_mc(
        self, trades_arr: np.ndarray, iterations: int, drop_pct: float
    ) -> dict:
        n_keep = int(len(trades_arr) * (1 - drop_pct))
        if n_keep < 1:
            n_keep = 1

        sims = []
        for _ in range(iterations):
            sample = np.random.choice(trades_arr, n_keep, replace=False)
            sims.append((np.prod(1 + sample) - 1) * 100)

        original_ret = (np.prod(1 + trades_arr) - 1) * 100
        arr = np.array(sims)

        return {
            "original_return_pct": round(float(original_ret), 4),
            "avg_sim_return_pct": round(float(np.mean(arr)), 4),
            "p5_return_pct": round(float(np.percentile(arr, 5)), 4),
            "p95_return_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Trade Sequence (shuffle trades)
    # ------------------------------------------------------------------
    def _stress_shuffle_trades_mc(
        self, trades_arr: np.ndarray, iterations: int
    ) -> dict:
        actual_dd = self._get_max_drawdown(trades_arr) * 100
        temp = trades_arr.copy()
        sims = []
        for _ in range(iterations):
            np.random.shuffle(temp)
            sims.append(self._get_max_drawdown(temp) * 100)

        arr = np.array(sims)
        return {
            "actual_dd_pct": round(float(actual_dd), 4),
            "avg_sim_dd_pct": round(float(np.mean(arr)), 4),
            "var95_dd_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Day Sequence (shuffle days)
    # ------------------------------------------------------------------
    def _stress_shuffle_days_mc(
        self, daily_sequences: list, original_dd_pct: float, iterations: int
    ) -> dict:
        n = len(daily_sequences)
        idx = np.arange(n)
        sims = []
        for _ in range(iterations):
            np.random.shuffle(idx)
            shuffled = np.concatenate([daily_sequences[i] for i in idx])
            sims.append(self._get_max_drawdown(shuffled) * 100)

        arr = np.array(sims)
        return {
            "actual_dd_pct": round(float(original_dd_pct), 4),
            "avg_sim_dd_pct": round(float(np.mean(arr)), 4),
            "var95_dd_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Gaussian GBM
    # ------------------------------------------------------------------
    def _stress_gaussian_gbm_mc(
        self,
        daily_returns: np.ndarray,
        current_dd_pct: float,
        actual_total_ret: float,
        iterations: int,
    ) -> dict:
        safe_returns = np.maximum(daily_returns, -0.9999)
        log_rets = np.log1p(safe_returns)
        mu = np.mean(log_rets)
        sigma = np.std(log_rets)
        n_days = len(daily_returns)

        if sigma == 0 or n_days == 0:
            return {
                "risk": {"actual_dd_pct": current_dd_pct, "avg_sim_dd_pct": 0, "var95_dd_pct": 0},
                "reward": {"actual_return_pct": round(actual_total_ret * 100, 4), "avg_sim_return_pct": 0, "worst5_return_pct": 0, "best5_return_pct": 0},
            }

        Z = np.random.normal(0, 1, (iterations, n_days))
        sim_log_steps = mu + sigma * Z
        sim_log_curves = np.cumsum(sim_log_steps, axis=1)
        sim_curves = np.exp(sim_log_curves)

        hwm = np.maximum.accumulate(sim_curves, axis=1)
        drawdowns = (hwm - sim_curves) / hwm
        sim_mdds = np.max(drawdowns, axis=1) * 100

        sim_total_rets = (sim_curves[:, -1] - 1) * 100

        dd_95 = np.percentile(sim_mdds, 95)
        dd_avg = np.mean(sim_mdds)

        ret_avg = np.mean(sim_total_rets)
        ret_05 = np.percentile(sim_total_rets, 5)
        ret_95 = np.percentile(sim_total_rets, 95)

        actual_ret_pct = actual_total_ret * 100

        return {
            "risk": {
                "actual_dd_pct": round(float(current_dd_pct), 4),
                "avg_sim_dd_pct": round(float(dd_avg), 4),
                "var95_dd_pct": round(float(dd_95), 4),
            },
            "reward": {
                "actual_return_pct": round(float(actual_ret_pct), 4),
                "avg_sim_return_pct": round(float(ret_avg), 4),
                "worst5_return_pct": round(float(ret_05), 4),
                "best5_return_pct": round(float(ret_95), 4),
            },
        }

    # ------------------------------------------------------------------
    #  TOOL REGISTRATION
    # ------------------------------------------------------------------
    def get_tools(self) -> dict[str, Any]:
        return {
            "run_robustness_suite": self,
        }

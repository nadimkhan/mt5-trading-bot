"""
Online parameter optimizer for scalping/trend/regime strategies.

Concept: Get last 5-6 hours of M5 data and try different parameter combinations
to find what would have been profitable in that recent window. Use the winning
parameters until market regime changes, then re-optimize.

This is fundamentally different from ML search:
- ML search: 1 year data, GA, looking for universal edge
- Optimizer: 6 hour data, brute force, looking for today's edge
"""
import itertools
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Parameter ranges per strategy. Keep small for fast brute force.
PARAM_RANGES = {
    "scalp": {
        "rsi_period": [7, 10, 14, 21],
        "rsi_overbought": [70, 75, 80],
        "rsi_oversold": [20, 25, 30],
        "ema_fast": [5, 8, 12],
        "ema_slow": [20, 30, 50],
        "atr_sl_multiplier": [1.0, 1.5, 2.0],
        "atr_tp_multiplier": [1.5, 2.0, 3.0],
        "volume_min_multiplier": [0.8, 1.0, 1.5],
    },
    "trend": {
        "ema_fast": [5, 8, 12, 20],
        "ema_slow": [20, 30, 50, 100],
        "adx_threshold": [15, 20, 25, 30],
        "macd_signal": [5, 9, 15],
        "atr_sl_multiplier": [1.0, 1.5, 2.0],
        "atr_tp_multiplier": [2.0, 3.0, 4.0],
    },
    "regime": {
        "bull_pullback_low": [30, 35, 40],
        "bull_pullback_high": [55, 60, 65],
        "bear_pullback_low": [35, 40, 45],
        "bear_pullback_high": [60, 65, 70],
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "adx_threshold": [20, 25, 30],
    },
}

# Per-symbol pip/size assumptions (for backtester P&L calculation)
SYMBOL_PROFILES = {
    "EURUSD": {"point": 0.0001, "digits": 5, "pip_value_per_lot": 10.0, "lot_size": 0.01},
    "GBPUSD": {"point": 0.0001, "digits": 5, "pip_value_per_lot": 10.0, "lot_size": 0.01},
    "USDJPY": {"point": 0.01,   "digits": 3, "pip_value_per_lot": 10.0, "lot_size": 0.01},
    "XAUUSD": {"point": 0.01,   "digits": 2, "pip_value_per_lot": 1.0,  "lot_size": 0.01},
    "BTCUSDT": {"point": 1.0,   "digits": 2, "pip_value_per_lot": 1.0,  "lot_size": 0.01},
    "BRNUSD": {"point": 0.01,   "digits": 2, "pip_value_per_lot": 1.0,  "lot_size": 0.01},
}


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI calculation returning array same length as closes (NaN-padded)."""
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    # Wilder smoothing
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    if len(gains) < period:
        return np.full(len(closes), np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = np.nan
    return rsi


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA calculation."""
    ema = np.full(len(values), np.nan)
    if len(values) < period:
        return ema
    k = 2 / (period + 1)
    ema[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR calculation."""
    if len(closes) < 2:
        return np.zeros(len(closes))
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, -DI calculation."""
    if len(closes) < period * 2:
        n = len(closes)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    plus_dm = np.zeros(len(closes))
    minus_dm = np.zeros(len(closes))
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    # Wilder smoothing
    atr_arr = np.zeros(len(closes))
    smooth_plus_dm = np.zeros(len(closes))
    smooth_minus_dm = np.zeros(len(closes))
    atr_arr[period] = np.sum(tr[1:period + 1])
    smooth_plus_dm[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, len(closes)):
        atr_arr[i] = atr_arr[i - 1] - (atr_arr[i - 1] / period) + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]
    plus_di = 100 * np.divide(smooth_plus_dm, atr_arr, out=np.zeros_like(smooth_plus_dm), where=atr_arr > 0)
    minus_di = 100 * np.divide(smooth_minus_dm, atr_arr, out=np.zeros_like(smooth_minus_dm), where=atr_arr > 0)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = np.zeros(len(closes))
    adx[period * 2] = np.mean(dx[period:period * 2 + 1])
    for i in range(period * 2 + 1, len(closes)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di


def _bbands(closes: np.ndarray, period: int, num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: upper, middle, lower."""
    n = len(closes)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window)
        middle[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return upper, middle, lower


def _score(pnl: float, win_rate: float, trades: int, max_dd: float) -> float:
    """Score a parameter combination. Higher is better.

    Strict scoring: requires minimum 5 trades AND win rate > 40% to be considered.
    """
    if trades < 5 or pnl <= 0:
        return 0.0
    if win_rate < 0.40:
        return 0.0  # Filter out low WR even with positive PnL (likely noise)
    # Reward: P&L scaled by win rate, penalize high drawdown
    wr_factor = max(0.0, win_rate - 0.30) / 0.70  # 0 at WR=30%, 1 at WR=100%
    trade_factor = min(1.0, trades / 10.0)  # Full credit at 10+ trades
    dd_penalty = max(0.0, 1.0 - max_dd / 30.0)  # 0 at DD>=30%
    return pnl * wr_factor * trade_factor * dd_penalty


def _simulate_trades(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, signals: np.ndarray, sl_distances: np.ndarray,
    tp_distances: np.ndarray, symbol_profile: Dict
) -> Dict:
    """Simulate trades given signal array (+1 buy, -1 sell, 0 hold) and SL/TP distances.

    Walk forward through bars, enter at next bar open on signal, exit at SL/TP or
    after `max_hold_bars` bars (whichever first).
    """
    n = len(closes)
    point = symbol_profile["point"]
    pip_value = symbol_profile["pip_value_per_lot"]
    lot = symbol_profile["lot_size"]

    trades = []
    in_trade = False
    direction = 0
    entry_price = 0.0
    sl_dist = 0.0
    tp_dist = 0.0
    entry_bar = 0

    max_hold_bars = 12  # 1 hour on M5

    for i in range(1, n - 1):
        if in_trade:
            # Check SL/TP hit on current bar
            exit_price = None
            if direction == 1:  # BUY
                if lows[i] <= entry_price - sl_dist:
                    exit_price = entry_price - sl_dist
                elif highs[i] >= entry_price + tp_dist:
                    exit_price = entry_price + tp_dist
            else:  # SELL
                if highs[i] >= entry_price + sl_dist:
                    exit_price = entry_price + sl_dist
                elif lows[i] <= entry_price - tp_dist:
                    exit_price = entry_price - tp_dist
            # Time-based exit
            if exit_price is None and (i - entry_bar) >= max_hold_bars:
                exit_price = closes[i]
            if exit_price is not None:
                pnl_points = (exit_price - entry_price) * direction
                pnl = pnl_points / point * pip_value * lot
                trades.append({"entry": entry_price, "exit": exit_price, "pnl": pnl, "direction": direction})
                in_trade = False
        else:
            # Look for signal on this bar, enter on next bar open
            if signals[i] != 0 and sl_distances[i] > 0 and tp_distances[i] > 0:
                in_trade = True
                direction = signals[i]
                entry_price = closes[i + 1] if i + 1 < n else closes[i]
                sl_dist = sl_distances[i]
                tp_dist = tp_distances[i]
                entry_bar = i + 1

    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pnl": 0.0, "max_dd": 0.0, "profit_factor": 0.0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    # Max drawdown of cumulative P&L
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "pnl": total_pnl,
        "max_dd": max_dd,
        "profit_factor": pf,
    }


def _generate_scalp_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate scalp strategy signals with given parameters."""
    n = len(closes)
    signals = np.zeros(n, dtype=int)
    sl_distances = np.zeros(n)
    tp_distances = np.zeros(n)

    rsi = _rsi(closes, params["rsi_period"])
    ema_fast = _ema(closes, params["ema_fast"])
    ema_slow = _ema(closes, params["ema_slow"])
    atr = _atr(highs, lows, closes, 14)
    avg_vol = np.array([np.mean(volumes[max(0, i - 20):i + 1]) if i >= 20 else volumes[i] for i in range(n)])

    for i in range(max(params["ema_slow"], params["rsi_period"], 14) + 1, n):
        if np.isnan(rsi[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(atr[i]):
            continue
        if volumes[i] < avg_vol[i] * params["volume_min_multiplier"]:
            continue
        # Trend filter: EMA fast > slow = bullish
        trend_up = ema_fast[i] > ema_slow[i]
        trend_down = ema_fast[i] < ema_slow[i]
        # RSI extremes with trend filter
        if trend_up and rsi[i] < params["rsi_oversold"] + 5:  # Pullback in uptrend
            signals[i] = 1
            sl_distances[i] = atr[i] * params["atr_sl_multiplier"]
            tp_distances[i] = atr[i] * params["atr_tp_multiplier"]
        elif trend_down and rsi[i] > params["rsi_overbought"] - 5:  # Pullback in downtrend
            signals[i] = -1
            sl_distances[i] = atr[i] * params["atr_sl_multiplier"]
            tp_distances[i] = atr[i] * params["atr_tp_multiplier"]
    return signals, sl_distances, tp_distances


def _generate_trend_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate trend strategy signals with given parameters."""
    n = len(closes)
    signals = np.zeros(n, dtype=int)
    sl_distances = np.zeros(n)
    tp_distances = np.zeros(n)

    ema_fast = _ema(closes, params["ema_fast"])
    ema_slow = _ema(closes, params["ema_slow"])
    adx, plus_di, minus_di = _adx(highs, lows, closes, 14)
    atr = _atr(highs, lows, closes, 14)

    for i in range(max(params["ema_slow"], 28) + 1, n - 1):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(adx[i]):
            continue
        if adx[i] < params["adx_threshold"]:
            continue
        # Crossover detection
        if i > 0 and not np.isnan(ema_fast[i - 1]) and not np.isnan(ema_slow[i - 1]):
            if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                signals[i] = 1
                sl_distances[i] = atr[i] * params["atr_sl_multiplier"]
                tp_distances[i] = atr[i] * params["atr_tp_multiplier"]
            elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                signals[i] = -1
                sl_distances[i] = atr[i] * params["atr_sl_multiplier"]
                tp_distances[i] = atr[i] * params["atr_tp_multiplier"]
    return signals, sl_distances, tp_distances


def _generate_regime_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate regime strategy signals with given parameters."""
    n = len(closes)
    signals = np.zeros(n, dtype=int)
    sl_distances = np.zeros(n)
    tp_distances = np.zeros(n)

    ema21 = _ema(closes, 21)
    ema50 = _ema(closes, 50)
    rsi = _rsi(closes, 14)
    adx, plus_di, minus_di = _adx(highs, lows, closes, 14)
    atr = _atr(highs, lows, closes, 14)
    bb_upper, bb_mid, bb_lower = _bbands(closes, params["bb_period"], params["bb_std"])

    for i in range(50, n - 1):
        if np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(adx[i]):
            continue
        # BULL regime: ADX > threshold, +DI > -DI
        if adx[i] > params["adx_threshold"] and plus_di[i] > minus_di[i]:
            # Pullback: RSI in range, price near EMA21
            if params["bull_pullback_low"] <= rsi[i] <= params["bull_pullback_high"]:
                dist = abs(closes[i] - ema21[i])
                if dist < atr[i] * 0.5:
                    signals[i] = 1
                    sl_distances[i] = atr[i] * 1.5
                    tp_distances[i] = atr[i] * 2.5
        # BEAR regime
        elif adx[i] > params["adx_threshold"] and minus_di[i] > plus_di[i]:
            if params["bear_pullback_low"] <= rsi[i] <= params["bear_pullback_high"]:
                dist = abs(closes[i] - ema21[i])
                if dist < atr[i] * 0.5:
                    signals[i] = -1
                    sl_distances[i] = atr[i] * 1.5
                    tp_distances[i] = atr[i] * 2.5
        # SIDEWAYS: Bollinger band extremes
        elif adx[i] < params["adx_threshold"]:
            if not np.isnan(bb_lower[i]) and not np.isnan(bb_upper[i]):
                if closes[i] <= bb_lower[i] and rsi[i] < 30:
                    signals[i] = 1
                    sl_distances[i] = atr[i] * 1.0
                    tp_distances[i] = atr[i] * 1.5
                elif closes[i] >= bb_upper[i] and rsi[i] > 70:
                    signals[i] = -1
                    sl_distances[i] = atr[i] * 1.0
                    tp_distances[i] = atr[i] * 1.5
    return signals, sl_distances, tp_distances


def optimize_strategy(
    strategy: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    symbol: str,
    min_trades: int = 5,
    progress_callback=None,
) -> Optional[Dict]:
    """Brute-force optimize a strategy on the given data window.

    Args:
        strategy: 'scalp' | 'trend' | 'regime'
        closes/highs/lows/volumes: numpy arrays of M5 data
        symbol: symbol name for pip value lookup
        min_trades: minimum trades to consider a combination valid
        progress_callback: optional fn(percent: int, current_best: Dict) for UI updates

    Returns:
        Best parameters dict with stats, or None if no profitable combo found
    """
    if strategy not in PARAM_RANGES:
        logger.error(f"Unknown strategy: {strategy}")
        return None
    ranges = PARAM_RANGES[strategy]
    keys = list(ranges.keys())
    value_lists = [ranges[k] for k in keys]
    total_combos = 1
    for vl in value_lists:
        total_combos *= len(vl)
    logger.info(f"Optimizer: {strategy} on {symbol} - {total_combos} combinations to test")

    if strategy == "scalp":
        signal_fn = _generate_scalp_signals
    elif strategy == "trend":
        signal_fn = _generate_trend_signals
    else:
        signal_fn = _generate_regime_signals

    profile = SYMBOL_PROFILES.get(symbol, SYMBOL_PROFILES["EURUSD"])
    best_score = -1
    best_result = None

    for i, combo in enumerate(itertools.product(*value_lists)):
        params = dict(zip(keys, combo))
        try:
            signals, sl_dists, tp_dists = signal_fn(closes, highs, lows, volumes, params)
            result = _simulate_trades(closes, highs, lows, volumes, signals, sl_dists, tp_dists, profile)
            if result["trades"] < min_trades:
                continue
            score = _score(result["pnl"], result["win_rate"], result["trades"], result["max_dd"])
            if score > best_score:
                best_score = score
                best_result = {
                    "params": params,
                    "stats": result,
                    "score": score,
                }
        except Exception as e:
            continue
        # Progress callback
        if progress_callback and i % 50 == 0:
            pct = int((i + 1) / total_combos * 100)
            progress_callback(pct, best_result)

    if best_result and best_result["score"] > 0:
        logger.info(
            f"Optimizer: best {strategy} for {symbol}: "
            f"PnL=${best_result['stats']['pnl']:.2f} "
            f"WR={best_result['stats']['win_rate']*100:.1f}% "
            f"Trades={best_result['stats']['trades']} "
            f"PF={best_result['stats']['profit_factor']:.2f} "
            f"Score={best_result['score']:.2f} "
            f"Params={best_result['params']}"
        )
    else:
        if best_result:
            logger.info(
                f"Optimizer: best combo for {strategy} on {symbol} had score=0 "
                f"(likely noise): PnL=${best_result['stats']['pnl']:.2f} "
                f"WR={best_result['stats']['win_rate']*100:.1f}% "
                f"Trades={best_result['stats']['trades']} - REJECTED"
            )
        best_result = None  # Only return results with positive score
        logger.info(f"Optimizer: no profitable combo found for {strategy} on {symbol}")
    return best_result


def get_mt5_recent_data(mt5_connector, symbol: str, hours: int = 6, timeframe_minutes: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fetch recent M5 data from MT5.

    Returns (closes, highs, lows, volumes) numpy arrays, or None on failure.
    """
    try:
        import MetaTrader5 as mt5
        timeframe_map = {
            1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15,
            60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4
        }
        tf = timeframe_map.get(timeframe_minutes, mt5.TIMEFRAME_M5)
        # Need at least 1 bar per 5 minutes in `hours` + warmup bars
        count = max(100, (hours * 60) // timeframe_minutes + 100)
        rates = mt5_connector.get_ohlcv(symbol, _timeframe_to_str(timeframe_minutes), count)
        if rates is None or len(rates) < 60:
            return None
        # Trim to last `hours`
        bars_needed = (hours * 60) // timeframe_minutes
        rates = rates[-bars_needed:] if len(rates) > bars_needed else rates
        closes = np.array([r['close'] for r in rates], dtype=float)
        highs = np.array([r['high'] for r in rates], dtype=float)
        lows = np.array([r['low'] for r in rates], dtype=float)
        volumes = np.array([r.get('tick_volume', 0) for r in rates], dtype=float)
        return closes, highs, lows, volumes
    except Exception as e:
        logger.error(f"Failed to fetch MT5 data for optimizer: {e}")
        return None


def _timeframe_to_str(minutes: int) -> str:
    """Convert minutes to timeframe string used by mt5_connector."""
    return {
        1: "M1", 5: "M5", 15: "M15", 30: "M30",
        60: "H1", 240: "H4", 1440: "D1"
    }.get(minutes, "M5")

"""
High-Win-Rate (HWR) strategy.

DATA-DRIVEN (research/wr_optimizer_v4_oos.py + wr_optimizer_xauusd_relaxed.py):
- Out-of-sample validated 60%+ WR setups on all 4 enabled symbols
- 70/30 chronological train/test split; only setups that pass BOTH train
  (n>=5, WR>=55%) AND OOS (n>=3 for XAUUSD else n>=5, WR>=60%, PF>=1.0)
  are accepted.
- Optimized for "constant small profits" (1:1 to 1:1.5 RR), not big wins.
- Fewer trades, tighter entries: n=5-24 OOS per symbol over 5 months.

ENTRY LOGIC PER REGIME:
- SIDEWAYS: BB+RSI mean reversion at extremes. 1:1 to 1:1.5 RR.
- BULL/BEAR: EMA-pullback with multi-bar RSI confirmation, EMA200 trend
  filter, ADX trend-strength gate. 1:1 to 1:1.5 RR.

PER-SYMBOL PARAMS (see config/high_wr_params.json):
- Each (symbol, regime) cell has its own OOS-validated params.
- Cells without a validated setup return HOLD (e.g. EURUSD SIDEWAYS).
"""
import json
import logging
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "high_wr_params.json",
)

SUPPORTED_SYMBOLS = {"EURUSD", "GBPUSD", "XAUUSD", "BRNUSD"}

# Per-symbol pip / point / digits (must match engine)
SYMBOL_PROFILES = {
    "EURUSD": {"point": 0.0001, "digits": 5},
    "GBPUSD": {"point": 0.0001, "digits": 5},
    "XAUUSD": {"point": 0.01,   "digits": 2},
    "BRNUSD": {"point": 0.01,   "digits": 2},
}


def _pip_size(symbol):
    """Return the value of 1 PIP in price terms.

    For 5-digit forex (EURUSD, GBPUSD): 1 pip = 0.0001 (the standard).
    For 3-digit JPY: 1 pip = 0.01.
    For XAUUSD/BRNUSD: 1 pip = 0.01 (whole cent).

    Note: the engine's pip_multiplier = (1/point) / (10 if digits>=5 else 1).
    For GBPUSD: pip_multiplier = 10000, so 1 pip = 0.0001 = 1/10000.
    The strategy must return sl_pips such that engine's
    sl_price = current - sl_pips / pip_multiplier equals the intended offset.
    So sl_pips in strategy units must match engine units: 1 pip = 0.0001 for GBPUSD.
    """
    p = SYMBOL_PROFILES.get(symbol, {"point": 0.0001})
    point = p["point"]
    if "JPY" in symbol:
        return point * 100
    if symbol in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"):
        # 5-digit forex: 1 pip = 1 point = 0.0001 (NOT point*10)
        return point
    return point


# =============================================================================
# Indicators (numpy)
# =============================================================================
def _ema(values, period):
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes, period=14):
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(n); avg_l = np.zeros(n)
    avg_g[period] = np.mean(gains[:period])
    avg_l[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i-1]*(period-1) + gains[i-1]) / period
        avg_l[i] = (avg_l[i-1]*(period-1) + losses[i-1]) / period
    rs = np.divide(avg_g, avg_l, out=np.zeros(n), where=avg_l > 0)
    out = 100 - 100 / (1 + rs)
    out[:period] = np.nan
    return out


def _atr(highs, lows, closes, period=14):
    n = len(closes)
    out = np.zeros(n)
    if n < 2:
        return out
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    out[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i-1]*(period-1) + tr[i]) / period
    return out


def _adx(highs, lows, closes, period=14):
    n = len(closes)
    a = np.full(n, np.nan); p_di = np.full(n, np.nan); m_di = np.full(n, np.nan)
    if n < period * 2:
        return a, p_di, m_di
    p_dm = np.zeros(n); m_dm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i-1]; down = lows[i-1] - lows[i]
        p_dm[i] = up if (up > down and up > 0) else 0
        m_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    s_tr = np.zeros(n); s_p = np.zeros(n); s_m = np.zeros(n)
    s_tr[period] = np.sum(tr[1:period+1])
    s_p[period] = np.sum(p_dm[1:period+1])
    s_m[period] = np.sum(m_dm[1:period+1])
    for i in range(period+1, n):
        s_tr[i] = s_tr[i-1] - s_tr[i-1]/period + tr[i]
        s_p[i] = s_p[i-1] - s_p[i-1]/period + p_dm[i]
        s_m[i] = s_m[i-1] - s_m[i-1]/period + m_dm[i]
    p_di_v = 100 * np.divide(s_p, s_tr, out=np.zeros(n), where=s_tr > 0)
    m_di_v = 100 * np.divide(s_m, s_tr, out=np.zeros(n), where=s_tr > 0)
    dx = 100 * np.abs(p_di_v - m_di_v) / np.where((p_di_v + m_di_v) > 0, (p_di_v + m_di_v), 1)
    a[period*2] = np.mean(dx[period:period*2+1])
    for i in range(period*2+1, n):
        a[i] = (a[i-1]*(period-1) + dx[i]) / period
    return a, p_di_v, m_di_v


# =============================================================================
# Param loader
# =============================================================================
_PARAMS_CACHE = {"mtime": 0.0, "data": {}}


def _load_params():
    try:
        if not os.path.exists(_PARAMS_PATH):
            return {}
        mtime = os.path.getmtime(_PARAMS_PATH)
        if mtime == _PARAMS_CACHE["mtime"] and _PARAMS_CACHE["data"]:
            return _PARAMS_CACHE["data"]
        with open(_PARAMS_PATH, "r") as f:
            data = json.load(f)
        # Strip meta keys
        clean = {}
        for sym, entry in data.items():
            if sym.startswith("_"):
                continue
            if not isinstance(entry, dict):
                continue
            clean[sym] = {}
            for regime, p in entry.items():
                if regime.startswith("_") or not isinstance(p, dict):
                    continue
                clean[sym][regime] = p
        _PARAMS_CACHE["mtime"] = mtime
        _PARAMS_CACHE["data"] = clean
        return clean
    except Exception as e:
        logger.warning(f"high_wr: failed to load params: {e}")
        return {}


def get_params(symbol: str, regime: str) -> Optional[dict]:
    data = _load_params()
    return data.get(symbol, {}).get(regime)


# =============================================================================
# Regime detection
# =============================================================================
def detect_regime(closes, highs, lows, adx, p_di, m_di, ema20, ema50, i):
    """Return 'SIDEWAYS' | 'TRENDING_UP' | 'TRENDING_DOWN' | 'UNKNOWN'."""
    if i < 50 or np.isnan(adx[i]):
        return "UNKNOWN"
    if adx[i] >= 25:
        if ema20[i] > ema50[i] and p_di[i] > m_di[i]:
            return "TRENDING_UP"
        if ema20[i] < ema50[i] and m_di[i] > p_di[i]:
            return "TRENDING_DOWN"
        return "TRENDING_UP" if p_di[i] > m_di[i] else "TRENDING_DOWN"
    if adx[i] >= 20:
        return "TRENDING_UP" if p_di[i] > m_di[i] else "TRENDING_DOWN"
    return "SIDEWAYS"


# =============================================================================
# Signal generation
# =============================================================================
def _sideways_entry(closes, highs, lows, bb_u, bb_l, rsi, atr_v, i, p, symbol):
    """BB+RSI mean reversion. Returns (action, sl_pips, tp_pips) or (None, _, _).

    bb_u and bb_l are floats (the band values at index i), pre-extracted by caller.
    rsi and atr_v are full numpy arrays (multi-bar confirmation needs the lookback).
    """
    if np.isnan(rsi[i]) or np.isnan(bb_l) or np.isnan(bb_u) or atr_v[i] <= 0:
        return None, 0, 0
    adx_min = p.get("adx_min", 8)
    adx_max = p.get("adx_max", 25)
    rsi_os = p["rsi_oversold"]
    rsi_ob = p["rsi_overbought"]
    require_close_beyond = p.get("require_close_beyond", False)
    k = int(p.get("rsi_extreme_bars", 1))
    sl_mult = float(p["sl_atr_mult"])
    tp_mult = float(p["tp_atr_mult"])

    # Multi-bar RSI confirmation
    if k > 1:
        if i - k + 1 < 0:
            return None, 0, 0
        win_buy = rsi[i-k+1:i+1]
        win_sell = rsi[i-k+1:i+1]
        if np.any(np.isnan(win_buy)):
            return None, 0, 0
        if not (np.all(win_buy < rsi_os) or np.all(win_sell > rsi_ob)):
            return None, 0, 0

    pip = _pip_size(symbol)
    sl_pips = atr_v[i] * sl_mult / pip
    tp_pips = atr_v[i] * tp_mult / pip

    if require_close_beyond:
        if closes[i] < bb_l and rsi[i] < rsi_os:
            return "BUY", sl_pips, tp_pips
        if closes[i] > bb_u and rsi[i] > rsi_ob:
            return "SELL", sl_pips, tp_pips
    else:
        if closes[i] <= bb_l and rsi[i] < rsi_os:
            return "BUY", sl_pips, tp_pips
        if closes[i] >= bb_u and rsi[i] > rsi_ob:
            return "SELL", sl_pips, tp_pips
    return None, 0, 0


def _trend_entry(closes, highs, lows, ema20, ema50, ema200, rsi, atr_v, adx, i, p, direction):
    """EMA-pullback entry with multi-bar confirmation.

    `direction` is 'TRENDING_UP' (look for BUY) or 'TRENDING_DOWN' (SELL).
    """
    if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i])
        or np.isnan(rsi[i]) or np.isnan(adx[i]) or atr_v[i] <= 0):
        return None, 0, 0

    adx_min = float(p["adx_min"])
    if p.get("adx_strong_only"):
        adx_min = max(adx_min, 30.0)
    if adx[i] < adx_min:
        return None, 0, 0

    if direction == "TRENDING_UP":
        if p.get("require_ema200", True):
            if closes[i] <= ema50[i] or closes[i] <= ema200[i]:
                return None, 0, 0
        else:
            if closes[i] <= ema50[i]:
                return None, 0, 0
        rsi_min = float(p["rsi_buy_min"]); rsi_max = float(p["rsi_buy_max"])
    else:
        if p.get("require_ema200", True):
            if closes[i] >= ema50[i] or closes[i] >= ema200[i]:
                return None, 0, 0
        else:
            if closes[i] >= ema50[i]:
                return None, 0, 0
        rsi_min = float(p["rsi_sell_min"]); rsi_max = float(p["rsi_sell_max"])

    # RSI in pullback zone
    if rsi[i] < rsi_min or rsi[i] > rsi_max:
        return None, 0, 0

    # Distance from fast EMA in ATRs
    dist = abs(closes[i] - ema20[i]) / atr_v[i]
    if dist > float(p["distance_max_atr"]):
        return None, 0, 0

    # Multi-bar confirmation
    k = int(p.get("rsi_zone_bars", 1))
    if k > 1:
        if i - k + 1 < 0:
            return None, 0, 0
        win_rsi = rsi[i-k+1:i+1]
        win_dist = np.array([abs(closes[i-k+1+j] - ema20[i-k+1+j]) / atr_v[i-k+1+j]
                             if (not np.isnan(closes[i-k+1+j]) and not np.isnan(ema20[i-k+1+j])
                                 and atr_v[i-k+1+j] > 0) else 999.0
                             for j in range(k)])
        if np.any(np.isnan(win_rsi)) or np.any(np.isnan(win_dist)):
            return None, 0, 0
        in_zone = np.all((win_rsi >= rsi_min) & (win_rsi <= rsi_max)) and np.all(win_dist <= float(p["distance_max_atr"]))
        if not in_zone:
            return None, 0, 0

    return "BUY" if direction == "TRENDING_UP" else "SELL", 0, 0  # sl_pips/tp_pips set below


def generate_signal(symbol: str, ohlcv: list, current_regime: Optional[str] = None):
    """Generate a high-WR signal.

    Args:
        symbol: e.g. 'EURUSD'
        ohlcv: list of {open, high, low, close} dicts (any timeframe)
        current_regime: optional pre-detected regime; auto-detect if None

    Returns dict with: action, confidence, sl_pips, tp_pips, reasoning, regime.
    """
    if symbol not in SUPPORTED_SYMBOLS:
        return {"action": "HOLD", "confidence": 0, "reasoning": f"{symbol} not in supported list",
                "regime": "UNSUPPORTED"}
    if not ohlcv or len(ohlcv) < 60:
        return {"action": "HOLD", "confidence": 0, "reasoning": "Insufficient data",
                "regime": "UNKNOWN"}
    # If we have <220 bars, we can still evaluate SIDEWAYS (BB+RSI needs only
    # 20 bars), but BULL/BEAR require EMA200 so we won't trade trends until
    # the engine has accumulated enough history. The engine fetches 250 H1
    # bars (10 days) so this is rarely a concern in practice.

    closes = np.array([b["close"] for b in ohlcv])
    highs = np.array([b["high"] for b in ohlcv])
    lows = np.array([b["low"] for b in ohlcv])
    n = len(closes)
    i = n - 1

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    rsi = _rsi(closes)
    atr_v = _atr(highs, lows, closes)
    adx, p_di, m_di = _adx(highs, lows, closes)

    if current_regime is None:
        current_regime = detect_regime(closes, highs, lows, adx, p_di, m_di, ema20, ema50, i)

    # Map regime to strategy-side label
    if current_regime == "TRENDING_UP":
        strat_regime = "BULL"
    elif current_regime == "TRENDING_DOWN":
        strat_regime = "BEAR"
    elif current_regime == "SIDEWAYS":
        strat_regime = "SIDEWAYS"
    else:
        return {"action": "HOLD", "confidence": 0,
                "reasoning": f"Regime={current_regime} (no validated setup)",
                "regime": current_regime}

    p = get_params(symbol, strat_regime)
    if p is None:
        return {"action": "HOLD", "confidence": 0,
                "reasoning": f"{symbol}/{strat_regime} has no OOS-validated setup (skip)",
                "regime": current_regime}

    pip = _pip_size(symbol)
    action = None
    sl_pips = 0.0
    tp_pips = 0.0
    reasoning = ""

    if strat_regime == "SIDEWAYS":
        bb_period = int(p["bb_period"]); bb_std = float(p["bb_std"])
        # Build BB on the fly
        if i < bb_period - 1:
            return {"action": "HOLD", "confidence": 0,
                    "reasoning": "Insufficient BB window", "regime": current_regime}
        window = closes[i-bb_period+1:i+1]
        bb_mid = float(np.mean(window))
        bb_std_v = float(np.std(window))
        bb_u = bb_mid + bb_std * bb_std_v
        bb_l = bb_mid - bb_std * bb_std_v
        # ADX gate
        if not (np.isnan(adx[i])) and (adx[i] < p.get("adx_min", 0) or adx[i] > p.get("adx_max", 100)):
            return {"action": "HOLD", "confidence": 0,
                    "reasoning": f"ADX={adx[i]:.1f} outside [{p.get('adx_min',0)},{p.get('adx_max',100)}]",
                    "regime": current_regime}
        action, sl_pips, tp_pips = _sideways_entry(closes, highs, lows, bb_u, bb_l, rsi, atr_v, i, p, symbol)
        if action is None:
            return {"action": "HOLD", "confidence": 0,
                    "reasoning": f"SIDEWAYS waiting for extremes (RSI={rsi[i]:.1f})",
                    "regime": current_regime}
        reasoning = (f"HWR SIDEWAYS {action}: BB period {bb_period} std {bb_std}, "
                     f"RSI={rsi[i]:.1f}, ATR={atr_v[i]:.5f}")
    else:
        # BULL or BEAR
        action, _, _ = _trend_entry(closes, highs, lows, ema20, ema50, ema200,
                                     rsi, atr_v, adx, i, p, current_regime)
        if action is None:
            return {"action": "HOLD", "confidence": 0,
                    "reasoning": f"{strat_regime} no pullback setup (RSI={rsi[i]:.1f}, ADX={adx[i]:.1f})",
                    "regime": current_regime}
        sl_mult = float(p["atr_sl_mult"])
        tp_mult = float(p.get("tp_atr_mult") or p["atr_tp_mult"])
        sl_pips = atr_v[i] * sl_mult / pip
        tp_pips = atr_v[i] * tp_mult / pip
        reasoning = (f"HWR {strat_regime} {action}: RSI={rsi[i]:.1f}, ADX={adx[i]:.1f}, "
                     f"dist={abs(closes[i]-ema20[i])/atr_v[i]:.2f} ATR")

    return {
        "action": action,
        "confidence": 75,  # high-WR setups get fixed 75% (above scalp/trend 50-60%)
        "sl_pips": round(sl_pips, 1),
        "tp_pips": round(tp_pips, 1),
        "reasoning": reasoning,
        "regime": current_regime,
        "strategy": "high_wr",
    }


# =============================================================================
# Engine integration
# =============================================================================
class HighWinRateStrategy:
    """High-WR strategy wrapper for the engine.

    Receives the engine's flat-dict market_data (closes, highs, lows, etc.)
    and returns the engine's expected signal dict (signal/confidence/reason/
    sl_pips/tp_pips/regime).
    """
    name = "high_wr"
    description = "OOS-validated 60%+ WR per-symbol setups (1:1 to 1:1.5 RR, small consistent profits)"

    def __init__(self, config: dict = None):
        self.config = config or {}
        # High-WR uses 1:1 to 1.5:1 RR by design ("constant small profits").
        # The engine's hard 2:1 RR gate at trading_engine.py:1836 checks this
        # attribute; without it, every high_wr signal is silently rejected.
        self.min_risk_reward = self.config.get("min_risk_reward", 1.0)
        # Streak-based lot adjustment (required by trade_manager)
        self.loss_streak_reduction_pct = self.config.get("loss_streak_reduction_pct", 50)
        self.loss_streak_threshold = self.config.get("loss_streak_threshold", 2)
        self.win_streak_boost_pct = self.config.get("win_streak_boost_pct", 20)
        self.win_streak_increase_pct = self.config.get("win_streak_increase_pct", 20)
        self.win_streak_threshold = self.config.get("win_streak_threshold", 2)
        self.max_streak_lot_multiplier = self.config.get("max_streak_lot_multiplier", 2.0)
        self.base_lot_size = self.config.get("base_lot_size", 0.01)
        # Daily limits
        self.max_trades_per_day = self.config.get("max_trades_per_day", 6)
        self.daily_loss_limit_pct = self.config.get("daily_loss_limit_pct", 3.0)

    def check_setup(self, market_data: dict) -> dict:
        symbol = market_data.get("symbol", "EURUSD")
        closes = market_data.get("closes")
        highs = market_data.get("highs")
        lows = market_data.get("lows")
        if closes is None or highs is None or lows is None or len(closes) == 0:
            return {"signal": "HOLD", "confidence": 0, "reason": "No OHLCV data",
                    "sl_pips": 10, "tp_pips": 15, "regime": "UNKNOWN", "strategy": self.name}
        n = min(len(closes), len(highs), len(lows))
        if n == 0:
            return {"signal": "HOLD", "confidence": 0, "reason": "No OHLCV data",
                    "sl_pips": 10, "tp_pips": 15, "regime": "UNKNOWN", "strategy": self.name}
        closes = closes[:n]; highs = highs[:n]; lows = lows[:n]
        ohlcv = [{"open": closes[i], "high": highs[i], "low": lows[i], "close": closes[i]}
                 for i in range(n)]
        sig = generate_signal(symbol, ohlcv)
        return {
            "signal": sig["action"],
            "confidence": sig["confidence"],
            "reason": sig["reasoning"],
            "sl_pips": sig.get("sl_pips", 10),
            "tp_pips": sig.get("tp_pips", 15),
            "regime": sig.get("regime", "UNKNOWN"),
            "strategy": self.name,
        }

    def get_signal(self, market_data: dict) -> dict:
        return self.check_setup(market_data)

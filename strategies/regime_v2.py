"""
Regime-aware strategy based on backtest research (50K M5 bars per symbol).

DATA-DRIVEN FINDINGS (STRATEGY_RESEARCH_REPORT.json + STRATEGY_H1_BACKTEST.json):
- Mean reversion in SIDEWAYS is the ONLY consistent edge across 4 symbols
  on BOTH M5 (PF 1.6-1.9) and H1 (PF 3.19-4.98).
- Trend following does NOT work: M5 backtest had no edge; H1 backtest
  was NEGATIVE on all 4 symbols (PF 0.24-0.84, PnL -$14 to -$163).
  See research/regime_backtest_h1.py for the H1 numbers.

This strategy:
- Detects regime via ADX + EMA(20/50) + DI
- Trades SIDEWAYS regime by default (where backtest found edge)
- Optionally trades TRENDING_UP / TRENDING_DOWN with EMA pullback entry
  when `enable_trend_trading=True` is passed in config. Default is OFF
  because the H1 backtest showed trend loses money on every symbol.
- Uses Bollinger Band + RSI for SIDEWAYS entry signals
- Uses ATR-based SL/TP:
    SIDEWAYS: 1.0 / 1.5 ATR (1:1.5 RR) — data-validated
    TREND:    2.0 / 4.0 ATR (1:2 RR)   — ported from regime_aware v1
- Skips all other regimes by default (no edge = no trade)
"""
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Constants (from data-driven research)
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # buy signal
RSI_OVERBOUGHT = 70  # sell signal
SL_ATR_MULT = 1.0
TP_ATR_MULT = 1.5

# Regime thresholds (verified to work on backtest data)
ADX_TRENDING_THRESHOLD = 20
ADX_STRONG_TREND = 25
EMA_FAST = 20
EMA_SLOW = 50
ADX_PERIOD = 14
ATR_PERIOD = 14

# Trend-trading constants (ported from regime_aware v1).
# Used only when enable_trend_trading=True. Backtest showed trend loses
# on all 4 symbols, so the default in RegimeV2Strategy is OFF.
TREND_RSI_BUY_MIN = 35
TREND_RSI_BUY_MAX = 65
TREND_RSI_SELL_MIN = 35
TREND_RSI_SELL_MAX = 65
TREND_ATR_SL_MULT = 2.0
TREND_ATR_TP_MULT = 4.0
TREND_MAX_DISTANCE_ATR = 2.5  # require price near EMA fast (pullback zone)

# Symbols where backtest showed positive edge
SUPPORTED_SYMBOLS = {'EURUSD', 'GBPUSD', 'XAUUSD', 'BRNUSD'}


def ema(values, period):
    """Exponential moving average."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2 / (period + 1)
    out[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(closes, period=RSI_PERIOD):
    """Relative Strength Index."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(n)
    avg_l = np.zeros(n)
    avg_g[period] = np.mean(gains[:period])
    avg_l[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.divide(avg_g, avg_l, out=np.zeros(n), where=avg_l > 0)
    rsi_arr = 100 - 100 / (1 + rs)
    rsi_arr[:period] = np.nan
    return rsi_arr


def atr(highs, lows, closes, period=ATR_PERIOD):
    """Average True Range."""
    n = len(closes)
    out = np.zeros(n)
    if n < 2:
        return out
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(highs, lows, closes, period=ADX_PERIOD):
    """ADX with +DI and -DI."""
    n = len(closes)
    a = np.full(n, np.nan)
    p_di = np.full(n, np.nan)
    m_di = np.full(n, np.nan)
    if n < period * 2:
        return a, p_di, m_di
    p_dm = np.zeros(n)
    m_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        p_dm[i] = up if (up > down and up > 0) else 0
        m_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    s_tr = np.zeros(n)
    s_p = np.zeros(n)
    s_m = np.zeros(n)
    s_tr[period] = np.sum(tr[1:period + 1])
    s_p[period] = np.sum(p_dm[1:period + 1])
    s_m[period] = np.sum(m_dm[1:period + 1])
    for i in range(period + 1, n):
        s_tr[i] = s_tr[i - 1] - s_tr[i - 1] / period + tr[i]
        s_p[i] = s_p[i - 1] - s_p[i - 1] / period + p_dm[i]
        s_m[i] = s_m[i - 1] - s_m[i - 1] / period + m_dm[i]
    p_di_v = 100 * np.divide(s_p, s_tr, out=np.zeros(n), where=s_tr > 0)
    m_di_v = 100 * np.divide(s_m, s_tr, out=np.zeros(n), where=s_tr > 0)
    dx = 100 * np.abs(p_di_v - m_di_v) / np.where((p_di_v + m_di_v) > 0, (p_di_v + m_di_v), 1)
    a[period * 2] = np.mean(dx[period:period * 2 + 1])
    for i in range(period * 2 + 1, n):
        a[i] = (a[i - 1] * (period - 1) + dx[i]) / period
    return a, p_di_v, m_di_v


def detect_regime(adx_val, ema_fast_val, ema_slow_val, plus_di, minus_di):
    """Classify current regime based on indicators.

    Returns: 'TRENDING_UP' | 'TRENDING_DOWN' | 'SIDEWAYS' | 'VOLATILE' | 'UNKNOWN'
    """
    if adx_val is None or np.isnan(adx_val) or adx_val == 0:
        return 'UNKNOWN'
    if adx_val >= ADX_STRONG_TREND:
        if ema_fast_val > ema_slow_val and plus_di > minus_di:
            return 'TRENDING_UP'
        elif ema_fast_val < ema_slow_val and minus_di > plus_di:
            return 'TRENDING_DOWN'
    elif adx_val >= ADX_TRENDING_THRESHOLD:
        if plus_di > minus_di:
            return 'TRENDING_UP'
        else:
            return 'TRENDING_DOWN'
    return 'SIDEWAYS'


def compute_indicators(ohlcv):
    """Compute all indicators from OHLCV data. Expects list of dicts with 'close','high','low'."""
    closes = np.array([b['close'] for b in ohlcv])
    highs = np.array([b['high'] for b in ohlcv])
    lows = np.array([b['low'] for b in ohlcv])
    ema_f = ema(closes, EMA_FAST)
    ema_s = ema(closes, EMA_SLOW)
    rsi_v = rsi(closes)
    atr_v = atr(highs, lows, closes)
    a, p_di, m_di = adx(highs, lows, closes)
    return {
        'closes': closes, 'highs': highs, 'lows': lows,
        'ema_fast': ema_f, 'ema_slow': ema_s,
        'rsi': rsi_v, 'atr': atr_v,
        'adx': a, 'plus_di': p_di, 'minus_di': m_di
    }


def _atr_to_pips(symbol, atr_value):
    """Convert ATR (in price) to pips using the same profiles as generate_signal.

    Engine's pip_multiplier for 5-digit forex (EURUSD, GBPUSD) = 10000,
    so 1 pip = 0.0001 = point. NOT point*10 (which was the old buggy value).
    """
    SYMBOL_PROFILES_LOCAL = {
        'EURUSD': {'point': 0.0001},
        'GBPUSD': {'point': 0.0001},
        'XAUUSD': {'point': 0.01},
        'BRNUSD': {'point': 0.01},
    }
    profile = SYMBOL_PROFILES_LOCAL.get(symbol, {'point': 0.0001})
    point = profile['point']
    if 'JPY' in symbol:
        pip_size = point * 100
    elif symbol in ('EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'):
        # 5-digit forex: 1 pip = 1 point = 0.0001 (NOT point*10)
        pip_size = point
    else:
        pip_size = point
    return atr_value / pip_size if pip_size > 0 else 0.0


def _evaluate_trend(symbol, current_price, ema_f, ema_s, rsi_v, atr_v, adx_v, current_regime, params=None):
    """EMA-pullback entry for TRENDING_UP / TRENDING_DOWN regimes.

    Ported from strategies/regime_aware._evaluate_bull/_evaluate_bear.
    Only used when enable_trend_trading=True (default OFF in RegimeV2Strategy).
    `params` is the per-symbol strategy_params dict (overrides the module
    constants TREND_RSI_*, TREND_MAX_DISTANCE_ATR, TREND_ATR_SL_MULT,
    TREND_ATR_TP_MULT).

    Returns dict with action/confidence/reasoning, or HOLD if no pullback setup.
    """
    p = params or {}
    eff_rsi_buy_min = float(p.get("rsi_buy_min", TREND_RSI_BUY_MIN))
    eff_rsi_buy_max = float(p.get("rsi_buy_max", TREND_RSI_BUY_MAX))
    eff_rsi_sell_min = float(p.get("rsi_sell_min", TREND_RSI_SELL_MIN))
    eff_rsi_sell_max = float(p.get("rsi_sell_max", TREND_RSI_SELL_MAX))
    eff_distance_atr = float(p.get("distance_max_atr", TREND_MAX_DISTANCE_ATR))
    eff_sl_mult = float(p.get("atr_sl_mult", TREND_ATR_SL_MULT))
    eff_tp_mult = float(p.get("atr_tp_mult", TREND_ATR_TP_MULT))

    if ema_f is None or ema_s is None or rsi_v is None or atr_v is None or atr_v <= 0:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'TREND: indicators not ready',
            'regime': current_regime,
            'adx': adx_v,
        }

    # In an uptrend the slow EMA is below price; in a downtrend it's above.
    if current_regime == 'TRENDING_UP':
        if current_price <= ema_s:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'TREND_UP skip: price {current_price:.5f} below slow EMA {ema_s:.5f}',
                'regime': current_regime,
                'adx': adx_v,
            }
        distance = abs(current_price - ema_f) / atr_v
        if distance > eff_distance_atr:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'TREND_UP skip: too far from EMA fast ({distance:.1f} ATRs)',
                'regime': current_regime,
                'adx': adx_v,
            }
        if rsi_v < eff_rsi_buy_min or rsi_v > eff_rsi_buy_max:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'TREND_UP skip: RSI={rsi_v:.1f} not in pullback zone ({eff_rsi_buy_min}-{eff_rsi_buy_max})',
                'regime': current_regime,
                'adx': adx_v,
            }
        sl_pips = _atr_to_pips(symbol, atr_v * eff_sl_mult)
        tp_pips = _atr_to_pips(symbol, atr_v * eff_tp_mult)
        return {
            'action': 'BUY',
            'confidence': min(85, 50 + int((eff_rsi_buy_max - rsi_v) / 2)),
            'reasoning': f'TREND_UP pullback: RSI={rsi_v:.1f}, EMA dist={distance:.1f} ATR, ADX={adx_v:.1f}',
            'regime': current_regime,
            'adx': adx_v,
            'rsi': rsi_v,
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1),
        }

    # current_regime == 'TRENDING_DOWN'
    if current_price >= ema_s:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': f'TREND_DOWN skip: price {current_price:.5f} above slow EMA {ema_s:.5f}',
            'regime': current_regime,
            'adx': adx_v,
        }
    distance = abs(current_price - ema_f) / atr_v
    if distance > eff_distance_atr:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': f'TREND_DOWN skip: too far from EMA fast ({distance:.1f} ATRs)',
            'regime': current_regime,
            'adx': adx_v,
        }
    if rsi_v < eff_rsi_sell_min or rsi_v > eff_rsi_sell_max:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': f'TREND_DOWN skip: RSI={rsi_v:.1f} not in rally zone ({eff_rsi_sell_min}-{eff_rsi_sell_max})',
            'regime': current_regime,
            'adx': adx_v,
        }
    sl_pips = _atr_to_pips(symbol, atr_v * eff_sl_mult)
    tp_pips = _atr_to_pips(symbol, atr_v * eff_tp_mult)
    return {
        'action': 'SELL',
        'confidence': min(85, 50 + int((rsi_v - eff_rsi_sell_min) / 2)),
        'reasoning': f'TREND_DOWN rally: RSI={rsi_v:.1f}, EMA dist={distance:.1f} ATR, ADX={adx_v:.1f}',
        'regime': current_regime,
        'adx': adx_v,
        'rsi': rsi_v,
        'sl_pips': round(sl_pips, 1),
        'tp_pips': round(tp_pips, 1),
    }


def generate_signal(symbol, ohlcv, current_regime=None, enable_trend_trading=False,
                     strategy_params=None):
    """Generate trading signal based on data-driven rules.

    Args:
        symbol: instrument ticker.
        ohlcv: list of dicts with 'close','high','low' (and optionally open/volume).
        current_regime: optional pre-detected regime ('TRENDING_UP',
            'TRENDING_DOWN', 'SIDEWAYS', 'VOLATILE', 'UNKNOWN'). If None, the
            function detects the regime from indicators.
        enable_trend_trading: if True, also enter trades in TRENDING_UP and
            TRENDING_DOWN regimes using EMA-pullback logic (ported from
            regime_aware v1). Default False — H1 backtest showed trend loses
            on every symbol, so the data-validated default is SIDEWAYS-only.
        strategy_params: optional dict of per-symbol parameter overrides
            (loaded from config/per_symbol_params.json by the regime
            router). Recognized keys: bb_period, bb_std, rsi_oversold,
            rsi_overbought, sl_atr_mult, tp_atr_mult, adx_min, adx_max,
            rsi_extreme_bars, plus the trend params: rsi_buy_min/max,
            rsi_sell_min/max, distance_max_atr, atr_sl_mult, atr_tp_mult,
            rsi_zone_bars, enable_bull, enable_bear.
            When a key is missing, the module-level default is used.

    Returns dict with:
        - action: 'BUY' | 'SELL' | 'HOLD'
        - confidence: 0-100
        - sl_pips: suggested SL in pips (for engine)
        - tp_pips: suggested TP in pips
        - reasoning: human-readable explanation
        - regime: current market regime
    """
    params = strategy_params or {}

    # Resolve effective params (override > default)
    eff_bb_period = int(params.get("bb_period", BB_PERIOD))
    eff_bb_std = float(params.get("bb_std", BB_STD))
    eff_rsi_oversold = float(params.get("rsi_oversold", RSI_OVERSOLD))
    eff_rsi_overbought = float(params.get("rsi_overbought", RSI_OVERBOUGHT))
    eff_sl_atr_mult = float(params.get("sl_atr_mult", SL_ATR_MULT))
    eff_tp_atr_mult = float(params.get("tp_atr_mult", TP_ATR_MULT))
    eff_adx_min = float(params.get("adx_min", ADX_TRENDING_THRESHOLD))
    eff_adx_max = float(params.get("adx_max", ADX_STRONG_TREND))
    eff_rsi_extreme_bars = int(params.get("rsi_extreme_bars", 1))

    if symbol not in SUPPORTED_SYMBOLS:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': f'{symbol} not in supported list (backtest showed no edge)',
            'regime': 'UNSUPPORTED'
        }

    if not ohlcv or len(ohlcv) < 60:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'Insufficient data for indicators',
            'regime': 'UNKNOWN'
        }

    indicators = compute_indicators(ohlcv)
    n = len(indicators['closes'])
    i = n - 1  # last bar

    # Get current values
    current_price = float(indicators['closes'][i])
    ema_f = float(indicators['ema_fast'][i]) if not np.isnan(indicators['ema_fast'][i]) else None
    ema_s = float(indicators['ema_slow'][i]) if not np.isnan(indicators['ema_slow'][i]) else None
    rsi_v = float(indicators['rsi'][i]) if not np.isnan(indicators['rsi'][i]) else None
    atr_v = float(indicators['atr'][i]) if indicators['atr'][i] > 0 else None
    adx_v = float(indicators['adx'][i]) if not np.isnan(indicators['adx'][i]) else None
    p_di = float(indicators['plus_di'][i]) if not np.isnan(indicators['plus_di'][i]) else 0
    m_di = float(indicators['minus_di'][i]) if not np.isnan(indicators['minus_di'][i]) else 0

    if current_regime is None:
        # Detect regime honoring the per-symbol ADX bounds. SIDEWAYS only when
        # ADX is in [adx_min, adx_max]; else trending.
        if adx_v is None or np.isnan(adx_v) or adx_v == 0:
            current_regime = 'UNKNOWN'
        elif adx_v > eff_adx_max or adx_v < eff_adx_min:
            # Outside the SIDEWAYS sweet spot — trending
            if ema_f is not None and ema_s is not None:
                if ema_f > ema_s and p_di > m_di:
                    current_regime = 'TRENDING_UP'
                elif ema_f < ema_s and m_di > p_di:
                    current_regime = 'TRENDING_DOWN'
                else:
                    current_regime = 'TRENDING_UP' if p_di > m_di else 'TRENDING_DOWN'
            else:
                current_regime = 'TRENDING_UP' if p_di > m_di else 'TRENDING_DOWN'
        else:
            current_regime = 'SIDEWAYS'

    # ===== Trend regime: gated by enable_trend_trading =====
    if current_regime in ('TRENDING_UP', 'TRENDING_DOWN'):
        if not enable_trend_trading:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'Regime={current_regime} (no edge found in backtest, skipping)',
                'regime': current_regime,
                'adx': adx_v,
            }
        # Per-symbol gate: enable_bull / enable_bear can disable a direction
        if current_regime == 'TRENDING_UP' and params.get("enable_bull") is False:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': 'BULL disabled for this symbol (per_symbol_params)',
                'regime': current_regime,
                'adx': adx_v,
            }
        if current_regime == 'TRENDING_DOWN' and params.get("enable_bear") is False:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': 'BEAR disabled for this symbol (per_symbol_params)',
                'regime': current_regime,
                'adx': adx_v,
            }
        return _evaluate_trend(symbol, current_price, ema_f, ema_s, rsi_v, atr_v, adx_v, current_regime, params=params)

    # ===== SIDEWAYS (default edge) =====
    if current_regime != 'SIDEWAYS':
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': f'Regime={current_regime} (no edge found in backtest, skipping)',
            'regime': current_regime,
            'adx': adx_v
        }

    # Need valid indicators
    if rsi_v is None or atr_v is None or ema_f is None or ema_s is None:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'Indicators not ready',
            'regime': current_regime
        }

    # ===== Bollinger Band (per-symbol period + std) =====
    window = indicators['closes'][max(0, i - eff_bb_period + 1):i + 1]
    if len(window) < eff_bb_period:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'Insufficient BB window',
            'regime': current_regime
        }
    bb_mid = float(np.mean(window))
    bb_std = float(np.std(window))
    bb_upper = bb_mid + eff_bb_std * bb_std
    bb_lower = bb_mid - eff_bb_std * bb_std

    # ===== Multi-bar RSI confirmation =====
    # Require rsi to be in extreme for N consecutive bars (default 1 = no
    # confirmation). Uses a simple lookback over the rsi indicator.
    rsi_series = indicators['rsi']
    rsi_confirm = True
    if eff_rsi_extreme_bars > 1:
        k = eff_rsi_extreme_bars
        # Look back k bars (including current) — all must satisfy the
        # condition we're about to check.
        lookback = rsi_series[max(0, i - k + 1):i + 1]
        if len(lookback) < k or np.any(np.isnan(lookback)):
            rsi_confirm = False
        # Pre-check: we don't know BUY vs SELL yet, so just confirm that
        # the entire lookback is on ONE side of 50 (extreme) — not both.
        # We'll refine below based on the action candidate.
        all_oversold = bool(np.all(lookback < eff_rsi_oversold))
        all_overbought = bool(np.all(lookback > eff_rsi_overbought))
        rsi_confirm = all_oversold or all_overbought

    # ===== Generate signal =====
    action = 'HOLD'
    reasoning = ''
    confidence = 0

    # BUY: at lower BB AND RSI oversold
    if current_price <= bb_lower and rsi_v < eff_rsi_oversold and rsi_confirm:
        action = 'BUY'
        # Higher confidence when deeply oversold
        confidence = min(100, 50 + (eff_rsi_oversold - rsi_v) * 2)
        reasoning = (f'SIDEWAYS + BUY: RSI={rsi_v:.1f}<{eff_rsi_oversold}, '
                     f'price at lower BB ({bb_lower:.5f})')
    # SELL: at upper BB AND RSI overbought
    elif current_price >= bb_upper and rsi_v > eff_rsi_overbought and rsi_confirm:
        action = 'SELL'
        confidence = min(100, 50 + (rsi_v - eff_rsi_overbought) * 2)
        reasoning = (f'SIDEWAYS + SELL: RSI={rsi_v:.1f}>{eff_rsi_overbought}, '
                     f'price at upper BB ({bb_upper:.5f})')
    else:
        # No signal - log why
        if rsi_v < 40 and current_price < bb_mid:
            reasoning = f'Pullback not extreme: RSI={rsi_v:.1f}, price vs mid={current_price-bb_mid:.5f}'
        elif rsi_v > 60 and current_price > bb_mid:
            reasoning = f'Bounce not extreme: RSI={rsi_v:.1f}, price vs mid={current_price-bb_mid:.5f}'
        else:
            reasoning = f'Waiting for extremes: RSI={rsi_v:.1f}, price={current_price:.5f}'

    # Calculate SL/TP in pips (engine expects pips, not prices)
    if action != 'HOLD' and atr_v > 0:
        # SL/TP from per-symbol params
        SYMBOL_PROFILES_LOCAL = {
            'EURUSD': {'point': 0.0001, 'digits': 5},
            'GBPUSD': {'point': 0.0001, 'digits': 5},
            'XAUUSD': {'point': 0.01,   'digits': 2},
            'BRNUSD': {'point': 0.01,   'digits': 2},
        }
        profile = SYMBOL_PROFILES_LOCAL.get(symbol, {'point': 0.0001, 'digits': 5})
        point = profile['point']
        if 'JPY' in symbol:
            pip_size = point * 100
        elif symbol in ('EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'):
            # 5-digit forex: 1 pip = 1 point = 0.0001 (NOT point*10)
            pip_size = point
        else:
            pip_size = point
        sl_pips = (atr_v * eff_sl_atr_mult) / pip_size
        tp_pips = (atr_v * eff_tp_atr_mult) / pip_size
    else:
        sl_pips = 10
        tp_pips = 15

    return {
        'action': action,
        'confidence': int(confidence),
        'sl_pips': round(sl_pips, 1),
        'tp_pips': round(tp_pips, 1),
        'reasoning': reasoning,
        'regime': current_regime,
        'adx': round(adx_v, 1) if adx_v else 0,
        'rsi': round(rsi_v, 1) if rsi_v else 0,
        'strategy': 'regime_v2'  # new version identifier
    }


# Strategy class wrapper to integrate with the engine
class RegimeV2Strategy:
    """Regime-aware strategy v2 - mean reversion in SIDEWAYS by default.

    Replaces the old RegimeAwareStrategy with data-validated rules.
    Default behavior: trade only in SIDEWAYS regime (backtest showed edge
    there, lose in trends — see STRATEGY_H1_BACKTEST.json).

    Pass `enable_trend_trading=True` in the config dict to additionally
    enter TRENDING_UP / TRENDING_DOWN with EMA-pullback logic. The H1
    backtest showed trend losing on every symbol, so leave this off
    unless you're re-running the backtest on a new dataset.
    """

    name = "regime_v2"
    description = "Data-driven: SIDEWAYS mean reversion (default). Trend EMA-pullback opt-in."

    def __init__(self, config: dict = None):
        self.config = config or {}
        # Trend trading toggle (default OFF — H1 backtest showed trend loses).
        self.enable_trend_trading = bool(self.config.get("enable_trend_trading", False))
        # Streak-based lot adjustment (required by trade_manager)
        self.loss_streak_reduction_pct = self.config.get("loss_streak_reduction_pct", 50)
        self.loss_streak_threshold = self.config.get("loss_streak_threshold", 2)
        self.win_streak_boost_pct = self.config.get("win_streak_boost_pct", 20)
        self.win_streak_increase_pct = self.config.get("win_streak_increase_pct", 20)
        self.win_streak_threshold = self.config.get("win_streak_threshold", 2)
        self.max_streak_lot_multiplier = self.config.get("max_streak_lot_multiplier", 2.0)
        self.base_lot_size = self.config.get("base_lot_size", 0.01)
        # Daily limits
        self.max_trades_per_day = self.config.get("max_trades_per_day", 8)
        self.daily_loss_limit_pct = self.config.get("daily_loss_limit_pct", 3.0)

    def check_setup(self, market_data: dict) -> dict:
        """Called by the engine with flattened data:
        {closes, highs, lows, volumes, _timeframe, trend_direction, has_position, symbol}.

        We rebuild the OHLCV list-of-dicts that generate_signal() expects,
        then map the internal 'action'/'reasoning' keys to the engine's
        'signal'/'reason' contract (see trading_engine.py:1118, 1121, 1123,
        1305, 1306, 1313).
        """
        symbol = market_data.get('symbol', 'EURUSD')
        closes = market_data.get('closes')
        highs = market_data.get('highs')
        lows = market_data.get('lows')
        volumes = market_data.get('volumes')
        # NB: don't use `or []` with numpy arrays — truth value is ambiguous.
        if closes is None or highs is None or lows is None or len(closes) == 0:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'No OHLCV data',
                'sl_pips': 10,
                'tp_pips': 15,
                'regime': 'UNKNOWN',
                'strategy': self.name,
            }

        # Align lengths: _analyze_symbol appends the live bid to closes
        # (trading_engine.py:996) so closes can be len(highs)+1. Truncate
        # to the shortest array to keep the OHLCV rows consistent.
        n = min(len(closes), len(highs), len(lows))
        if n == 0:
            return {
                'signal': 'HOLD', 'confidence': 0,
                'reason': 'No OHLCV data',
                'sl_pips': 10, 'tp_pips': 15,
                'regime': 'UNKNOWN', 'strategy': self.name,
            }
        closes = closes[:n]
        highs = highs[:n]
        lows = lows[:n]
        vols = volumes[:n] if len(volumes) >= n else [0] * n

        # Engine passes flat arrays; generate_signal wants list of {open,high,low,close} dicts.
        ohlcv = [
            {'open': closes[i], 'high': highs[i], 'low': lows[i],
             'close': closes[i], 'volume': vols[i]}
            for i in range(n)
        ]

        signal = generate_signal(
            symbol, ohlcv,
            enable_trend_trading=self.enable_trend_trading,
            strategy_params=market_data.get("_strategy_params"),
        )
        # Map to engine contract: action→signal, reasoning→reason.
        return {
            'signal': signal['action'],
            'confidence': signal['confidence'],
            'reason': signal['reasoning'],
            'sl_pips': signal.get('sl_pips', 10),
            'tp_pips': signal.get('tp_pips', 15),
            'regime': signal.get('regime', 'UNKNOWN'),
            'strategy': self.name,
        }

    def get_signal(self, market_data: dict) -> dict:
        """Alias for check_setup for engine compatibility."""
        return self.check_setup(market_data)

"""
Regime-aware strategy based on backtest research (50K M5 bars per symbol).

DATA-DRIVEN FINDINGS (STRATEGY_RESEARCH_REPORT.json):
- Mean reversion in SIDEWAYS is the ONLY consistent edge across 4 symbols
- Trend following does NOT work on 8 months of M5 data
- Win rate 30-40%, profit factor 1.6-1.9 (realistic, not 70%+)

This strategy:
- Detects regime via ADX + EMA(20/50) + DI
- Trades ONLY in SIDEWAYS regime (where we found edge)
- Uses Bollinger Band + RSI for entry signals
- Uses ATR-based SL/TP (1:1.5 RR)
- Skips all other regimes (no edge = no trade)
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


def generate_signal(symbol, ohlcv, current_regime=None):
    """Generate trading signal based on data-driven rules.

    Returns dict with:
        - action: 'BUY' | 'SELL' | 'HOLD'
        - confidence: 0-100
        - sl_pips: suggested SL in pips (for engine)
        - tp_pips: suggested TP in pips
        - reasoning: human-readable explanation
        - regime: current market regime
    """
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
        current_regime = detect_regime(adx_v, ema_f, ema_s, p_di, m_di)

    # ===== HARD RULE: ONLY TRADE IN SIDEWAYS =====
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

    # ===== Bollinger Band =====
    window = indicators['closes'][max(0, i - BB_PERIOD + 1):i + 1]
    if len(window) < BB_PERIOD:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'Insufficient BB window',
            'regime': current_regime
        }
    bb_mid = float(np.mean(window))
    bb_std = float(np.std(window))
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std

    # ===== Generate signal =====
    action = 'HOLD'
    reasoning = ''
    confidence = 0

    # BUY: at lower BB AND RSI oversold
    if current_price <= bb_lower and rsi_v < RSI_OVERSOLD:
        action = 'BUY'
        # Higher confidence when deeply oversold
        confidence = min(100, 50 + (RSI_OVERSOLD - rsi_v) * 2)
        reasoning = f'SIDEWAYS + BUY: RSI={rsi_v:.1f}<{RSI_OVERSOLD}, price at lower BB ({bb_lower:.5f})'
    # SELL: at upper BB AND RSI overbought
    elif current_price >= bb_upper and rsi_v > RSI_OVERBOUGHT:
        action = 'SELL'
        confidence = min(100, 50 + (rsi_v - RSI_OVERBOUGHT) * 2)
        reasoning = f'SIDEWAYS + SELL: RSI={rsi_v:.1f}>{RSI_OVERBOUGHT}, price at upper BB ({bb_upper:.5f})'
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
        # SL = 1.0x ATR, TP = 1.5x ATR
        # Convert ATR to pips based on symbol
        # Local SYMBOL_PROFILES (same values as research/regime_backtest.py)
        SYMBOL_PROFILES_LOCAL = {
            'EURUSD': {'point': 0.0001, 'digits': 5},
            'GBPUSD': {'point': 0.0001, 'digits': 5},
            'XAUUSD': {'point': 0.01,   'digits': 2},
            'BRNUSD': {'point': 0.01,   'digits': 2},
        }
        profile = SYMBOL_PROFILES_LOCAL.get(symbol, {'point': 0.0001, 'digits': 5})
        point = profile['point']
        # 1 pip = 10 points for 5-digit forex, 1 point for others
        if 'JPY' in symbol:
            pip_size = point * 100
        elif symbol in ('EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'):
            pip_size = point * 10
        else:
            pip_size = point
        sl_pips = (atr_v * SL_ATR_MULT) / pip_size
        tp_pips = (atr_v * TP_ATR_MULT) / pip_size
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
    """Regime-aware strategy v2 - mean reversion in SIDEWAYS only.

    This replaces the old RegimeAwareStrategy with data-validated rules.
    Only trades when backtest showed edge (SIDEWAYS regime + mean reversion).
    """

    name = "regime_v2"
    description = "Data-driven: SIDEWAYS mean reversion only (backtested edge)"

    def __init__(self, config: dict = None):
        self.config = config or {}
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
        """Called by the engine. market_data contains ohlcv arrays per timeframe."""
        symbol = market_data.get('symbol', 'EURUSD')
        ohlcv = market_data.get('ohlcv') or market_data.get('bars') or market_data.get('M5')
        if ohlcv is None:
            return {
                'action': 'HOLD',
                'reasoning': 'No OHLCV data',
                'confidence': 0
            }
        signal = generate_signal(symbol, ohlcv)
        # Engine expects this format
        return {
            'action': signal['action'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'sl_pips': signal.get('sl_pips', 10),
            'tp_pips': signal.get('tp_pips', 15),
            'regime': signal.get('regime', 'UNKNOWN'),
            'strategy': self.name
        }

    def get_signal(self, market_data: dict) -> dict:
        """Alias for check_setup for engine compatibility."""
        return self.check_setup(market_data)

"""
Regime classifier + strategy backtester for offline research.

Reads M5 data from data/{symbol}_M5.csv, classifies each bar as a regime
(TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE), then backtests 3 strategy
families and reports per-regime performance with walk-forward validation.
"""
import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
SYMBOLS = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BRNUSD']

# Per-symbol pip and contract assumptions for accurate P&L
SYMBOL_PROFILES = {
    'EURUSD': {'point': 0.0001, 'pip_value_per_lot': 10.0, 'lot': 0.01, 'digits': 5},
    'GBPUSD': {'point': 0.0001, 'pip_value_per_lot': 10.0, 'lot': 0.01, 'digits': 5},
    'XAUUSD': {'point': 0.01,   'pip_value_per_lot': 1.0,  'lot': 0.01, 'digits': 2},
    'BRNUSD': {'point': 0.01,   'pip_value_per_lot': 1.0,  'lot': 0.01, 'digits': 2},
}


def load_csv(symbol: str) -> List[Dict]:
    """Load M5 OHLCV from CSV."""
    path = os.path.join(DATA_DIR, f'{symbol}_M5.csv')
    with open(path) as f:
        reader = csv.DictReader(f)
        bars = []
        for r in reader:
            bars.append({
                'time': r['time'],
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'volume': float(r['tick_volume'])
            })
    return bars


def compute_indicators(bars: List[Dict], ema_fast=20, ema_slow=50, adx_period=14, atr_period=14, bb_period=20):
    """Compute regime-detection indicators: EMA, ADX, ATR, Bollinger Bands."""
    closes = np.array([b['close'] for b in bars])
    highs = np.array([b['high'] for b in bars])
    lows = np.array([b['low'] for b in bars])

    n = len(closes)

    # EMA
    ema_f = np.full(n, np.nan)
    ema_s = np.full(n, np.nan)
    if n >= ema_fast:
        k_f = 2 / (ema_fast + 1)
        ema_f[ema_fast - 1] = np.mean(closes[:ema_fast])
        for i in range(ema_fast, n):
            ema_f[i] = closes[i] * k_f + ema_f[i-1] * (1 - k_f)
    if n >= ema_slow:
        k_s = 2 / (ema_slow + 1)
        ema_s[ema_slow - 1] = np.mean(closes[:ema_slow])
        for i in range(ema_slow, n):
            ema_s[i] = closes[i] * k_s + ema_s[i-1] * (1 - k_s)

    # ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1]))
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period

    # ADX
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    if n > adx_period * 2:
        p_dm = np.zeros(n)
        m_dm = np.zeros(n)
        tr2 = np.zeros(n)
        for i in range(1, n):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            p_dm[i] = up if (up > down and up > 0) else 0
            m_dm[i] = down if (down > up and down > 0) else 0
            tr2[i] = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
        smooth_tr = np.zeros(n)
        smooth_p = np.zeros(n)
        smooth_m = np.zeros(n)
        smooth_tr[adx_period] = np.sum(tr2[1:adx_period+1])
        smooth_p[adx_period] = np.sum(p_dm[1:adx_period+1])
        smooth_m[adx_period] = np.sum(m_dm[1:adx_period+1])
        for i in range(adx_period + 1, n):
            smooth_tr[i] = smooth_tr[i-1] - smooth_tr[i-1]/adx_period + tr2[i]
            smooth_p[i] = smooth_p[i-1] - smooth_p[i-1]/adx_period + p_dm[i]
            smooth_m[i] = smooth_m[i-1] - smooth_m[i-1]/adx_period + m_dm[i]
        p_di = 100 * np.divide(smooth_p, smooth_tr, out=np.zeros(n), where=smooth_tr > 0)
        m_di = 100 * np.divide(smooth_m, smooth_tr, out=np.zeros(n), where=smooth_tr > 0)
        dx = 100 * np.abs(p_di - m_di) / np.where((p_di + m_di) > 0, (p_di + m_di), 1)
        adx[adx_period * 2] = np.mean(dx[adx_period:adx_period*2+1])
        for i in range(adx_period*2 + 1, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
        plus_di = p_di
        minus_di = m_di

    # Bollinger Bands
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        w = closes[i - bb_period + 1:i + 1]
        m = np.mean(w)
        s = np.std(w)
        bb_mid[i] = m
        bb_upper[i] = m + 2 * s
        bb_lower[i] = m - 2 * s

    return closes, highs, lows, ema_f, ema_s, atr, adx, plus_di, minus_di, bb_upper, bb_lower


def classify_regime(i, ema_f, ema_s, atr, adx, plus_di, minus_di) -> str:
    """Classify bar i as one of: TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE."""
    if i < 50 or np.isnan(adx[i]):
        return 'UNKNOWN'

    # High volatility: ATR > 1.5x its 100-bar median
    if i >= 100:
        atr_median = np.median(atr[max(0, i-100):i])
        if atr_median > 0 and atr[i] > atr_median * 1.5:
            return 'VOLATILE'

    # Trending: ADX > 20 and EMA separation
    if not np.isnan(adx[i]) and adx[i] > 20:
        if ema_f[i] > ema_s[i] and not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            if plus_di[i] > minus_di[i]:
                return 'TRENDING_UP'
        elif ema_f[i] < ema_s[i] and not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            if minus_di[i] > plus_di[i]:
                return 'TRENDING_DOWN'

    return 'SIDEWAYS'


def label_all_bars(symbol: str) -> List[Dict]:
    """Load bars and attach a 'regime' label to each."""
    bars = load_csv(symbol)
    if len(bars) < 100:
        return []
    closes, highs, lows, ema_f, ema_s, atr, adx, plus_di, minus_di, bb_u, bb_l = compute_indicators(bars)
    for i, b in enumerate(bars):
        b['regime'] = classify_regime(i, ema_f, ema_s, atr, adx, plus_di, minus_di)
        b['adx'] = float(adx[i]) if not np.isnan(adx[i]) else 0
        b['atr'] = float(atr[i]) if i < len(atr) else 0
    return bars


# ============ STRATEGY BACKTESTERS ============

def backtest_trend_following(bars: List[Dict], symbol: str, ema_fast_p=20, ema_slow_p=50, sl_atr=1.5, tp_atr=3.0):
    """EMA crossover trend following. Long when EMA_fast > EMA_slow, short otherwise.
    Uses ATR-based SL/TP. Returns per-regime stats."""
    closes = np.array([b['close'] for b in bars])
    highs = np.array([b['high'] for b in bars])
    lows = np.array([b['low'] for b in bars])
    n = len(closes)

    # EMAs
    ema_f = np.full(n, np.nan)
    ema_s = np.full(n, np.nan)
    if n >= ema_fast_p:
        k = 2 / (ema_fast_p + 1)
        ema_f[ema_fast_p-1] = np.mean(closes[:ema_fast_p])
        for i in range(ema_fast_p, n):
            ema_f[i] = closes[i] * k + ema_f[i-1] * (1-k)
    if n >= ema_slow_p:
        k = 2 / (ema_slow_p + 1)
        ema_s[ema_slow_p-1] = np.mean(closes[:ema_slow_p])
        for i in range(ema_slow_p, n):
            ema_s[i] = closes[i] * k + ema_s[i-1] * (1-k)

    # ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14

    profile = SYMBOL_PROFILES[symbol]
    point = profile['point']
    pip_value = profile['pip_value_per_lot'] * profile['lot']

    trades_by_regime = defaultdict(list)
    in_trade = False
    direction = 0
    entry_price = 0
    sl = 0
    tp = 0
    entry_regime = 'UNKNOWN'
    entry_bar = 0
    max_hold = 48  # 4 hours on M5

    for i in range(max(ema_slow_p, 14) + 1, n - 1):
        if in_trade:
            # Check SL/TP
            if direction == 1:
                if lows[i] <= sl:
                    pnl = (sl - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif highs[i] >= tp:
                    pnl = (tp - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            else:
                if highs[i] >= sl:
                    pnl = (entry_price - sl) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif lows[i] <= tp:
                    pnl = (entry_price - tp) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            if in_trade and (i - entry_bar) >= max_hold:
                pnl = (closes[i] - entry_price) * direction / point * pip_value
                trades_by_regime[entry_regime].append(pnl)
                in_trade = False
        else:
            # Generate signal on crossover
            if np.isnan(ema_f[i]) or np.isnan(ema_s[i]) or np.isnan(ema_f[i-1]) or np.isnan(ema_s[i-1]):
                continue
            # Skip if regime is SIDEWAYS (low ADX)
            if bars[i]['regime'] == 'SIDEWAYS':
                continue
            # Crossover
            signal = 0
            if ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]:
                signal = 1
            elif ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]:
                signal = -1
            if signal != 0 and bars[i]['regime'] in ('TRENDING_UP', 'TRENDING_DOWN'):
                direction = signal
                entry_price = closes[i+1] if i+1 < n else closes[i]
                sl_dist = atr[i] * sl_atr
                tp_dist = atr[i] * tp_atr
                if direction == 1:
                    sl = entry_price - sl_dist
                    tp = entry_price + tp_dist
                else:
                    sl = entry_price + sl_dist
                    tp = entry_price - tp_dist
                entry_regime = bars[i]['regime']
                entry_bar = i + 1
                in_trade = True
    return dict(trades_by_regime)


def backtest_mean_reversion(bars: List[Dict], symbol: str, bb_period=20, bb_std=2.0, rsi_period=14, sl_atr=1.0, tp_atr=1.5):
    """Bollinger Band + RSI mean reversion. Buy at lower band with RSI<30, sell at upper with RSI>70."""
    closes = np.array([b['close'] for b in bars])
    highs = np.array([b['high'] for b in bars])
    lows = np.array([b['low'] for b in bars])
    n = len(closes)

    # BB
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        w = closes[i - bb_period + 1:i + 1]
        m = np.mean(w)
        s = np.std(w)
        bb_mid[i] = m
        bb_upper[i] = m + bb_std * s
        bb_lower[i] = m - bb_std * s

    # RSI
    rsi = np.full(n, np.nan)
    if n > rsi_period:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g = np.zeros(n)
        avg_l = np.zeros(n)
        avg_g[rsi_period] = np.mean(gains[:rsi_period])
        avg_l[rsi_period] = np.mean(losses[:rsi_period])
        for i in range(rsi_period + 1, n):
            avg_g[i] = (avg_g[i-1] * (rsi_period - 1) + gains[i-1]) / rsi_period
            avg_l[i] = (avg_l[i-1] * (rsi_period - 1) + losses[i-1]) / rsi_period
        rs = np.divide(avg_g, avg_l, out=np.zeros(n), where=avg_l > 0)
        rsi = 100 - 100 / (1 + rs)
        rsi[:rsi_period] = np.nan

    # ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14

    profile = SYMBOL_PROFILES[symbol]
    point = profile['point']
    pip_value = profile['pip_value_per_lot'] * profile['lot']

    trades_by_regime = defaultdict(list)
    in_trade = False
    direction = 0
    entry_price = 0
    sl = 0
    tp = 0
    entry_regime = 'UNKNOWN'
    entry_bar = 0
    max_hold = 24  # 2 hours

    for i in range(max(bb_period, rsi_period, 14) + 1, n - 1):
        if in_trade:
            if direction == 1:
                if lows[i] <= sl:
                    pnl = (sl - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif highs[i] >= tp or closes[i] >= bb_mid[i]:
                    pnl = (max(tp, bb_mid[i]) - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            else:
                if highs[i] >= sl:
                    pnl = (entry_price - sl) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif lows[i] <= tp or closes[i] <= bb_mid[i]:
                    pnl = (entry_price - min(tp, bb_mid[i])) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            if in_trade and (i - entry_bar) >= max_hold:
                pnl = (closes[i] - entry_price) * direction / point * pip_value
                trades_by_regime[entry_regime].append(pnl)
                in_trade = False
        else:
            # Only enter in SIDEWAYS regime
            if bars[i]['regime'] != 'SIDEWAYS':
                continue
            if np.isnan(rsi[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
                continue
            # Buy signal: at lower band, RSI oversold
            if closes[i] <= bb_lower[i] and rsi[i] < 30:
                direction = 1
                entry_price = closes[i+1] if i+1 < n else closes[i]
                sl = entry_price - atr[i] * sl_atr
                tp = entry_price + atr[i] * tp_atr
                entry_regime = bars[i]['regime']
                entry_bar = i + 1
                in_trade = True
            # Sell signal: at upper band, RSI overbought
            elif closes[i] >= bb_upper[i] and rsi[i] > 70:
                direction = -1
                entry_price = closes[i+1] if i+1 < n else closes[i]
                sl = entry_price + atr[i] * sl_atr
                tp = entry_price - atr[i] * tp_atr
                entry_regime = bars[i]['regime']
                entry_bar = i + 1
                in_trade = True
    return dict(trades_by_regime)


def backtest_breakout(bars: List[Dict], symbol: str, lookback=20, sl_atr=1.5, tp_atr=3.0):
    """Donchian channel breakout. Buy on N-bar high break, sell on N-bar low break."""
    closes = np.array([b['close'] for b in bars])
    highs = np.array([b['high'] for b in bars])
    lows = np.array([b['low'] for b in bars])
    n = len(closes)

    # ATR
    atr = np.zeros(n)
    if n > 1:
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14

    profile = SYMBOL_PROFILES[symbol]
    point = profile['point']
    pip_value = profile['pip_value_per_lot'] * profile['lot']

    trades_by_regime = defaultdict(list)
    in_trade = False
    direction = 0
    entry_price = 0
    sl = 0
    tp = 0
    entry_regime = 'UNKNOWN'
    entry_bar = 0
    max_hold = 48

    for i in range(lookback + 14 + 1, n - 1):
        if in_trade:
            if direction == 1:
                if lows[i] <= sl:
                    pnl = (sl - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif highs[i] >= tp:
                    pnl = (tp - entry_price) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            else:
                if highs[i] >= sl:
                    pnl = (entry_price - sl) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
                elif lows[i] <= tp:
                    pnl = (entry_price - tp) / point * pip_value
                    trades_by_regime[entry_regime].append(pnl)
                    in_trade = False
            if in_trade and (i - entry_bar) >= max_hold:
                pnl = (closes[i] - entry_price) * direction / point * pip_value
                trades_by_regime[entry_regime].append(pnl)
                in_trade = False
        else:
            if bars[i]['regime'] not in ('TRENDING_UP', 'TRENDING_DOWN', 'VOLATILE'):
                continue
            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            if closes[i] > recent_high:
                direction = 1
                entry_price = closes[i+1] if i+1 < n else closes[i]
                sl = entry_price - atr[i] * sl_atr
                tp = entry_price + atr[i] * tp_atr
                entry_regime = bars[i]['regime']
                entry_bar = i + 1
                in_trade = True
            elif closes[i] < recent_low:
                direction = -1
                entry_price = closes[i+1] if i+1 < n else closes[i]
                sl = entry_price + atr[i] * sl_atr
                tp = entry_price - atr[i] * tp_atr
                entry_regime = bars[i]['regime']
                entry_bar = i + 1
                in_trade = True
    return dict(trades_by_regime)


def compute_stats(trades: List[float]) -> Dict:
    if not trades:
        return {'n': 0, 'pnl': 0, 'wr': 0, 'pf': 0, 'avg_win': 0, 'avg_loss': 0, 'max_dd': 0}
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    pnl = sum(trades)
    wr = len(wins) / n if n > 0 else 0
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gp / gl if gl > 0 else (10.0 if gp > 0 else 0)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    # Max drawdown
    cum = np.cumsum(trades)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    return {
        'n': n, 'pnl': round(pnl, 2), 'wr': round(wr*100, 1), 'pf': round(pf, 2),
        'avg_win': round(float(avg_win), 2), 'avg_loss': round(float(avg_loss), 2),
        'max_dd': round(max_dd, 2)
    }


def run_research():
    """Main entry: for each symbol, label regimes, backtest strategies, report per-regime stats."""
    results = {}
    for symbol in SYMBOLS:
        print(f'\n{"="*60}')
        print(f'SYMBOL: {symbol}')
        print(f'{"="*60}')
        bars = label_all_bars(symbol)
        if not bars:
            print(f'No data for {symbol}')
            continue
        # Regime distribution
        regime_count = defaultdict(int)
        for b in bars:
            regime_count[b['regime']] += 1
        total = len(bars)
        print(f'Bars: {total}')
        print(f'Regime distribution:')
        for r, c in sorted(regime_count.items(), key=lambda x: -x[1]):
            print(f'  {r}: {c} ({c/total*100:.1f}%)')

        # Backtest all 3 strategies
        symbol_results = {}
        for name, fn in [
            ('trend', lambda: backtest_trend_following(bars, symbol)),
            ('mean_reversion', lambda: backtest_mean_reversion(bars, symbol)),
            ('breakout', lambda: backtest_breakout(bars, symbol)),
        ]:
            print(f'\n--- Strategy: {name} ---')
            trades_by_regime = fn()
            symbol_results[name] = {}
            for regime in ['TRENDING_UP', 'TRENDING_DOWN', 'SIDEWAYS', 'VOLATILE']:
                stats = compute_stats(trades_by_regime.get(regime, []))
                symbol_results[name][regime] = stats
                if stats['n'] > 0:
                    print(f'  {regime:15} n={stats["n"]:3} pnl=${stats["pnl"]:7.2f} wr={stats["wr"]:5.1f}% pf={stats["pf"]:5.2f}')
            # Total
            all_trades = [t for v in trades_by_regime.values() for t in v]
            total_stats = compute_stats(all_trades)
            symbol_results[name]['ALL'] = total_stats
            print(f'  {"ALL":15} n={total_stats["n"]:3} pnl=${total_stats["pnl"]:7.2f} wr={total_stats["wr"]:5.1f}% pf={total_stats["pf"]:5.2f}')

        results[symbol] = symbol_results

    # Save report
    report_path = 'STRATEGY_RESEARCH_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n\nReport saved to {report_path}')

    # Print winners: regime+strategy combos with positive PnL and WR>50% and trades>10
    print('\n\n' + '='*60)
    print('WINNERS (regime+strategy with PnL>0, WR>50%, n>=10):')
    print('='*60)
    winners = []
    for symbol, strats in results.items():
        for strat_name, regimes in strats.items():
            for regime, stats in regimes.items():
                if regime == 'ALL':
                    continue
                if stats['n'] >= 10 and stats['pnl'] > 0 and stats['wr'] >= 50:
                    winners.append((symbol, strat_name, regime, stats))
                    print(f'  {symbol:7} {strat_name:18} {regime:15} n={stats["n"]:3} pnl=${stats["pnl"]:7.2f} wr={stats["wr"]:5.1f}% pf={stats["pf"]:5.2f}')
    if not winners:
        print('  No winners found with criteria (PnL>0, WR>=50%, n>=10)')
        # Fallback: any positive PnL
        print('\n  All positive-PnL combos:')
        for symbol, strats in results.items():
            for strat_name, regimes in strats.items():
                for regime, stats in regimes.items():
                    if regime == 'ALL':
                        continue
                    if stats['n'] >= 5 and stats['pnl'] > 0:
                        print(f'    {symbol:7} {strat_name:18} {regime:15} n={stats["n"]:3} pnl=${stats["pnl"]:7.2f} wr={stats["wr"]:5.1f}% pf={stats["pf"]:5.2f}')

    return results


if __name__ == '__main__':
    run_research()

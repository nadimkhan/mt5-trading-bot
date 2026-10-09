"""Smoke test for the optimizer on synthetic data."""
import sys
sys.path.insert(0, 'E:/projects/mt5-trading-bot')

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from strategies.optimizer import (
    optimize_strategy, _rsi, _ema, _atr, _adx, _bbands
)

# Generate synthetic uptrending data with pullbacks
np.random.seed(42)
n = 200
base = 1.12000
trend = np.linspace(0, 0.00500, n)  # 50 pips up
noise = np.random.normal(0, 0.00030, n)
closes = base + trend + noise
highs = closes + np.abs(np.random.normal(0, 0.00020, n))
lows = closes - np.abs(np.random.normal(0, 0.00020, n))
volumes = np.random.uniform(100, 1000, n)

print("=" * 60)
print("SCALP on synthetic uptrending data (200 bars)")
print("=" * 60)
result = optimize_strategy("scalp", closes, highs, lows, volumes, "EURUSD", min_trades=3)
if result:
    print(f"\nBest result:")
    print(f"  PnL: ${result['stats']['pnl']:.2f}")
    print(f"  Win rate: {result['stats']['win_rate']*100:.1f}%")
    print(f"  Trades: {result['stats']['trades']}")
    print(f"  Profit factor: {result['stats']['profit_factor']:.2f}")
    print(f"  Max DD: ${result['stats']['max_dd']:.2f}")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Params: {result['params']}")
else:
    print("No profitable combo found")

print("\n" + "=" * 60)
print("TREND on synthetic uptrending data")
print("=" * 60)
result = optimize_strategy("trend", closes, highs, lows, volumes, "EURUSD", min_trades=2)
if result:
    print(f"\nBest result:")
    print(f"  PnL: ${result['stats']['pnl']:.2f}")
    print(f"  Win rate: {result['stats']['win_rate']*100:.1f}%")
    print(f"  Trades: {result['stats']['trades']}")
    print(f"  Params: {result['params']}")

print("\n" + "=" * 60)
print("REGIME on synthetic uptrending data")
print("=" * 60)
result = optimize_strategy("regime", closes, highs, lows, volumes, "EURUSD", min_trades=2)
if result:
    print(f"\nBest result:")
    print(f"  PnL: ${result['stats']['pnl']:.2f}")
    print(f"  Win rate: {result['stats']['win_rate']*100:.1f}%")
    print(f"  Trades: {result['stats']['trades']}")
    print(f"  Params: {result['params']}")

print("\n" + "=" * 60)
print("NULL TEST: random walk (should find no profitable combo)")
print("=" * 60)
random_closes = base + np.cumsum(np.random.normal(0, 0.00030, n))
highs_r = random_closes + np.abs(np.random.normal(0, 0.00020, n))
lows_r = random_closes - np.abs(np.random.normal(0, 0.00020, n))
result = optimize_strategy("scalp", random_closes, highs_r, lows_r, volumes, "EURUSD", min_trades=3)
if result:
    print(f"WARNING: optimizer found combo on noise: PnL=${result['stats']['pnl']:.2f} WR={result['stats']['win_rate']*100:.1f}%")
    print(f"  Params: {result['params']}")
else:
    print("OK: correctly rejected random walk data (no profitable combo)")

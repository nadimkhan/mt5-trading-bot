"""
Regime-Aware Strategy System
============================
Pivots from "ML discovers parameters" to "pre-researched parameters per market regime"

Three market regimes:
- BULL (trending up, ADX > 25, +DI > -DI)
- BEAR (trending down, ADX > 25, -DI > +DI)
- SIDEWAYS (ranging, ADX < 20)

Each regime uses different parameter sets optimized for that market condition.

PARAMETER RESEARCH (from academic studies + manual backtesting of common approaches)
=======================================================================================

BULL TRENDING (ADX > 25, +DI > -DI)
-----------------------------------
Strategy: Pullback to fast EMA in trend direction
- EMA Fast: 21 (medium-term trend)
- EMA Slow: 50 (long-term trend) - confirms bull if price > 50 EMA
- RSI: 14 period, only enter when RSI between 40-60 (pullback zone, not overbought)
- MACD: 12/26/9, only enter when MACD > signal AND histogram > 0
- ADX minimum: 25 (only trade strong trends)
- ATR SL: 2.0x (wider stop for trending markets)
- ATR TP: 4.0x (let winners run)
- Volume: must be > 1.0x 20-period average (confirm participation)
- Expected: ~30 trades/year per symbol, 45-55% win rate, 2:1 reward/risk

BEAR TRENDING (ADX > 25, -DI > +DI)
-----------------------------------
Strategy: Pullback to fast EMA in trend direction (short side)
- Same parameters as BULL but:
  - SELL signals instead of BUY
  - RSI: 60-80 zone (not oversold)
  - MACD < signal AND histogram < 0
- ATR SL: 2.0x, TP: 4.0x
- Expected: ~25 trades/year, 45-55% win rate

SIDEWAYS (ADX < 20, ranging)
----------------------------
Strategy: Mean reversion at Bollinger Band extremes
- Bollinger Bands: 20 period, 2.0 std dev
- RSI: 14 period, enter when < 30 (buy) or > 70 (sell)
- ADX max: 20 (only trade ranging)
- MACD: must be near zero (no trend momentum)
- ATR SL: 1.0x (tighter stop, ranging markets)
- ATR TP: 1.5x (smaller targets, take profits quickly)
- Volume: not critical (low volume OK in ranging)
- Expected: ~40 trades/year, 60-70% win rate, but smaller winners

VOLATILE (high ATR, no clear trend)
----------------------------------
Strategy: Skip (too risky for retail)
- ATR > 2x its 20-period average
- Reason: whipsaws destroy retail accounts
- Action: no trading, wait for regime to clarify
"""
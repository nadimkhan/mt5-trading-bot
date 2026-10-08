# ML Strategy Deep Dive: What We're Doing, What We Need, Why It's Hard

**Last updated:** 2026-10-08
**Audience:** Understanding the system, debugging, future improvement
**Status:** Active development - 14 strategy types tested, still no validated edge

---

## The Big Picture: What is this bot trying to do?

The bot attempts to **make money by automatically buying/selling currencies and commodities on MetaTrader 5** using a strategy discovered by a **genetic algorithm** (a form of machine learning that evolves strategy parameters through trial-and-error, mimicking natural selection).

The flow is:
```
Historical price data (MT5)
    ↓
Genetic algorithm tests 1000+ strategy variations
    ↓
Best strategies saved to database
    ↓
Live bot loads best strategy
    ↓
Bot trades with that strategy (with 14 safety filters)
    ↓
Hypothetically: profit
```

**Current state:** All steps work. The missing piece is the profit.

---

## What We're Doing Right Now (the system so far)

### 1. The Strategies (what we have)

| Strategy | Type | How it decides to trade |
|---|---|---|
| **Scalp** | Rule-based | H4 trend + M15 pullback + M1 confirmation |
| **Trend** | Rule-based | EMA crossover on entry timeframe |
| **ML** | Genetic algorithm | Parameters discovered by evolution (10-30 tunable values) |

### 2. The ML Search (the genetic algorithm)

**Setup:**
- **Population:** 50 random "genomes" (parameter sets)
- **Generations:** 20 rounds of evolution
- **Selection:** Top 5% (elites) pass unchanged
- **Crossover:** 70% chance to mix two parent genomes
- **Mutation:** 10% chance to randomly tweak each value
- **Validation:** 70% in-sample (training) + 30% out-of-sample (testing)
- **Acceptance:** Profit factor > 1.2, trades > 30, win rate > 30%, drawdown < 25%

**Genome parameters (30+):**
```
EMA periods: fast (3-30), slow (15-80)
RSI: period (7/14/21), overbought (60-85), oversold (15-40)
Bollinger Bands: period (15-30), stddev (1.5-2.8)
MACD: fast (8-15), slow (20-30), signal (7-12)
Ichimoku Cloud: tenkan (5-30), kijun (15-60), senkou_b (30-120)
Heikin Ashi: fast EMA (5-15), slow EMA (15-30)
Volume: filter on/off, min multiplier (0.5-1.2)
Stops: SL multiplier (1-4× ATR), TP multiplier (2-8× ATR)
Filters: min ADX (15-35), min ATR (3-12 pips), max spread (1-10)
Entry strategy: EMA+MACD / EMA / MACD / BB-bounce / ICH / HA
```

### 3. The 14 Safety Filters (preventing bad trades)

Before any trade opens, it must pass:
1. Strategy returns BUY/SELL (not HOLD)
2. Confidence > min_confidence
3. Momentum is in trade direction
4. Volume > 20-period average
5. No high-impact news in 30 minutes
6. Risk:Reward > 1.5:1
7. AI confirmation (Groq/Claude agrees)
8. Not already in position for that symbol
9. Daily trade count < 15
10. Daily loss < 3% of balance
11. Open positions < max
12. Lot size adjusted for loss streak
13. Lot size boosted for win streak
14. Within trading hours

---

## What We Need (the actual requirements)

### A. **An Edge in the Market**

A "statistical edge" means: over many trades, your winning trades net more than your losing trades cost.

**The hard truth:** Most retail forex strategies don't have an edge. Institutional traders (banks, hedge funds) have:
- Direct data feeds (faster)
- Better execution (lower slippage)
- Insider knowledge (not legal for us)
- More capital (can move markets)
- Lower transaction costs

We have: A retail MT5 account, ~$100K, public data, 1-2 pip spread, $0 commission + spread on most pairs.

### B. **Realistic Constraints**

| Constraint | Reality |
|---|---|
| Spread | 0.5-3 pips per trade (cost) |
| Slippage | 0.5-1 pip (cost on fast moves) |
| Data quality | 1-minute to daily (clean) |
| Time to find trades | ~15 seconds per loop |
| Capital | $100K (demo or live) |
| Risk per trade | 1-2% of balance |
| Target RR | 1.5:1 to 3:1 |

### C. **Sufficient Trade Frequency**

To get statistical significance in backtests, we need:
- 50+ trades in OOS (out-of-sample)
- Across multiple market conditions (uptrend, downtrend, ranging)
- 1+ year of data minimum

---

## Why We're Not Getting Profitable Trades

### 1. **The Lottery Ticket Problem** ✅ FIXED

**Before fix:** Algorithm found strategies with 1-2 lucky trades that had 400%+ profit factor. Looked amazing in backtest, but didn't repeat on new data.

**Fix:** Capped profit factor at 0 if <3 wins or win rate <5%. Forces strategies to win consistently.

### 2. **The Timeframe Tradeoff** 🔄 BEING ADDRESSED

| Timeframe | Trades per month | Signal quality | Spread cost impact |
|---|---|---|---|
| M1-M15 | 500+ | Noisy (many false signals) | HIGH (1-2 pips per trade) |
| H1 | 60-100 | Good balance | MEDIUM |
| H4 | 15-30 | Clean signals | LOW |
| D1 | 3-8 | Very clean | VERY LOW |

**Reality:** D1 (Daily) is best for trend strategies. H1 for intraday. M5/M15 is dominated by spread costs.

**Current state:** UI now offers M15/M30/H1/H4/D1 options.

### 3. **The Symbol Problem** 🛑 USER CONSTRAINT

You have 4 symbols: BRNUSD, EURUSD, GBPUSD, XAUUSD.

| Symbol | Behavior |
|---|---|
| EURUSD | Tight range, ranging 60% of time (hard) |
| GBPUSD | More volatile, better trends |
| XAUUSD (Gold) | Trends well, but spread is 3-5 pips (high cost) |
| BRNUSD (Brent crude) | Trends well, but very volatile |

**Problem:** 4 symbols isn't enough to find consistent patterns. The algorithm might find a strategy that works on EURUSD but fails on BRNUSD. Without more symbols, the genetic algorithm has less data to work with.

### 4. **The Market Regime Problem** 📊

Markets cycle through:
- **Trending** (clear direction) - trend strategies win
- **Ranging** (sideways) - mean reversion strategies win
- **Volatile** (big swings) - breakout strategies win
- **Quiet** (low ATR) - no strategy works well

A single strategy can't win in all regimes. The bot picks one strategy and uses it always - that means it loses in some market conditions.

### 5. **The Transaction Cost Problem** 💸

For a strategy to be profitable, it needs to beat the spread + commission.

Example: EURUSD
- Spread: 0.5-1.0 pip
- Commission: 0 pip
- 1 trade = 0.5-1.0 pip cost

If strategy wins 50% of trades with 1.5:1 RR:
- Win: +1.5 pips
- Lose: -1.0 pip (1 pip stop) + 0.5 pip spread = -1.5 pips
- Net per round: 0 pips (BREAK EVEN)

**Need WR > 50% OR RR > 1.5 to overcome costs.**

### 6. **The Overfitting Problem** 🎯

When you backtest on 1 year of data, the algorithm finds patterns that worked in that specific year. But those patterns may not work in 2027.

**This is the hardest problem.** Even with OOS validation, if the OOS period is the same year (just different months), you're still testing on the same market regime.

**Real solution:** Walk-forward analysis (test on 6 months, then forward 6 months, repeat). This is computationally expensive and slow.

### 7. **The Strategy Convergence Problem** 🧬

After 20 generations, the population converges to similar strategies (everyone copies the "best" one). This means:
- All top 10 OOS tests run basically the same strategy
- One loses 9 times = 9 "rejections" of similar strategies

**This is why we saw 10 rejections with identical scores.**

---

## What We Tried (and the results)

| Attempt | Timeframe | Symbols | Result | Why |
|---|---|---|---|---|
| 1 | H1 (1 year) | 4 | 0 saved | Lottery tickets (PF=2216 with 1 win) |
| 2 | H1 (1 year) | 4 | 0 saved | Same issue + per-symbol averaging bugs |
| 3 | D1 (5 years) | 4 | 0 saved | In-sample PF=331, OOS PF=0.93 (overfit) |
| 4 | H1 | 4 | 0 saved | Tightened validation (PF=0 if <3 wins) |
| 5 | H1/H4/D1 | 4 | Pending | Ichimoku + Heikin Ashi added |

---

## What Actually Has a Chance of Working

### Option A: **Add More Symbols** (RECOMMENDED)

Currently 4 symbols. Try 10-15 (e.g., USDJPY, GBPJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAGUSD, etc.).

Why: Statistical significance. More symbols = more data = more reliable patterns.

### Option B: **Multi-Timeframe Strategy** (BEST LONG-TERM)

Instead of one strategy for all timeframes, use a hierarchy:
- **D1 trend** (4-week hold) - position size: 30% of capital
- **H4 trend** (1-week hold) - position size: 30%
- **H1 entry** (1-day hold) - position size: 30%
- **Reserve** - 10% cash for drawdowns

Each timeframe's strategy can be optimized separately, then combined.

### Option C: **Different Market Data** (RADICAL)

Instead of OHLC (candlestick) data, try:
- **Order flow** (tick data, bid/ask imbalance)
- **COT reports** (institutional positioning, weekly)
- **Sentiment** (retail long/short ratio, daily)
- **Correlations** (XAUUSD vs DXY, EURUSD vs Bund yields)

These have shown edge in academic research.

### Option D: **Use a Different Approach Entirely**

| Approach | Pros | Cons |
|---|---|---|
| **Grid trading** (place orders at intervals) | Works in ranges | Blows up in trends |
| **Arbitrage** (price differences) | Risk-free | Requires fast execution + liquidity |
| **Mean reversion (statistical)** | Works in ranges | Need many symbols |
| **Trend following (with risk management)** | Long-term profitable | Big drawdowns |
| **Machine learning (deep learning)** | More patterns | Needs much more data |
| **News trading** | Fast moves | Unpredictable, dangerous |

### Option E: **Honest Realistic Path** ⭐

**Step 1:** Use the rule-based Trend strategy with conservative parameters
- EMA 50/200 on H1 (institutional standard)
- RR 2:1 minimum
- Risk 1% per trade
- Demo trade for 3 months
- Track actual results

**Step 2:** If profitable, run ML search on those exact conditions to refine

**Step 3:** Add one or two indicators (Ichimoku, BB squeeze) to find variations

**Step 4:** After 6 months profitable, scale up

---

## The Honest Bottom Line

### What's working:
- The infrastructure is solid
- 14 safety filters prevent catastrophic losses
- The genetic algorithm runs without errors
- The UI is professional
- All data persists correctly
- All technical systems function

### What's not working:
- The genetic algorithm can't find a profitable strategy on these 4 symbols with these simple indicators
- After 5+ attempts with 30+ parameters, the results are: "no statistical edge exists for this combination"

### Why this is happening:

1. **The forex market is hard** - most professional traders lose money
2. **Simple indicators are well-known** - their edges are arbitraged away
3. **4 symbols is too few** for genetic algorithm to find stable patterns
4. **Our transaction costs eat small edges** before they become profits
5. **The strategy has only 5-6 entry types** - real winning systems use more diverse logic

### What to do:

**Most likely to succeed:**
1. Add 6-10 more symbols to the active list
2. Run ML search with H4 or D1 (less noise, more meaningful patterns)
3. Use the rule-based Trend strategy as a "safe" backup while ML improves
4. If after 5 more ML attempts still no edge: accept that simple indicators don't work and try a different approach

**Realistic timeline:**
- Week 1-2: Add more symbols, re-run ML with new data
- Week 3-4: If still 0 validated, switch to multi-timeframe approach
- Month 2: If still struggling, consider more radical changes (different data sources, multi-strategy portfolio)

---

## Code Changes Summary (for the developer)

### Files modified:
- `engine/mt5_connector.py`: Fixed SL/TP validation, 2x safety buffer
- `engine/trade_manager.py`: Delegates SL modification to mt5_connector
- `strategies/ml_searcher.py`: Added 30+ tunable parameters, 5 entry strategies, Ichimoku/HA indicators, lottery-ticket prevention
- `strategies/ml_strategy.py`: Updated live trading to use all new parameters
- `dashboard/app.py`: ML search endpoint accepts timeframe, days auto-adjusted
- `dashboard/templates/dashboard.html`: ML search dropdown for timeframe

### Key metrics:
- In-sample data: 16,688 bars (H1) / 3,616 bars (D1) / 60+ symbols/year
- Out-of-sample: 7,153 bars (H1) / 1,550 bars (D1)
- Search duration: 15-20 minutes
- Genomes evaluated: 1,000 per search (50 pop × 20 gen)
- OOS validation: 10 best genomes

### Configuration that affects results:
- `min_profit_factor` (default 1.2)
- `min_trades` (default 20, OOS requires 30+)
- `max_drawdown_pct` (default 25%)
- `min_win_rate` (default 30% - filters lottery tickets)
- `population_size` (default 50)
- `generations` (default 20)

---

## The Critical Question: Should we keep trying ML?

**YES** - but with different conditions:
- More symbols (10+)
- Multi-timeframe approach
- Walk-forward validation (not just OOS)
- Longer OOS period (6+ months, not 30% of 1 year)

**NO** if:
- After 5 more attempts with 10+ symbols, still no validated strategies
- The rule-based strategies (Scalp, Trend) also show no edge in forward testing
- Then the honest answer is: simple technical analysis doesn't work in 2026 markets

---

## Next Steps (in order of priority)

1. **Add more symbols** to expand the search space
2. **Run ML Search on H4** (2 years data) - good middle ground
3. **Use rule-based Trend** with 50/200 EMA on H1 as the live strategy
4. **Track real performance** on demo for 4-6 weeks
5. **If no profit after 4-6 weeks**: try multi-timeframe (D1+H1 combined)
6. **If still struggling**: research/order-block strategies or use as a signal source only (not full auto-trade)

The bot is a tool. It works. The challenge is the market.

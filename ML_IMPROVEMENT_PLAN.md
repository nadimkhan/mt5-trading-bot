# ML Strategy Improvement Plan (post-critique)

**Date:** 2026-10-08
**Status:** Planning phase - no code changes yet
**Trigger:** External code review identified 9 methodology issues

---

## Context

We received a thorough critique of the ML strategy implementation. The reviewer identified that our system was **measuring the wrong things** more than the market being impossible. Before any code changes, we need to fix the validation framework.

## The 9 Issues Identified

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Too many parameters (30+) for too little data (1 year, 4 symbols) | CRITICAL | Not fixed |
| 2 | OOS data leaked through repeated tweaking | HIGH | Needs holdout |
| 3 | Acceptance criteria never tested against noise | CRITICAL | Need null test |
| 4 | PF on small samples is noisy; "lottery ticket" patch treats symptom | MEDIUM | Fitness needs redesign |
| 5 | All indicators (EMA, MACD, BB, ICH, HA) are smoothed price variants | HIGH | Less correlation, not more |
| 6 | Cost math wrong (spread only on losses), no look-ahead protection | HIGH | Backtester audit needed |
| 7 | Backtested strategy ≠ live strategy (no 14 filters in backtest) | HIGH | Major mismatch |
| 8 | GA converges prematurely (50 pop, 20 gen, 2 elites) | MEDIUM | Need diversity pressure |
| 9 | Proposed path (more symbols, 50/200 EMA, 3-month demo) is weak | MEDIUM | Plan needs rework |

---

## The 6 Improvement Options (in priority order)

### Option A: Null Test (FOUNDATION - DO FIRST) ⭐

**Goal:** Calibrate acceptance criteria against random data

**What it does:**
- Run the exact same GA on shuffled or random-walk price data
- See if the criteria (PF > 1.2, 30+ trades, 30%+ WR) still pass
- If yes: the criteria are broken, not the market
- If no: the criteria can detect real edges

**Time:** 30 minutes
**Code changes:** None (just testing)

**How to do it:**
1. Create `tests/null_test.py`:
   ```python
   # Generate random walk: price[i] = price[i-1] + random.normal(0, 0.001)
   # Or shuffle real bars (destroy time order)
   # Run ML search on this
   # Check: how many "validated" genomes?
   ```
2. Expected: zero (criteria should reject random)
3. If not zero: criteria need adjustment

**Success criteria:**
- 0 genomes pass on random data
- 1+ genomes pass on real data (after fix)

---

### Option B: Simplify the Genome (SIMPLIFY)

**Goal:** Reduce parameters to 5-8, focus on one entry logic

**Current genome (30+ params):**
```
EMA periods: 2
RSI: 3
Bollinger: 2
MACD: 3
Ichimoku: 3
Heikin Ashi: 2
Volume: 2
Stops: 2
Filters: 3
Strategy selection: 5 booleans
TOTAL: ~30
```

**Proposed genome (5-8 params):**
```
ema_fast: 5-25
ema_slow: 20-100
rsi_period: [7, 14, 21]
atr_sl_multiplier: 1-3
atr_tp_multiplier: 2-6
min_adx: 15-30
[optional] use_macd_confirm: True/False
[optional] volume_filter: True/False
TOTAL: 5-8
```

**Time:** 2 hours
**Code changes:**
- `strategies/ml_searcher.py`: remove 20+ parameters
- `strategies/ml_strategy.py`: same
- `db schema`: migrate or drop unused columns
- `dashboard`: remove param controls (or keep for power users)

**Trade-off:** Less expressive, but more statistically reliable

---

### Option C: Walk-Forward Validation (ROBUST)

**Goal:** Replace 70/30 split with rolling 3-year train + 1-year test

**Current:** 70% in-sample, 30% OOS (last 3.5 months of 1 year)
**Proposed:**
```
Step 1: Train 2019-2022, Test 2023
Step 2: Train 2020-2023, Test 2024
Step 3: Train 2021-2024, Test 2025
Step 4: Train 2022-2025, Test 2026 (live)
Each step is a full GA run.
```

**Time:** 2 days
**Code changes:**
- Need to fetch 10+ years of data (only have 1)
- Major refactor of `run_search` to do rolling steps
- Storage for historical data (or fetch on each step)

**Trade-off:** Much more honest, but 10x slower

**Data requirement:** 10+ years of D1/H4 data for 10-15 symbols = 100K+ bars. Need to either:
- Fetch in advance (storage)
- Fetch on demand (slow)

---

### Option D: Fix Backtester Cost Modeling (CRITICAL FIX)

**Goal:** Make backtest honest about real trading costs

**Current issues:**
1. Spread subtracted only from losses (should be on every trade)
2. No slippage modeled
3. May enter on signal bar's close (look-ahead bias)
4. No swap (overnight fees)
5. Same cost for all symbols (XAUUSD has 3x EURUSD spread)

**Fixes needed:**

```python
# Every trade:
entry_price = next_bar_open  # not current close (no look-ahead)
spread_cost = bid_ask_spread[symbol]  # per-symbol
slippage = 0.5 * pip_size  # 0.5 pip average
commission = per_lot_fee

if direction == 'long':
    effective_entry = next_bar_open + spread_cost/2 + slippage
    effective_exit = exit_price - spread_cost/2 - slippage
else:  # short
    effective_entry = next_bar_open - spread_cost/2 - slippage
    effective_exit = exit_price + spread_cost/2 + slippage
```

**Per-symbol cost table:**
```python
COSTS = {
    'EURUSD': {'spread_pips': 0.5, 'swap_long': -0.5, 'swap_short': 0.3},
    'XAUUSD': {'spread_pips': 3.0, 'swap_long': -2.0, 'swap_short': 0.5},
    'BRNUSD': {'spread_pips': 4.0, 'swap_long': -3.0, 'swap_short': 0.0},
    'GBPUSD': {'spread_pips': 0.8, 'swap_long': -0.3, 'swap_short': 0.2},
}
```

**Time:** 4 hours
**Code changes:**
- `backtester/backtester.py`: add cost model, fix entry timing
- `strategies/ml_searcher.py`: use new cost values per symbol

**Trade-off:** Slower backtest (need to fetch spread per symbol), but realistic

---

### Option E: Include 14 Filters in Backtest (MATCH)

**Goal:** Backtested strategy = live strategy (apples to apples)

**Current state:** Backtested strategy ignores the 14 safety filters that live trading uses.

**Required additions:**
1. News filter (fetch economic calendar)
2. AI confirmation (mock Groq for backtest)
3. Streak-based lot sizing (martingale removal)
4. Daily trade limits
5. Volatility filter

**Time:** 1-2 days
**Code changes:**
- Create `backtester/live_simulator.py` that mimics live engine
- Add news calendar fetcher (`akshare`, `forexfactory` API)
- Add mock AI (or skip - too expensive for backtest)
- Wire into ML search scoring

**Trade-off:** 5-10x slower backtest, but true comparison

---

### Option F: Skip ML, Use Rule-Based (PRAGMATIC)

**Goal:** Use proven simple strategy, trade demo, see if it makes money

**Current rule-based:**
- Scalp: H4 trend + M15 pullback + M1 confirm
- Trend: EMA crossover

**Why try this:**
- The 14 safety filters + simple rules might be enough
- Real-world data is more honest than any backtest
- 3-month demo run gives 30-60 trades (not enough for certainty, but real)

**Setup:**
1. Switch to "Trend" strategy
2. Use H1 timeframe, 50/200 EMA (basic)
3. Risk 1% per trade
4. Track every trade in `bot.db`
5. After 3 months, evaluate

**Time:** 5 min to configure
**Code changes:** None
**Benefit:** Real performance data, no overfitting concerns

---

## Recommended Path

### Phase 1: Calibrate the measurement (FOUNDATION)

**Step 1: Option A (Null Test)** - 30 min
- Confirms if our criteria can detect real edges at all

**Step 2: Option D (Cost Fix)** - 4 hours
- Makes backtest realistic
- Without this, all other results are fantasy

**Step 3: Re-test the original H1 setup** - 1 hour
- Run ML search with corrected costs
- See if any strategies now pass
- If yes, we have something
- If no, we know the cost fix didn't change the verdict (rules are sound)

### Phase 2: Simplify (if Phase 1 shows potential)

**Step 4: Option B (Simplify Genome)** - 2 hours
- 5-8 params, single entry type
- Less overfitting, more interpretable

**Step 5: Option A again** - 30 min
- Null test the simplified version
- Confirm criteria still work

### Phase 3: Robust validation (if Phase 2 finds something)

**Step 6: Option C (Walk-Forward)** - 2 days
- Only worth doing if there's a candidate
- Prevents future overfitting

### Phase 4: Live matching (if going to production)

**Step 7: Option E (Backtest = Live)** - 1-2 days
- Required before real money
- Ensures backtested performance matches live

### Parallel path (if ML keeps failing)

**Always: Option F (Rule-Based Demo)**
- 5 min to start
- Run while we work on ML
- Real data on simple approach

---

## Decision Log (to be filled as we go)

### Decision 1: Start with Option A
- **Date:** 2026-10-08
- **Choice:** Run null test first
- **Reason:** Cheapest test, biggest information gain

### Decision 2: (pending)
- **Choice:**
- **Reason:**

### Decision 3: (pending)
- **Choice:**
- **Reason:**

---

## Current Status

| Option | Status | Next step |
|---|---|---|
| A - Null Test | Not started | Discuss & start |
| B - Simplify | Not started | After A |
| C - Walk-Forward | Not started | If A+B show promise |
| D - Cost Fix | Not started | In parallel with A |
| E - Match Live | Not started | Before live trading |
| F - Rule-Based | Available | Can start anytime |

---

## Risks if we don't do this

If we keep adding more indicators and more parameters without fixing the foundation:
- **Continue to overfit** - every "improvement" makes the OOS worse
- **Wasted time** - each fix introduces new overfitting opportunities
- **False confidence** - a "validated" genome might be useless
- **Real money loss** - if we ever go live with a fake strategy

The critique was right: **we were building a sophisticated overfitting machine**.

---

## What to discuss

1. **Start with Option A (null test) tomorrow?** - 30 min
2. **Should we also start Option F (rule-based demo) in parallel?** - 5 min
3. **If null test shows our criteria accept random data, what's the plan?**
4. **Are you OK with the rule-based approach for 3 months as a backup plan?**

I will not change any code until you tell me which option to start with.

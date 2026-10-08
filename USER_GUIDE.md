# MT5 Trading Bot - User Guide

**Version:** 2.0
**Last updated:** 2026-10-08
**Audience:** Users running the bot (not developers)

This guide explains **how the bot works** in plain language, **how data is stored**, and **how each strategy trades**.

---

## Quick Start

```bash
# 1. Start the bot
python main.py

# 2. Open the dashboard
# Browser: http://localhost:5000

# 3. Add assets (in sidebar)
# Click "+ Add Asset", select symbols (XAUUSD, EURUSD, etc.)

# 4. Configure strategy (optional)
# Click "Configure Strategies" in sidebar
# Adjust EMA, RSI, timeframes, risk limits

# 5. Start trading
# Click "Start" button in header
# Bot will analyze market every 15-60 seconds

# 6. Monitor
# Watch AI Decisions card for live signals
# Watch Open Positions for active trades
# Watch Recent Trades for closed P&L
```

---

## Understanding Timeframes

The bot uses **3 timeframes** for multi-timeframe analysis:

| Timeframe | Role | What it does |
|---|---|---|
| **Trend** (H4/H1/D1) | Higher TF | Determines overall direction (BUY or SELL bias) |
| **Entry** (M15/M5) | Mid TF | Generates the actual signal (crossover, etc) |
| **Confirm** (M5/M1) | Lower TF | Confirms the signal with price action |

**Example:** With `H4/M15/M5`:
- Bot checks H4 to know if XAUUSD is in an uptrend or downtrend
- Looks for crossover signal on M15
- Confirms with M5 candle before entering

### Where to change timeframes

In the dashboard:
1. Sidebar → "Configure Strategies"
2. Top section: "Timeframes & Timing"
3. Change Trend/Entry/Confirm dropdowns
4. Click "Save Changes"
5. **Takes effect immediately** (no restart needed)

### Recommended setups

| Style | Trend | Entry | Confirm | Best for |
|---|---|---|---|---|
| Ultra-scalp | M15 | M5 | M1 | Very fast trades, 1-5 min holds |
| Day trading | H1 | M15 | M5 | Intraday, 15-60 min holds |
| **Default (balanced)** | **H4** | **M15** | **M5** | **Most symbols, 1-4 hour holds** |
| Swing | D1 | H4 | H1 | Multi-day holds |

---

## How Data is Stored (Important!)

**Nothing is lost when you restart the bot.** Everything persists to disk:

### What's stored where

| Data Type | Storage Location | Survives Restart? |
|---|---|---|
| **Trade history** (all closed trades with P&L) | `bot.db` (SQLite database) | ✅ YES |
| **AI decisions log** (every signal the bot made) | `bot.db` | ✅ YES |
| **ML genomes** (discovered strategy parameters) | `bot.db` | ✅ YES |
| **Symbols list** (which pairs you're trading) | `bot.db` | ✅ YES |
| **Strategy parameters** (EMA periods, RSI, etc) | `strategy_configs.json` | ✅ YES |
| **Timeframe settings** (H4/M15/M5) | `strategy_configs.json` | ✅ YES |
| **Active strategy** (which strategy is active) | `strategy_configs.json` | ✅ YES |
| **Daily trade logs** (readable JSON) | `logs/trades_YYYYMMDD.json` | ✅ YES |

### What gets reset on restart

Only the engine's **in-memory state**:
- Currently open positions (re-fetched from MT5 on startup)
- Market data buffers (refetched)
- AI decisions in memory (still logged to DB)

### Backup recommendation

**Important:** Back up these files regularly:
- `bot.db` - your trade history and ML genomes
- `strategy_configs.json` - your tuned parameters
- `config.yaml` - your API keys (NEVER commit to git)

---

## The Three Strategies

You can choose between 3 strategies. Only ONE is active at a time.

### 1. Scalp Strategy (Default)

**What it does:** H4 trend + M15 pullback + M1 confirmation

**Entry logic:**
1. Check H4 trend (EMA 50 vs EMA 200)
2. Wait for M15 pullback to EMA 20
3. Confirm with M1 candle in trend direction

**Best for:** Frequent trades, 5-30 min holds

**Configuration:** ~15 parameters (EMA periods, RSI, ATR multipliers, etc)

### 2. Trend Following Strategy

**What it does:** EMA crossover on entry timeframe + confirm timeframe

**Entry logic:**
1. Wait for EMA fast crosses above slow (BUY) or below (SELL) on entry TF
2. Confirm with same signal on confirm TF

**Best for:** Longer holds, 1-4 hour positions

**Configuration:** ~15 parameters (EMA periods, RSI, etc)

### 3. ML (Genetic Algorithm) Strategy

**What it does:** Uses parameters **discovered by running 1000+ strategy variations** on historical MT5 data

**Entry logic:**
1. Loads the best "genome" (set of parameters) from `ml_genomes` table
2. Uses those exact EMA periods, RSI thresholds, ATR multipliers
3. Looks for EMA crossover on entry timeframe

**Best for:** Data-driven, rules optimized for YOUR specific symbols

**Configuration:** NO manual parameters - it's all discovered

---

## How to use the ML Strategy (Step by Step)

The ML strategy needs "genomes" (parameter sets) to work. Here's the complete flow:

### Step 1: Start the bot
```bash
python main.py
```
Bot connects to MT5. Must be running for ML search.

### Step 2: Run ML Search
1. In sidebar, click **"Run ML Search"**
2. Confirm the prompt (searches take 2-5 min)
3. Bot downloads 1 year of H1 data for ALL enabled symbols
4. Tests 1000+ strategy variations using genetic algorithm
5. Saves best ones to `ml_genomes` table

**Note:** Run during market hours for best data.

### Step 3: Switch to ML strategy
1. In sidebar, click **"ML"** strategy button
2. Bot will use the best genome for trade decisions
3. Your selection is **persisted** - survives restart

### Step 4: Watch for trades
- AI Decisions card shows ML signals (BUY/SELL/HOLD)
- Open Positions card shows active ML trades
- Recent Trades card shows closed P&L

### How the genetic algorithm works

```
1. Start with 50 random genomes (parameter sets)
   Example: {ema_fast: 9, ema_slow: 21, rsi_period: 14, ...}

2. Backtest each on real MT5 data
   - Calculate profit factor (gross profit / gross loss)
   - Count trades, win rate, max drawdown

3. Keep top 5 (elitism)

4. Create 45 new genomes via:
   - Crossover: combine params from 2 parents
   - Mutation: randomly adjust some params

5. Repeat for 20 generations
   - 20 * 50 = 1000 evaluations
   - Best genomes emerge organically

6. Validate top 10 on out-of-sample data
   - Final score must be PF > 1.3, trades > 50, DD < 25%

7. Save to ml_genomes table for live trading
```

---

## The 14 Safety Filters (Why Trades Don't Happen)

Before any trade executes, it passes through 14 filters. If ANY fails, the trade is skipped:

| # | Filter | What it does | Where to configure |
|---|---|---|---|
| 1 | **Signal check** | Strategy must return BUY/SELL (not HOLD) | N/A - strategy logic |
| 2 | **Confidence** | Strategy confidence >= min_confidence | Configure → EMA Periods |
| 3 | **Momentum** | Price moving in signal direction (not flat) | N/A - automatic |
| 4 | **Volume** | Current volume >= 20-period average | N/A - automatic |
| 5 | **News filter** | No high-impact news in next 30 min | N/A - automatic |
| 6 | **Risk-Reward** | TP distance >= SL distance * min_RR | Configure → Risk |
| 7 | **AI confirmation** | AI (Groq/Claude) must agree with system | Configure → AI Required |
| 8 | **Already in position** | One position per symbol max | N/A - automatic |
| 9 | **Daily trade limit** | Trades today < max_trades_per_day | Configure → Daily Loss |
| 10 | **Daily loss limit** | Daily P&L > -daily_loss_limit_pct% | Configure → Daily Loss |
| 11 | **Open position limit** | Open positions < max_open_trades | Configure → Max Open |
| 12 | **Loss streak** | Lot reduced after consecutive losses | Configure → Streak |
| 13 | **Win streak** | Lot increased after consecutive wins | Configure → Streak |
| 14 | **Session filter** | Within trading hours | N/A - automatic |

### How to debug "why no trades"

Run this in your terminal:
```bash
curl http://localhost:5000/api/diagnostics
```

This shows:
- Last scalp loop time (proves bot is running)
- Rejection count by reason
- Recent rejections (last 10)
- MT5 connection status
- Active strategy

Example output:
```json
{
  "engine_status": "RUNNING",
  "mt5_connected": true,
  "active_strategy": "ml",
  "seconds_since_last_loop": 12,
  "rejection_count_by_reason": {
    "ai_hold_low_conf": 145,    ← most common rejection
    "low_confidence": 12,
    "no_signal": 8
  },
  "recent_rejections": [
    {"symbol": "EURUSD", "reason": "ai_hold_low_conf", "details": "conf=30%"},
    ...
  ]
}
```

---

## How to Read the Dashboard

### Top KPI Cards (4 stats)
- **Total P&L** - Sum of all closed trades (green = profit, red = loss)
- **Win Rate** - % of trades that were profitable, with streak indicator
- **Open Positions** - Currently active trades
- **Total Trades** - All-time trade count

### 2x2 Cards
- **Open Positions** - Live trades with current P&L
- **Market Analysis** - Current bid/ask and trend for each symbol
- **AI Decisions** - Last 8 decisions with confidence, veto reasons
- **Recent Trades** - Last 5 closed trades with entry/exit/P&L

### Status Badges
- Green dot = engine RUNNING
- Red dot = engine ERROR
- Grey dot = STOPPED
- Yellow dot = INITIALIZING

### AI Decision Card Badges
- **BUY OK** / **SELL OK** (green, pulsing) - AI confirmed the trade
- **VETO** (red) - AI rejected
- **MISMATCH** (amber) - AI disagreed
- **REJECT** (amber) - Filter rejected (low conf, bad RR, etc)
- **AI SKIP** (amber) - AI returned HOLD

---

## Common Tasks

### Add a new symbol to trade
1. Sidebar → "+ Add Asset"
2. Dropdown opens, pick a symbol
3. Click "Add" button next to it
4. Symbol moves to "Active" box
5. **No restart needed** - engine picks it up on next loop

### Remove a symbol
1. Click the **×** button next to the symbol in "Active" box
2. Symbol gets disabled (not deleted, history preserved)

### Change strategy
1. Sidebar → Click any strategy button (Scalp/Trend/ML)
2. Selection is **persisted** - survives restart
3. Engine switches immediately on next loop

### Adjust a parameter
1. Sidebar → "Configure Strategies"
2. Drag the slider or type exact value
3. Click "Save Changes"
4. Engine reloads params on next loop (~15 sec)

### Run ML Search
1. Sidebar → "Run ML Search" button
2. Takes 2-5 minutes
3. Tests 1000+ strategies on real MT5 data
4. Saves best ones
5. Switch to ML strategy to use them

### Check why no trades
```bash
curl http://localhost:5000/api/diagnostics | python -m json.tool
```

---

## Troubleshooting

### "No open positions" but I'm watching it
- Check `ai_decisions` card - is the strategy returning BUY/SELL?
- Most common: **AI is rejecting** (low confidence) → try setting `ai_required: false`
- Or: **Confidence too low** → lower `min_confidence` in config

### "Invalid stops" error
- This was a bug, now fixed. Make sure you restarted after the fix.

### "Active assets" shows 0
- Database had legacy disabled symbols
- Fixed in migration - restart to apply

### "ML strategy not finding trades"
- The ML strategy only trades if a genome is loaded
- Click "Run ML Search" first to populate genomes
- Check `curl http://localhost:5000/api/ml/genomes` to see if any exist

### Bot won't start
- Check terminal output for the exact error
- Most common: missing MT5 credentials in `config.yaml`
- Make sure MT5 terminal is running and logged in

### Dashboard won't load
- Check `curl http://localhost:5000/api/status`
- If you get JSON, backend is working - hard refresh browser
- If you get nothing, restart the bot

---

## Best Practices

1. **Run ML Search once per week** - markets change, parameters may need updating
2. **Start with AI disabled** (`ai_required: false`) to see raw strategy signals
3. **Use demo account** until you're confident in the bot
4. **Start with 1-2 symbols** - don't trade 10 pairs at once initially
5. **Check the AI Decisions card** - understand WHY trades are taken or rejected
6. **Use the diagnostics endpoint** to debug issues: `curl http://localhost:5000/api/diagnostics`
7. **Backup `bot.db` regularly** - it has all your trade history

---

## File Locations Reference

```
E:/projects/mt5-trading-bot/
├── main.py                          ← Start the bot here
├── config.yaml                      ← Your API keys, MT5 credentials
├── bot.db                           ← SQLite database (all trade history)
├── strategy_configs.json            ← Your saved parameters + active strategy
├── PROJECT_CHECKLIST.md             ← Project status, what to do next
├── USER_GUIDE.md                    ← This file
├── engine/
│   ├── trading_engine.py            ← Main bot logic
│   ├── mt5_connector.py             ← MT5 communication
│   ├── trade_manager.py             ← Position management (SL, TP, breakeven)
│   └── trade_counter.py             ← Daily stats, streaks
├── strategies/
│   ├── rule_based.py                ← Scalp + Trend strategies
│   ├── ml_strategy.py               ← ML strategy (uses discovered genome)
│   ├── ml_searcher.py               ← Genetic algorithm
│   └── strategy_config.py           ← Config save/load
├── ai/
│   ├── ai_filter.py                 ← AI confirmation gate
│   └── ai_analyzer.py               ← Old AI analyzer (deprecated)
├── dashboard/
│   ├── app.py                       ← Flask backend
│   └── templates/dashboard.html     ← Frontend
└── logs/
    └── trades_YYYYMMDD.json         ← Daily trade logs
```

---

## Glossary

- **EMA** - Exponential Moving Average, smooths price data
- **RSI** - Relative Strength Index, momentum indicator (0-100)
- **ATR** - Average True Range, measures volatility
- **ADX** - Average Directional Index, measures trend strength
- **PF (Profit Factor)** - Gross profit / gross loss (>1 = profitable)
- **Win Rate** - % of trades that closed in profit
- **RR (Risk:Reward)** - TP distance / SL distance (higher = better)
- **Genome** - A complete set of strategy parameters (like DNA)
- **Walk-forward** - Testing on data the algorithm hasn't seen
- **In-sample vs Out-of-sample** - Training data vs test data
- **Genetic Algorithm** - Optimization method that mimics natural selection
- **Stop level** - MT5's minimum distance for SL/TP from current price
- **IOC** - Immediate Or Cancel order filling mode
- **Magic number** - Unique ID to identify our bot's orders

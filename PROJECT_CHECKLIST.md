# MT5 Trading Bot - Project Checklist

**Last updated:** 2026-10-08
**Repository:** https://github.com/nadimkhan/mt5-trading-bot
**Account:** 57452518479 (Demo $100k)
**Current Status:** UI polish phase COMPLETE. Ready to resume backend features.

---

## COMPLETED

### Backend Features
- [x] AI confirmation gate (AI must agree with system before trade)
- [x] AI veto/mismatch/skip all properly logged
- [x] Configurable timeframes (H4/M15/M1 etc.) via dropdown
- [x] Configurable timing intervals (scalp/trend interval)
- [x] Configurable strategy parameters (EMA, RSI, ATR, etc.)
- [x] Win/loss streak-based lot size adjustment
- [x] Max trades per day limit
- [x] Daily loss limit
- [x] Spread filter
- [x] News filter (NFP, FOMC, CPI detection)
- [x] Momentum filter
- [x] Volume filter
- [x] AI veto reasons shown in dashboard
- [x] MT5 historical data download (fixed - passes datetime + constants)
- [x] One trade per symbol enforcement
- [x] ML/Genetic strategy searcher
- [x] Walk-forward validation (in-sample / out-of-sample)
- [x] ML search uses real MT5 data across all enabled symbols
- [x] DB_PATH centralized in engine/constants.py
- [x] Bare except clauses replaced with specific exceptions
- [x] Order result validation (retcode check)
- [x] Lot size safety check (no zero/negative)
- [x] Live WebSocket updates (positions, decisions, analytics, market, history, status)
- [x] Hot-reload of dashboard HTML/CSS (no restart needed)
- [x] Hot-reload of strategy configs in trading loop
- [x] `/api/strategies/reload` endpoint for force-reload
- [x] Strategy-aware context buttons (ML Search only when ML selected)
- [x] Asset picker dropdown with categorized list
- [x] Add/remove assets via UI with DB persistence

### Bug Fixes
- [x] `'Tick' object has no attribute 'spread'` - fixed using `symbol_info`
- [x] `datetime is not JSON serializable` - all datetimes converted to ISO strings
- [x] `detect_market_regime() missing 2 required positional arguments` - fixed
- [x] `TradeDeal object has no attribute 'price_open'` - use `price` instead
- [x] `copy_rates_range` failed - fixed to pass datetime + timeframe constant
- [x] Duplicate news check removed
- [x] AI auto-approve fallback fixed (now returns HOLD on error)
- [x] ML button not showing as active (missing in updateStrategyUI)
- [x] Sidebar default symbols were disabled - migration enabled them
- [x] Double padding in sidebar (container + element)
- [x] Regime strategy missing `win_streak_increase_pct` and `loss_streak_threshold` attributes
- [x] Engine reading `stop_loss`/`take_profit` but regime returns `sl`/`tp` (field name mismatch)
- [x] `UnboundLocalError: cannot access local variable 'ai_action'` - initialized before block
- [x] `trades` table missing `spread`/`regime`/`slippage` columns - added via ALTER
- [x] `trades` table missing `spread_at_exit`/`exit_slippage` columns - added
- [x] `close_position` failed with "Unsupported filling mode" - now probes `symbol_info.filling_mode` bitmask
- [x] `send_order` failed with "No prices (retcode=10021)" on non-tradable symbols - now validates `trade_mode=4` and bid/ask
- [x] Daily trade limit counted ALL symbols - now per-symbol via `get_trades_today_for_symbol()`
- [x] Orphan closer assigning same close deal to multiple trades - now unique per orphan
- [x] Trade history showed OPEN positions - filtered to `status='CLOSED' AND pnl IS NOT NULL`
- [x] Trade history limited to 5 trades - now shows all in scrollable container

### Recent Trade History & UI
- [x] Close button per open position (calls `/api/positions/close`)
- [x] Scrollable trade history with themed scrollbar (max-height 320px)
- [x] P&L display robust to null/undefined, supports both `lot_size` and `volume` field names
- [x] Engine broadcasts 100 closed trades instead of 20

### Trade Management Features
- [x] Regime-change exit: closes positions when market regime flips (BULL<->BEAR)
- [x] Hidden SL/TP option: send orders with `sl=0, tp=0` and manage in-memory
- [x] Internal SL/TP monitor: when hidden mode on, monitors price and closes at SL/TP
- [x] Trailing stop, breakeven, partial TP via trade_manager.manage_all_positions()
- [x] Per-symbol daily trade limit (10 trades/day per symbol, was global)
- [x] Orbital trade cleanup: orphan closer runs every loop, assigns unique close deals
- [x] Config option `trade_management.hidden_sl_tp` (default false) and `regime_exit` (default true)

### Online Parameter Optimizer (NEW - this session)
- [x] `strategies/optimizer.py` with brute-force parameter search
- [x] Per-strategy parameter ranges (Scalp/Trend/Regime)
- [x] Quick backtester on M5 data (6h window)
- [x] Scoring: PnL × win rate × trade count × drawdown penalty
- [x] Smoke test in `tests/optimizer_smoke_test.py`
- [ ] UI button to run optimizer
- [ ] Auto-trigger on regime change
- [ ] Auto-trigger every 6h
- [ ] Apply best parameters to live strategy

### UI Polish (DONE in this session)
- [x] **Loading states** - Spinner + "Loading..." text on initial load
- [x] **Error states** - `safeFetch` wrapper with friendly error messages
- [x] **Empty states** - Icon + message + hint for each section
- [x] **Typography** - Poppins for headings, Inter 11px body, JetBrains Mono for numbers
- [x] **Logo** - "MT5TB" gradient text (no box), Poppins font
- [x] **Background** - Darker (#060912), subtler borders (#1a2335)
- [x] **Border radius** - All elements max 6px (no more 16px)
- [x] **Sidebar** - 10px container padding (no double padding inside)
- [x] **Symbol font** - Unified across all sections (JetBrains Mono 400)
- [x] **Section separators** - Subtle border-bottom between nav sections
- [x] **Strategy buttons** - Full-width stacked rows, name left + details right
- [x] **Start/Stop** - Moved to header with smart enabled/disabled states
- [x] **Hover effects** - Cards lift on hover
- [x] **Version indicator** - "v2.0" badge in title for cache debugging
- [x] **Console log** - Logs version on load to help debug caching
- [x] **Asset picker** - Categorized dropdown (Forex/Commodities/Crypto/Indices)
- [x] **Active assets box** - Shows enabled symbols with remove button
- [x] **Smart context buttons** - Run ML Search only enabled when ML selected
- [x] **ML button active state** - Properly toggles with active class
- [x] **Status indicators** - Color-coded dots (running/error/ready)
- [x] **Streak counter** - Live win/loss streak with badge
- [x] **Format consistency** - formatMoney(), formatPct() helpers

---

## REMAINING UI WORK (LOW PRIORITY)

### Layout & Display
- [ ] **Mobile responsive design** - Layout breaks on smaller screens
- [ ] **Modal close (X) button** - Currently only click-outside closes modals
- [ ] **Trade history pagination** - Only shows 5 latest trades, need full scrollable list
- [ ] **Config modal too long** - Need tabs or accordion (Timeframes, Parameters, Risk)
- [ ] **Modal better scroll handling** - Long content overflows

### User Experience
- [ ] **No onboarding** - First-time users don't know what to do (welcome modal?)
- [ ] **No tooltips** - Hover explanations for technical terms (EMA, RSI, ATR)
- [ ] **No keyboard shortcuts** - Power users would love keyboard nav
- [ ] **Settings page** - Currently only strategy config, need app settings (account, AI, MT5)
- [ ] **Theme toggle** - Dark/light mode (currently only dark)
- [ ] **Sound alerts** - For key events (trade open, kill switch)
- [ ] **Time display** - 24h/12h toggle

### Visualization (Charts)
- [ ] **Equity curve chart** - Performance over time
- [ ] **Drawdown chart** - Visualize max drawdown
- [ ] **Win rate by hour/day** - When does strategy work best?
- [ ] **Per-symbol breakdown** - Which symbols are profitable?

---

## REMAINING BACKEND FEATURES (Future)

### Debugging & Diagnostics (URGENT - investigate why bot isn't trading)
- [ ] **Investigate: why trades are not happening** - Add diagnostic endpoint showing all filter rejections, current state of each symbol, why no signals pass
- [ ] **Investigate: why AI decisions are not showing up** - Check if AI is firing, why decisions aren't being broadcast, debug the decision flow

### Risk Management
- [ ] Correlation exposure dashboard
- [ ] Risk per trade percentage display
- [ ] Margin usage monitor
- [ ] Sector/currency exposure breakdown (USD pairs, EUR pairs)

### Notifications
- [ ] Telegram bot integration for trade alerts
- [ ] Email notifications for daily summary
- [ ] Desktop notifications for browser
- [ ] Sound alerts for key events (kill switch, large P&L)

### Analytics
- [ ] Monthly returns heatmap
- [ ] Sharpe/Sortino ratio calculation
- [ ] Strategy comparison (which strategy performs best?)
- [ ] Slippage tracking
- [ ] Commission tracking

### Multi-Account
- [ ] Support multiple MT5 accounts
- [ ] Account switcher in dashboard
- [ ] Aggregate P&L across accounts

### Advanced Strategies
- [ ] Strategy marketplace (share/import genomes)
- [ ] Strategy A/B testing (run two strategies simultaneously)
- [ ] Auto-retrain weekly/monthly
- [ ] Multi-strategy portfolio (combine ML + Scalp + Trend)

### Operations
- [ ] Docker container
- [ ] Systemd service for auto-restart
- [ ] Log rotation
- [ ] Health check endpoint
- [ ] Metrics export (Prometheus format)
- [ ] Backup/restore for database

### Testing
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] Load testing for WebSocket
- [ ] Stress test for ML searcher

### Documentation
- [ ] User guide
- [ ] API documentation
- [ ] Architecture diagram
- [ ] Configuration reference
- [ ] Troubleshooting guide

### Advanced ML
- [ ] More genome features (volume profile, market structure)
- [ ] Multi-objective optimization (PF + DD + Sharpe)
- [ ] Real-time genome re-evaluation
- [ ] Walk-forward with rolling window

---

## KNOWN ISSUES / GOTCHAS

### Code Smells
- Some emoji in print() statements (works on file, fails on Windows console cp1252)
- Multiple sys.path.insert(0, ...) calls to handle imports
- Some long functions (>100 lines) in trading_engine.py

### Configuration
- config.yaml is in .gitignore (good - contains API keys)
- Need to add validation for all config values
- Need migration path for breaking config changes

### Performance
- ML search is synchronous (blocks dashboard)
- No caching of MT5 calls
- DB queries not optimized

---

## DEPLOYMENT NOTES

### Requirements
- Python 3.x
- MetaTrader 5 terminal installed and logged in
- Required packages: flask, flask-socketio, MetaTrader5, requests, pyyaml, pandas, numpy
- Optional: pytz (for timezone handling)

### First-time Setup
1. Clone repo
2. `pip install -r requirements.txt`
3. Copy `config.example.yaml` to `config.yaml`
4. Edit `config.yaml` with MT5 credentials and API keys
5. Run `python main.py`
6. Open `http://localhost:5000`

### Common Issues
- MT5 not connecting - check terminal is logged in
- AI not working - check API key in config.yaml
- ML search finds nothing - run during market hours for more data
- Dashboard not updating - hard refresh browser (Ctrl+Shift+R)
- Old UI showing - check console for "v2.0" version log

---

## PRIORITY FOR NEXT SESSION

The UI is now in a polished, professional state. Next priorities should be:

**High Priority (Backend features with user value):**
1. **Equity curve chart** - visualize performance over time
2. **Telegram notifications** - get alerts when away from computer
3. **Drawdown tracking** - see risk exposure in real-time
4. **Health check endpoint** - monitor bot status remotely
5. **Unit tests** - protect against regressions

**Medium Priority (Backend improvements):**
6. Walk-forward with rolling window - keep ML strategy fresh
7. Multi-strategy portfolio - run multiple strategies simultaneously
8. Performance comparison - which strategy works best?
9. Risk per trade display - show current risk exposure
10. Margin usage monitor - avoid over-leveraging

**Low Priority (Remaining UI polish):**
11. Mobile responsive design
12. Tooltips for technical terms
13. Onboarding flow for new users
14. Dark/light mode toggle
15. Sound alerts

---

## GIT HISTORY (Recent)

```
138100e Fix ML button: add to updateStrategyUI active toggle
9739bac Strategy selector: redesign as full-width stacked rows
c581fa9 Sidebar: add separator line between nav sections
d3ecad7 Unify symbol font: all symbol displays now use JetBrains Mono
b36554e Sidebar: remove double padding
3c54126 Sidebar: make added asset symbol normal weight
474cb4e Add v2.0 version indicator + console log
a979e63 Fix Active assets showing 0: enable default symbols
e1533f5 UI: Move Start/Stop to header with smart enabled/disabled states
84a5319 Sidebar: Add asset picker dropdown
7473aa1 Hot-reload: HTML/CSS changes appear on browser refresh
d1fb0f5 UI: Poppins for headings, Inter 11px body, MT5TB logo
3fbdd6e UI: Add loading spinners, friendly empty states
1076dad ML search now uses real MT5 historical data
5bb448f Add ML genetic algorithm strategy discovery
6c007c0 Track strategy decisions in self.ai_decisions
aed29cf AI as confirmation gate
3ff0f22 Add configurable timeframes
9864aba Fix bugs: centralize DB_PATH, bare excepts
b60bd0a Add momentum, volume, news filters
de255d7 Add win/loss streak handling
7c4857b Enforce confidence threshold, RR check
5979da2 Add configurable strategies
c8f078c Fix dashboard - live WebSocket
8f8ed1a Add strategy selection
```

---

## NOTES

- User is budget-constrained - prefer free solutions
- User prefers concrete step-by-step instructions
- User wants to see P&L in dashboard (working now)
- User wants AI confirmation before trades (working now)
- User wants one trade per symbol (working now)
- User wants ML/genetic algorithm option (working now)
- User wants UI improvements (UI POLISH PHASE COMPLETE)
- User wants to know what's next (this checklist)

---

## SESSION SUMMARY

**What's been accomplished in this UI session (~12 commits):**
- 25+ UI improvements
- 3 critical bug fixes (ML button, default symbols, double padding)
- Hot-reload capability (no restarts)
- Smart context-aware buttons
- Unified typography (Poppins/Inter/JetBrains Mono)
- Professional dark theme with subtle borders
- Asset picker with categories
- Full-width stacked strategy selector
- Section separators in sidebar
- Version indicators for debugging

**Time spent on UI**: ~4-5 hours
**Status**: UI is production-ready, time to move to backend features

# MT5 Trading Bot - Project Checklist

**Last updated:** 2026-10-08
**Repository:** https://github.com/nadimkhan/mt5-trading-bot
**Account:** 57452518479 (Demo $100k)
**Current Status:** Backend features mostly complete, UI needs polish

---

## COMPLETED (Last Session)

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

### Bug Fixes
- [x] `'Tick' object has no attribute 'spread'` - fixed using `symbol_info`
- [x] `datetime is not JSON serializable` - all datetimes converted to ISO strings
- [x] `detect_market_regime() missing 2 required positional arguments` - fixed
- [x] `TradeDeal object has no attribute 'price_open'` - use `price` instead
- [x] `copy_rates_range` failed - fixed to pass datetime + timeframe constant
- [x] Duplicate news check removed
- [x] AI auto-approve fallback fixed (now returns HOLD on error)

---

## IN PROGRESS / NEEDS FIXING

### UI Issues to Address (CURRENT FOCUS)
- [ ] **Layout polish** - Cards could be more uniform, spacing inconsistent
- [ ] **Color consistency** - Some status colors don't match across cards
- [ ] **Mobile responsiveness** - Layout breaks on smaller screens
- [ ] **Loading states** - No visual feedback during data loads
- [ ] **Error states** - Generic error messages, not user-friendly
- [ ] **Modal design** - Settings modal could be more intuitive
- [ ] **Iconography** - Need consistent icons (text labels work but icons would be better)
- [ ] **Typography** - Some areas use inconsistent font weights
- [ ] **Dark/light mode** - Only dark mode currently
- [ ] **Empty states** - "No data" messages could be more helpful
- [ ] **Trade history pagination** - Only shows 5 latest trades
- [ ] **Config modal too long** - Need tabs or accordion
- [ ] **No onboarding** - First-time users don't know what to do
- [ ] **No tooltips** - Hover explanations for technical terms

### Specific UI Bugs
- [ ] "Configure Strategies" button text alignment in sidebar
- [ ] Stat cards could have better number formatting
- [ ] Market Analysis card shows "Loading..." on initial load
- [ ] AI Decisions card doesn't show newest first
- [ ] Time display sometimes shows 24h, sometimes 12h
- [ ] Modal close button (X) missing - only click outside
- [ ] Strategy buttons in modal don't show currently selected
- [ ] Run ML Search button doesn't show progress percentage

---

## REMAINING BACKEND FEATURES (Future)

### Risk Management
- [ ] Correlation exposure dashboard
- [ ] Drawdown tracking chart
- [ ] Equity curve visualization
- [ ] Risk per trade percentage display
- [ ] Margin usage monitor

### Notifications
- [ ] Telegram bot integration for trade alerts
- [ ] Email notifications for daily summary
- [ ] Desktop notifications for browser
- [ ] Sound alerts for key events (kill switch, large P&L)

### Analytics
- [ ] Performance charts (equity, drawdown, win rate over time)
- [ ] Per-symbol statistics breakdown
- [ ] Time-of-day performance analysis
- [ ] Trade duration analysis
- [ ] Win streak visualization
- [ ] Sharpe/Sortino ratio calculation
- [ ] Monthly returns heatmap

### Multi-Account
- [ ] Support multiple MT5 accounts
- [ ] Account switcher in dashboard
- [ ] Aggregate P&L across accounts

### Advanced Strategies
- [ ] Strategy marketplace (share/import genomes)
- [ ] Strategy A/B testing (run two strategies simultaneously)
- [ ] Walk-forward optimization (re-run search periodically)
- [ ] Multi-strategy portfolio (combine ML + Scalp + Trend)
- [ ] News-based strategy override

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
- [ ] Auto-retrain weekly/monthly

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
- Dashboard not updating - check WebSocket connection in browser console

---

## PRIORITY FOR NEXT SESSION

**High Priority (UI Polish):**
1. Fix layout consistency across cards
2. Add loading states and progress indicators
3. Improve error messages
4. Add tooltips for technical terms
5. Better empty states

**Medium Priority:**
6. Mobile responsive design
7. Pagination for trade history
8. Config modal tabs/accordion
9. Visual feedback for ML search progress
10. Recent activity log

**Low Priority (Future):**
11. Light mode
12. Charts/graphs
13. Telegram notifications
14. Docker deployment
15. Unit tests

---

## GIT HISTORY (Recent)

```
66839cd Fix MT5 historical data download - pass datetime objects
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
- User wants UI improvements (STARTING NEXT)

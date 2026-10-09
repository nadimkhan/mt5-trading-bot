# Investigation Report: 3 User-Reported Issues

## Issue #1: Max trades per day not updating (logs show 10/10 after changing to 20)

**Root Cause:** Strategy attribute is cached at initialization, not hot-reloaded.

- `RegimeAwareStrategy.__init__` (line 88 of `strategies/regime_aware.py`):
  ```python
  self.max_trades_per_day = self.config.get("max_trades_per_day", 10)
  ```
  This runs once when the strategy is created.

- `trading_engine.py` line 1141:
  ```python
  max_trades = getattr(strategy_obj, 'max_trades_per_day', 10)
  ```
  Reads from the strategy object, not from the live config file.

- When user changes `strategy_configs.json` (file shows `"max_trades_per_day": {"value": 20, ...}`), the file is updated but the strategy object still holds the old value of 10.

**Evidence:** `grep "max_trades_per_day"` on `strategy_configs.json` shows `"value": 20` (the user's intended value), but `regime_aware.py` defaults to 10 and was instantiated at startup with whatever value was in config at that time.

**Fix:** Either (a) re-instantiate strategy on config reload, or (b) read the live value from the config each loop.

---

## Issue #2: P&L per trade values wrong in Recent Trades (huge difference)

**Root Cause:** `/api/history` endpoint uses MT5 history deals and matches incorrectly.

- `dashboard/app.py` lines 325-366: `api_history()` fetches all deals from MT5 for last 7 days
- Splits into `in_deals` (entry=0, opens) and `out_deals` (entry=1, closes)
- For each close deal, finds the **first matching IN deal for that symbol**:
  ```python
  for d in in_deals:
      if d.get('symbol') == symbol:
          in_deal = d
          break  # Takes the FIRST one, not the matching one
  ```
- This means if EURUSD has 10 open/close cycles in 7 days, ALL 10 close deals get matched against the same first entry deal.

**Symptoms:**
- All EURUSD entries show the same entry price (the very first open)
- The P&L values are correct (from each close), but the entry price is wrong
- The "huge difference" is because the entry doesn't match the actual trade

**Fix:** Match IN deal to OUT deal by sequence (each close should match the most recent open of the same symbol+type, not the first one ever).

---

## Issue #3: SL/TP visible in MT5 terminal despite hidden_sl_tp config

**Root Cause:** The `hidden_sl_tp` config is not enabled.

- `config.yaml` (default): `trade_management.hidden_sl_tp: false`
- The user has to manually set this to `true` in their local config
- If they didn't change it, all new orders will have visible SL/TP

**Other possible causes:**
- Old positions opened before the feature was added (but those are separate)
- The flag is passed correctly in `send_order` (verified in code)
- `sl=0, tp=0` is the correct way to hide them (verified in MQL5 docs)

**Fix:** User needs to set `hidden_sl_tp: true` in `config.yaml` and restart the bot. The existing code will then send `sl=0, tp=0` to MT5 and the order will appear without SL/TP levels.

---

## Plan to Fix (in priority order)

1. **#1 - Max trades per day**: Read from live config each loop instead of cached attribute
2. **#2 - P&L mismatch**: Match IN/OUT deals correctly by position sequence
3. **#3 - Hidden SL/TP**: Just need to enable config (already implemented, just needs config change)

After fixing all 3, continue with optimizer completion (UI button, auto-trigger, apply best params).

"""
Trading Engine - Multi-timeframe scalping with AI decision making
"""
import logging
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path

from engine.mt5_connector import MT5Connector
from engine.indicators import analyze_market, detect_market_regime, is_tradeable_regime, is_volume_confirmed, is_momentum_strong
from engine.trade_manager import TradeManager
from engine.session_filter import SessionFilter
from engine.correlation_controller import CorrelationController
from engine.safeguards import Safeguards
from strategies.rule_based import ScalpStrategy, TrendFollowingStrategy, StrategyManager
from ai.ai_filter import AIFilter, NewsChecker
from ai.ai_analyzer import AIAnalyzer

logger = logging.getLogger(__name__)

# Database path
DB_PATH = "E:/projects/mt5-trading-bot/bot.db"
CONFIG_PATH = "E:/projects/mt5-trading-bot/config.yaml"


def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_enabled_symbols():
    """Get list of enabled trading symbols from DB"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM symbols WHERE enabled = 1 ORDER BY symbol")
        symbols = [row['symbol'] for row in cursor.fetchall()]
        conn.close()
        if symbols:
            return symbols
    except Exception as e:
        logger.warning(f"Failed to get symbols from DB: {e}")
    # Fallback to config.yaml
    try:
        import yaml
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('trading', {}).get('symbols', ['XAUUSD', 'EURUSD', 'GBPUSD', 'BRNUSD'])
    except Exception:
        return ['XAUUSD', 'EURUSD', 'GBPUSD', 'BRNUSD']


def db_insert_trade(symbol, action, lot_size, entry_price, spread=0, regime="UNKNOWN", slippage=0):
    """Insert a new trade record into DB with enhanced logging"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, action, lot_size, entry_price, spread, regime, slippage, status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (symbol, action, lot_size, entry_price, spread, regime, slippage, datetime.now().isoformat()))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to insert trade to DB: {e}")
        return None


def db_update_trade(trade_id, exit_price, pnl, spread_at_exit=0, slippage=0):
    """Update trade record when closed with enhanced logging"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET exit_price = ?, pnl = ?, spread_at_exit = ?, slippage = ?, status = 'CLOSED', closed_at = ?
            WHERE id = ?
        """, (exit_price, pnl, spread_at_exit, slippage, datetime.now().isoformat(), trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update trade in DB: {e}")


def db_close_orphaned_trades(mt5_positions):
    """Close OPEN trades in DB that are no longer open in MT5 (mark as closed with real P&L)"""
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        # Fetch history deals ONCE for the whole sync
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        deals = mt5.history_deals_get(from_date, to_date) or []
        # Build map: symbol -> latest OUT deal (entry=1 means closing trade)
        close_deals_by_symbol = {}
        for d in deals:
            if d.entry == 1 and d.symbol:  # OUT (closing deal)
                # Keep the most recent close per symbol
                if d.symbol not in close_deals_by_symbol:
                    close_deals_by_symbol[d.symbol] = d
                elif d.time > close_deals_by_symbol[d.symbol].time:
                    close_deals_by_symbol[d.symbol] = d

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, symbol, lot_size, entry_price, opened_at FROM trades WHERE status='OPEN'")
        open_trades = cursor.fetchall()
        conn.close()

        for trade_id, symbol, lot_size, entry_price, opened_at in open_trades:
            # Check if still open in MT5
            still_open = any(
                p.get('symbol') == symbol
                for p in (mt5_positions or [])
            )
            if not still_open:
                # Look up the matching close deal for this symbol
                close_pnl = 0
                close_price = entry_price
                if symbol in close_deals_by_symbol:
                    close = close_deals_by_symbol[symbol]
                    close_pnl = close.profit
                    close_price = close.price
                db_update_trade(trade_id, close_price, close_pnl)
                logger.info(f"Closed orphan trade #{trade_id} {symbol} entry={entry_price} exit={close_price} P&L={close_pnl}")
    except Exception as e:
        logger.error(f"Failed to close orphaned trades: {e}")


def db_insert_ai_decision(symbol, action, lot_size, reasoning, confidence):
    """Insert AI decision into DB"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ai_decisions (symbol, action, lot_size, reasoning, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, action, lot_size, reasoning, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to insert AI decision to DB: {e}")


class TradingEngine:
    def __init__(self, config):
        self.config = config
        self.mt5 = None
        self.ai = None
        self.running = False
        self.daily_loss = 0
        self.last_reset = datetime.now().date()
        self.last_h4_update = datetime.min  # For tracking H4 updates
        
        # Multi-timeframe configuration for scalping
        self.timeframes = {
            "trend": config.get("timeframes", {}).get("trend", "M15"),     # M15 - intra trend
            "entry": config.get("timeframes", {}).get("entry", "M5"),      # M5 - entry signals
            "confirm": config.get("timeframes", {}).get("confirm", "M1")    # M1 - precise entry
        }
        
        # Loop intervals for scalping (seconds)
        self.scalp_interval = config.get("engine", {}).get("scalp_loop_seconds", 15)  # Check every 15s
        self.trend_interval = config.get("engine", {}).get("trend_loop_seconds", 60)  # Update M15 trend every 60s
        
        # Load symbols from DB or config
        self.symbols = get_enabled_symbols()
        
        # State
        self.market_data = {}        # Multi-timeframe data: {symbol: {H4: {...}, M5: {...}, M15: {...}}}
        self.positions = []
        self.trade_history = []
        self.ai_decisions = []
        self.status = "INITIALIZING"
        
        # Cached H4 trend (changes slowly)
        self.trend_direction = {}    # {symbol: "BULL" | "BEAR" | "SIDEWAYS"}
        
        # Statistics
        self.stats = {
            "trades_today": 0,
            "wins_today": 0,
            "losses_today": 0,
            "pnl_today": 0,
            "total_trades": 0,
            "total_pnl": 0
        }
        
        # Initialize trade manager
        self.trade_manager = TradeManager(None, config)  # Will be set after MT5 init
        
        # Initialize session filter
        self.session_filter = SessionFilter(config)
        
        # Initialize correlation controller
        self.correlation_controller = CorrelationController(config)
        
        # Initialize safeguards
        self.safeguards = Safeguards(None, config)  # Will be set after MT5 init
        
        # Initialize strategies
        self.strategy_manager = StrategyManager(config)
        self.ai_filter = None
        self.news_checker = NewsChecker()

    def initialize(self):
        """Initialize MT5 and AI connections"""
        try:
            self.status = "CONNECTING"
            logger.info("Initializing Trading Engine...")
            
            # Reload symbols from DB in case they changed
            self.symbols = get_enabled_symbols()
            
            # Initialize MT5
            mt5_config = self.config.get("mt5", {})
            self.mt5 = MT5Connector(
                login=mt5_config.get("login"),
                password=mt5_config.get("password"),
                server=mt5_config.get("server"),
                path=mt5_config.get("path")
            )
            
            if not self.mt5.connect():
                self.status = "MT5_CONNECTION_FAILED"
                logger.error("Failed to connect to MT5")
                return False

            # Get current positions from MT5
            self.positions = self.mt5.get_positions()
            logger.info(f"Loaded {len(self.positions)} open positions from MT5")

            # Initialize trade manager with MT5 connection
            self.trade_manager.mt5 = self.mt5

            # Initialize safeguards with MT5 and account info
            account_info = self.mt5.get_account_info()
            self.safeguards.mt5 = self.mt5
            self.safeguards.initialize(account_info)
            
            # Filter positions to only our magic number
            self.positions = self.safeguards.filter_our_positions(self.positions)
            logger.info(f"Safeguards initialized - tracking {len(self.positions)} positions by magic {self.safeguards.magic}")

            # Initialize AI Filter (not AI decision-maker)
            ai_config = self.config.get("ai", {})
            if ai_config.get("api_key"):
                self.ai_filter = AIFilter(
                    provider=ai_config.get("provider", "groq"),
                    api_key=ai_config.get("api_key"),
                    model=ai_config.get("model"),
                    max_tokens=500,
                    temperature=0.3
                )
                logger.info("AI Filter initialized - AI will veto/approve setups")
            else:
                logger.info("AI Filter disabled - using rule-based signals only")
            
            self.status = "READY"
            logger.info(f"Trading Engine initialized - Symbols: {self.symbols}, Timeframes: {self.timeframes}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.status = "ERROR"
            return False

    def start(self):
        """Start the trading loop"""
        if not self.mt5 or not self.mt5.is_connected():
            logger.error("Cannot start - not connected to MT5")
            return False
            
        self.running = True
        self.status = "RUNNING"
        logger.info("Trading Engine STARTED - Multi-timeframe scalping active")
        
        last_scalp_check = time.time()
        last_trend_update = time.time()
        last_trade_management = time.time()
        last_broadcast = time.time()
        
        while self.running:
            try:
                now = time.time()
                
                # Connection health check
                if not self.safeguards.check_connection():
                    logger.warning("Connection lost - attempting reconnect...")
                    reconnect_result = self.safeguards.handle_disconnect()
                    if not reconnect_result.get("reconnected"):
                        logger.error("Reconnection failed - pausing")
                        self.status = "RECONNECTING"
                        time.sleep(5)
                        continue
                    else:
                        # Re-sync positions after reconnect
                        self.positions = reconnect_result.get("positions", [])
                        self.status = "RUNNING"
                
                # Check drawdown kill switch
                account = self.mt5.get_account_info()
                should_kill, kill_reason = self.safeguards.should_kill_switch(account)
                if should_kill:
                    logger.warning(f"KILL SWITCH: {kill_reason}")
                    self.trade_manager.close_all_positions(kill_reason)
                    self.status = "KILL_SWITCH"
                    # Don't stop - wait for manual intervention
                
                # TRADE MANAGEMENT LOOP (every 5 seconds) - Check positions
                if now - last_trade_management >= 5:
                    self._trade_management_loop()
                    last_trade_management = now
                
                # SCALP LOOP (every 15 seconds) - Entry signals
                if now - last_scalp_check >= self.scalp_interval:
                    self._scalp_loop()
                    last_scalp_check = now
                
                # TREND LOOP (every 60 seconds) - M15 trend update
                if now - last_trend_update >= self.trend_interval:
                    self._trend_loop()
                    last_trend_update = now

                # Broadcast updates to dashboard (every 5 seconds)
                if now - last_broadcast >= 5:
                    self._broadcast_update()
                    last_broadcast = now

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                self.status = "ERROR"

        self.status = "STOPPED"
        logger.info("Trading Engine STOPPED")

    def stop(self):
        """Stop the trading loop"""
        self.running = False
        if self.mt5:
            self.disconnect()

    def _broadcast_update(self):
        """Broadcast real-time updates to dashboard via WebSocket"""
        try:
            from dashboard.app import socketio, db_get_analytics

            # Get current positions from MT5
            positions = []
            if self.mt5:
                try:
                    positions = self.mt5.get_positions() or []
                    # Convert datetime objects to strings
                    for p in positions:
                        if 'time' in p and hasattr(p['time'], 'isoformat'):
                            p['time'] = p['time'].isoformat()
                    # Sync DB - close orphaned trades
                    db_close_orphaned_trades(positions)
                except:
                    pass

            # Get recent AI decisions (extract nested decision object)
            decisions = []
            for item in (self.ai_decisions[-5:] if self.ai_decisions else []):
                if isinstance(item, dict) and 'decision' in item:
                    d = item['decision'].copy()
                    ts = item.get('timestamp', '')
                    if hasattr(ts, 'isoformat'):
                        d['created_at'] = ts.isoformat()
                    else:
                        d['created_at'] = str(ts)
                    decisions.append(d)

            # Broadcast positions
            socketio.emit('positions_update', positions)

            # Broadcast AI decisions
            socketio.emit('decisions_update', decisions)

            # Broadcast analytics
            from engine.trade_counter import get_trade_stats
            stats = get_trade_stats()
            analytics = db_get_analytics()
            analytics.update(stats)
            socketio.emit('analytics_update', analytics)

            # Broadcast history (closed trades)
            from dashboard.app import db_get_trades
            history = db_get_trades(20)
            socketio.emit('history_update', history)

            # Broadcast status
            socketio.emit('status_update', {'status': self.status})

            # Broadcast market data
            market_data = {}
            for symbol in (self.symbols or []):
                try:
                    tick = self.mt5.get_current_price(symbol) if self.mt5 else None
                    spread = self.mt5.get_spread(symbol) if self.mt5 else 0
                    trend = 'SIDEWAYS'
                    if hasattr(self, 'trend_direction') and symbol in self.trend_direction:
                        trend = self.trend_direction[symbol]
                    market_data[symbol] = {
                        'bid': tick.get('bid', 0) if tick else 0,
                        'ask': tick.get('ask', 0) if tick else 0,
                        'spread': spread or 0,
                        'trend_direction': trend
                    }
                except:
                    pass
            socketio.emit('market_update', market_data)

        except Exception as e:
            logger.error(f"Broadcast error: {e}")

    def _trade_management_loop(self):
        """Manage open positions - breakeven, trailing stops, partial TP"""
        if not self.trade_manager or not self.trade_manager.mt5:
            return
        
        try:
            # Check daily loss limit
            if self.trade_manager.check_daily_loss_limit():
                logger.warning("Daily loss limit breached - activating kill switch")
                result = self.trade_manager.close_all_positions("Daily loss limit")
                self.status = "KILL_SWITCH"
                logger.info(f"Kill switch activated: {result}")
                return
            
            # Manage all positions
            results = self.trade_manager.manage_all_positions()
            
            if results['managed'] > 0:
                logger.info(f"Trade management: {results['managed']} positions managed")
                for action in results['actions']:
                    logger.info(f"  {action['symbol']} ticket {action['ticket']}: "
                               f"BE={action.get('breakeven')}, "
                               f"PTP={action.get('partial_tp')}, "
                               f"Trail={action.get('trailing')}")
        except Exception as e:
            logger.error(f"Trade management error: {e}")

    def _check_regime_filter(self, symbol, action):
        """
        Check if market regime is suitable for trading.
        Uses ADX to detect trending vs ranging markets.
        """
        try:
            # Get M15 data for regime detection
            bars = self.mt5.get_ohlcv(symbol, "M15", 50)
            if not bars or len(bars) < 20:
                return {"allowed": True, "reason": "Insufficient data for regime check"}
            
            closes = [b["close"] for b in bars]
            highs = [b["high"] for b in bars]
            lows = [b["low"] for b in bars]
            
            # Detect regime
            regime = detect_market_regime(closes, highs, lows, closes)
            
            # Check if tradeable for trend strategy
            tradeable, reason = is_tradeable_regime(regime, strategy_type="trend")
            
            if not tradeable:
                return {"allowed": False, "reason": reason, "regime": regime}
            
            # Also check if direction aligns with our trade
            if regime["regime"] == "TRENDING":
                if action == "BUY" and regime["direction"] == "BEAR":
                    return {"allowed": False, "reason": f"ADX={regime['adx']}: Bear trend, no buys", "regime": regime}
                if action == "SELL" and regime["direction"] == "BULL":
                    return {"allowed": False, "reason": f"ADX={regime['adx']}: Bull trend, no sells", "regime": regime}
            
            return {"allowed": True, "reason": reason, "regime": regime}
            
        except Exception as e:
            logger.error(f"Regime check failed for {symbol}: {e}")
            return {"allowed": True, "reason": "Regime check error - allowing trade"}

    def _scalp_loop(self):
        """Fast loop - Check M5 for entry signals"""
        # Reset daily stats if new day
        self._check_daily_reset()
        
        # Check risk limits
        if self._check_risk_limits():
            self.status = "RISK_LIMIT_REACHED"
            logger.warning("Risk limit reached, skipping trading")
            return
        
        # Check session filter - only trade during optimal hours
        session_check = self.session_filter.is_trading_allowed()
        if not session_check["allowed"]:
            logger.debug(f"Session filter: {session_check['reason']}")
            return
        
        # Get current positions
        self.positions = self.mt5.get_positions()
        
        # Reload symbols from DB (in case changed via dashboard)
        self.symbols = get_enabled_symbols()
        
        # Analyze each symbol on entry timeframe (M5)
        self.market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get multi-timeframe analysis
                analysis = self._analyze_symbol_multitimeframe(symbol)
                self.market_data[symbol] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                
        # Get strategy signals
        strategy_decisions = self._get_strategy_decisions()
        
        # Execute strategy signals
        for symbol, decision in strategy_decisions.items():
            if decision.get("signal") not in ["HOLD", "SKIP", None]:
                self._execute_decision(decision)
        
        # Update status
        self.status = "RUNNING"

    def _trend_loop(self):
        """Slow loop - Update H4 trend direction"""
        for symbol in self.symbols:
            try:
                # Get H4 analysis for trend
                h4_analysis = self._analyze_symbol(symbol, self.timeframes["trend"])
                if h4_analysis:
                    trend = h4_analysis.get("overall", "SIDEWAYS")
                    if trend != self.trend_direction.get(symbol):
                        logger.info(f"{symbol} trend changed to: {trend}")
                        self.trend_direction[symbol] = trend
            except Exception as e:
                logger.error(f"Failed to update trend for {symbol}: {e}")

    def _analyze_symbol_multitimeframe(self, symbol):
        """Analyze a symbol across multiple timeframes"""
        analysis = {}
        
        # Trend timeframe (H4) - for direction
        trend_tf = self.timeframes["trend"]
        analysis[trend_tf] = self._analyze_symbol(symbol, trend_tf)
        
        # Entry timeframe (M5) - for scalp entries
        entry_tf = self.timeframes["entry"]
        analysis[entry_tf] = self._analyze_symbol(symbol, entry_tf)
        
        # Confirm timeframe (M15) - for momentum
        confirm_tf = self.timeframes["confirm"]
        analysis[confirm_tf] = self._analyze_symbol(symbol, confirm_tf)
        
        # Add current trend direction to all timeframes
        analysis["trend_direction"] = self.trend_direction.get(symbol, "SIDEWAYS")
        
        # Check if we already have position for this symbol
        analysis["has_position"] = any(p["symbol"] == symbol for p in self.positions)
        
        return analysis

    def _analyze_symbol(self, symbol, timeframe):
        """Analyze a single symbol on a specific timeframe"""
        # Get OHLCV data
        bars = self.mt5.get_ohlcv(symbol, timeframe, 100)

        if len(bars) < 50:
            logger.warning(f"Not enough data for {symbol} {timeframe}")
            return None

        # Extract price data
        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        volumes = [b.get("volume", 0) for b in bars]

        # Get current price
        price_info = self.mt5.get_current_price(symbol)
        if price_info:
            closes.append(price_info["bid"])

        # Calculate indicators and analysis
        analysis = analyze_market(closes, highs, lows, timeframe)
        # Inject volume data
        if analysis is not None:
            analysis["volumes"] = volumes
            if price_info:
                analysis["bid"] = price_info["bid"]
                analysis["ask"] = price_info["ask"]
            analysis["spread"] = round((price_info["ask"] - price_info["bid"]) * 10000, 1)
            
        return analysis

    def _get_ai_decision(self):
        """Get trading decision from AI with multi-timeframe data"""
        if not self.ai:
            return {"error": "AI not initialized", "decision": "SKIP"}
            
        if not self.config.get("ai", {}).get("api_key"):
            return {"error": "No API key", "decision": "SKIP"}
            
        try:
            recent_trades = self.trade_history[-10:] if self.trade_history else None
            
            # Call AI with multi-timeframe data
            decision = self.ai.analyze_market(
                market_data=self.market_data,
                positions=self.positions,
                trade_history=recent_trades
            )
            
            # Save AI decision to DB (all decisions, including HOLD)
            if decision:
                # Save all decisions if _all_decisions is present
                all_decisions = decision.get('_all_decisions', [decision])
                for d in all_decisions:
                    # Skip decisions without valid symbol
                    if not d.get("symbol") or d.get("symbol") == "UNKNOWN":
                        continue
                    db_insert_ai_decision(
                        symbol=d.get("symbol", "UNKNOWN"),
                        action=d.get("action", "HOLD"),
                        lot_size=d.get("lot_size", 0.01),
                        reasoning=(d.get("entry_reason") or d.get("reasoning") or (d.get("raw_response", "")[:500] if d.get("raw_response") else ""))[:500],
                        confidence=d.get("confidence", 0)
                    )
            
            # Log decision
            self.ai_decisions.append({
                "timestamp": datetime.now(),
                "decision": decision,
                "market_data": self.market_data
            })
            
            # Keep only last 50 decisions
            self.ai_decisions = self.ai_decisions[-50:]
            
            return decision
            
        except Exception as e:
            logger.error(f"AI decision failed: {e}")
            return {"error": str(e), "decision": "SKIP"}

    def _get_strategy_decisions(self):
        """
        Get trading decisions from rule-based strategies.
        AI only acts as a filter to veto setups.
        """
        decisions = {}
        
        try:
            for symbol in self.symbols:
                # Check if we already have position for this symbol
                if any(p["symbol"] == symbol for p in self.positions):
                    continue
                
                # Get multi-timeframe market data
                market_data = self.market_data.get(symbol, {})
                if not market_data:
                    continue

                # Check regime filter first
                closes = market_data.get('closes', [])
                highs = market_data.get('highs', [])
                lows = market_data.get('lows', [])
                if len(closes) >= 20 and len(highs) >= 20 and len(lows) >= 20:
                    regime = detect_market_regime(closes, highs, lows, closes)
                    if not is_tradeable_regime(regime, "trend"):
                        continue
                else:
                    continue
                
                # Check session filter
                if not self.session_filter.is_tradeable_time(symbol):
                    continue
                
                # Check news filter
                if self.news_checker.is_news_window(symbol):
                    continue
                
                # Get strategy signal
                setup = self.strategy_manager.get_signal(market_data)

                # If no clear signal, skip
                if setup.get("signal") in ["HOLD", "NONE", None]:
                    continue

                # Check confidence threshold from strategy config
                strategy_obj = self.strategy_manager.strategies.get(self.strategy_manager.active_strategy)
                min_conf = 60  # default
                if strategy_obj and hasattr(strategy_obj, 'min_confidence'):
                    min_conf = strategy_obj.min_confidence

                setup_confidence = setup.get("confidence", 0)
                if setup_confidence < min_conf:
                    logger.info(f"{symbol}: REJECTED - confidence {setup_confidence}% < min {min_conf}%")
                    continue

                # Check momentum (is price moving strongly enough?)
                m5_data = market_data.get('M5', {})
                m5_closes = m5_data.get('closes', [])
                if len(m5_closes) >= 11:
                    mom_ok, momentum, mom_dir = is_momentum_strong(m5_closes, period=10, min_momentum_pct=0.05)
                    if not mom_ok:
                        logger.info(f"{symbol}: REJECTED - momentum too weak ({momentum:.3f}%)")
                        continue
                    # Momentum should align with signal
                    signal = setup.get("signal")
                    if signal == "BUY" and mom_dir != "BULL":
                        logger.info(f"{symbol}: REJECTED - momentum bearish ({momentum:.3f}%) but signal BUY")
                        continue
                    if signal == "SELL" and mom_dir != "BEAR":
                        logger.info(f"{symbol}: REJECTED - momentum bullish ({momentum:.3f}%) but signal SELL")
                        continue

                # Check volume (is current volume above average?)
                m5_volumes = m5_data.get('volumes', [])
                if len(m5_volumes) >= 21:
                    vol_ok, curr_vol, avg_vol, vol_ratio = is_volume_confirmed(m5_volumes, period=20, multiplier=1.0)
                    if not vol_ok:
                        logger.info(f"{symbol}: REJECTED - volume below average (ratio {vol_ratio:.2f})")
                        continue

                # Check news filter (no high-impact news in next 30 min)
                if self.news_checker:
                    try:
                        news_events = self.news_checker.check_upcoming_news(symbol, hours_ahead=0.5)
                        if news_events:
                            news_name = news_events[0].get('name', 'Unknown')
                            hours_away = news_events[0].get('hours_away', 0)
                            logger.info(f"{symbol}: REJECTED - news window: {news_name} in {hours_away}h")
                            continue
                    except Exception as e:
                        logger.error(f"News check error: {e}")

                # Check risk-reward ratio
                entry = setup.get("entry_zone")
                sl = setup.get("stop_loss")
                tp = setup.get("take_profit")
                if entry and sl and tp:
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    if risk > 0:
                        rr = reward / risk
                        min_rr = getattr(strategy_obj, 'min_risk_reward', 1.5) if strategy_obj else 1.5
                        if rr < min_rr:
                            logger.info(f"{symbol}: REJECTED - RR {rr:.2f} < min {min_rr}")
                            continue

                # Apply AI filter (veto/approve)
                if self.ai_filter:
                    news_events = self.news_checker.check_upcoming_news(symbol)
                    ai_result = self.ai_filter.evaluate_setup(
                        symbol=symbol,
                        setup=setup,
                        market_data=market_data,
                        news_events=news_events,
                        regime=regime
                    )

                    # If AI vetoes, skip this setup but log the reason
                    if not ai_result.get("approved", True):
                        veto_reason = ai_result.get("reason", "No reason given")
                        logger.info(f"{symbol}: AI VETOED - {veto_reason}")
                        # Store veto as a decision so dashboard can show it
                        decisions[symbol] = {
                            "action": "HOLD",
                            "symbol": symbol,
                            "lot_size": 0,
                            "stop_loss_pips": 0,
                            "take_profit_pips": 0,
                            "reasoning": f"AI VETO: {veto_reason}",
                            "signal": "VETO",
                            "strategy": "rule_based",
                            "confidence": setup_confidence,
                            "entry_price": None,
                            "stop_loss": None,
                            "take_profit": None,
                            "market_regime": regime.get("regime") if regime else "UNKNOWN",
                            "vetoed": True,
                            "veto_reason": veto_reason
                        }
                        continue

                    # Apply AI adjustments if any
                    adjustments = ai_result.get("adjustments", {})
                    if adjustments:
                        if adjustments.get("sl"):
                            setup["stop_loss"] = adjustments["sl"]
                        if adjustments.get("tp"):
                            setup["take_profit"] = adjustments["tp"]

                # Build decision dict
                action = "BUY" if setup.get("signal") == "BUY" else "SELL"

                decisions[symbol] = {
                    "action": action,
                    "symbol": symbol,
                    "lot_size": setup.get("lot_size", self.config.get("trading", {}).get("default_lot_size", 0.01)),
                    "stop_loss_pips": setup.get("sl_pips", self.config.get("trading", {}).get("default_stop_loss_pips", 30)),
                    "take_profit_pips": setup.get("tp_pips", self.config.get("trading", {}).get("default_take_profit_pips", 50)),
                    "reasoning": setup.get("reason", "Strategy signal"),
                    "signal": action,
                    "strategy": "rule_based",
                    "confidence": setup_confidence,
                    "entry_price": setup.get("entry_zone"),
                    "stop_loss": setup.get("stop_loss"),
                    "take_profit": setup.get("take_profit"),
                    "market_regime": regime.get("regime") if regime else "UNKNOWN",
                    "vetoed": False
                }

        except Exception as e:
            logger.error(f"Strategy decisions failed: {e}")

        return decisions

    def _execute_decision(self, decision):
        """Execute trading decision"""
        if not decision:
            return

        action = decision.get("action", "HOLD")
        symbol = decision.get("symbol")

        if action == "HOLD" or action == "SKIP":
            logger.debug(f"{symbol}: HOLD - {decision.get('reasoning', 'No clear setup')}")
            return

        # Check if we already have position
        if any(p["symbol"] == symbol for p in self.positions):
            logger.info(f"{symbol}: Already have position, skipping")
            return

        # Check daily trade limit and loss limit
        try:
            from engine.trade_counter import can_trade_today, calculate_lot_adjustment
            strategy_obj = self.strategy_manager.strategies.get(self.strategy_manager.active_strategy) if self.strategy_manager else None
            max_trades = getattr(strategy_obj, 'max_trades_per_day', 10) if strategy_obj else 10
            max_daily_loss = getattr(strategy_obj, 'daily_loss_limit_pct', 3.0) if strategy_obj else 3.0
            account_balance = self.mt5.get_account_info().get('balance', 0) if self.mt5 else 0
            can_trade, reason = can_trade_today(max_trades, max_daily_loss, account_balance)
            if not can_trade:
                logger.info(f"{symbol}: {reason}")
                return
        except Exception as e:
            logger.error(f"Trade counter error: {e}")

        # Check market regime filter
        regime_check = self._check_regime_filter(symbol, action)
        if not regime_check["allowed"]:
            logger.info(f"{symbol}: {regime_check['reason']}")
            return

        # Check correlation filter
        lot_size = decision.get("lot_size", self.config.get("trading", {}).get("default_lot_size", 0.01))

        # Apply streak-based lot adjustment
        try:
            from engine.trade_counter import calculate_lot_adjustment
            strategy_obj = self.strategy_manager.strategies.get(self.strategy_manager.active_strategy) if self.strategy_manager else None
            if strategy_obj:
                multiplier = calculate_lot_adjustment(
                    loss_reduction_pct=strategy_obj.loss_streak_reduction_pct,
                    loss_threshold=strategy_obj.loss_streak_threshold,
                    win_increase_pct=strategy_obj.win_streak_increase_pct,
                    win_threshold=strategy_obj.win_streak_threshold
                )
                original_lot = lot_size
                lot_size = round(lot_size * multiplier, 2)
                if multiplier != 1.0:
                    logger.info(f"{symbol}: Lot adjusted {original_lot} -> {lot_size} (streak multiplier {multiplier})")
        except Exception as e:
            logger.error(f"Streak adjustment error: {e}")

        corr_check = self.correlation_controller.can_open_position(symbol, self.positions, lot_size)
        if not corr_check["allowed"]:
            logger.info(f"{symbol}: {corr_check['reason']}")
            return
        
        # Execute trade
        sl_pips = decision.get("stop_loss_pips", self.config.get("trading", {}).get("default_stop_loss_pips", 30))
        tp_pips = decision.get("take_profit_pips", self.config.get("trading", {}).get("default_take_profit_pips", 50))
        
        # Use entry timeframe price
        entry_data = self.market_data.get(symbol, {}).get(self.timeframes["entry"], {})
        current_price = entry_data.get("bid")
        if not current_price:
            logger.error(f"Cannot get price for {symbol}")
            return
            
        # Get symbol info for pip calculation
        symbol_info = self.mt5.get_symbol_info(symbol)
        digits = symbol_info.get("digits", 5) if symbol_info else 5
        pip_multiplier = 10 ** (digits - 4) if digits == 5 else 10 ** (digits - 2)
        
        sl_price = None
        tp_price = None
        
        if action == "BUY":
            sl_price = current_price - (sl_pips / pip_multiplier)
            tp_price = current_price + (tp_pips / pip_multiplier)
        else:  # SELL
            sl_price = current_price + (sl_pips / pip_multiplier)
            tp_price = current_price - (tp_pips / pip_multiplier)
            
        # Send order
        result = self.mt5.send_order(
            symbol=symbol,
            order_type=action,
            volume=lot_size,
            sl=sl_price,
            tp=tp_price,
            comment=f"AI:{decision.get('strategy', 'AI')}"
        )
        
        # Get spread at entry and market regime
        spread_at_entry = self.mt5.get_spread(symbol) if hasattr(self.mt5, 'get_spread') else 0
        market_regime = regime_check.get("regime", {}).get("regime", "UNKNOWN")
        
        # Calculate slippage (difference between expected and actual fill price)
        expected_price = current_price
        actual_price = result.get("price", expected_price)
        slippage_pips = abs(actual_price - expected_price) * pip_multiplier if actual_price else 0
        
        if result:
            self.stats["trades_today"] += 1
            self.stats["total_trades"] += 1
            logger.info(f"ORDER SENT: {action} {lot_size} {symbol} @ {current_price} | Spread: {spread_at_entry} | Regime: {market_regime}")
            
            # Insert trade to DB with enhanced logging
            trade_id = db_insert_trade(
                symbol, action, lot_size, current_price,
                spread=spread_at_entry,
                regime=market_regime,
                slippage=round(slippage_pips, 1)
            )
            
            # Log trade
            trade_log = {
                "ticket": result.get("ticket"),
                "trade_id": trade_id,
                "symbol": symbol,
                "type": action,
                "volume": lot_size,
                "entry_price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "entry_time": datetime.now(),
                "strategy": decision.get("strategy", "AI"),
                "ai_confidence": decision.get("confidence", 0),
                "ai_reasoning": decision.get("reasoning", ""),
                "market_analysis": self.market_data.get(symbol, {})
            }
            self.trade_history.append(trade_log)
            self._save_trade_log(trade_log)

    def _check_risk_limits(self):
        """Check if risk limits are reached"""
        risk = self.config.get("risk", {})
        
        max_daily_loss = risk.get("max_daily_loss_percent", 5.0)
        max_drawdown = risk.get("max_drawdown_percent", 10.0)
        
        account = self.mt5.get_account_info()
        if not account:
            return True
            
        initial_balance = 100000  # Demo account
        current_drawdown = ((initial_balance - account["equity"]) / initial_balance) * 100
        
        if current_drawdown > max_drawdown:
            logger.error(f"Max drawdown reached: {current_drawdown:.2f}%")
            return True
            
        if self.stats["pnl_today"] < 0:
            daily_loss_percent = abs(self.stats["pnl_today"]) / initial_balance * 100
            if daily_loss_percent > max_daily_loss:
                logger.error(f"Daily loss limit reached: {daily_loss_percent:.2f}%")
                return True
                
        return False

    def _check_daily_reset(self):
        """Reset daily stats if new day"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.stats["trades_today"] = 0
            self.stats["wins_today"] = 0
            self.stats["losses_today"] = 0
            self.stats["pnl_today"] = 0
            self.last_reset = today
            logger.info("Daily stats reset")

    def _save_trade_log(self, trade):
        """Save trade to log file"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
            
            logs = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    logs = json.load(f)
                    
            logs.append({
                **trade,
                "entry_time": trade["entry_time"].isoformat() if isinstance(trade["entry_time"], datetime) else str(trade["entry_time"])
            })
            
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save trade log: {e}")

    def get_status(self):
        """Get current engine status"""
        return {
            "status": self.status,
            "running": self.running,
            "connected": self.mt5.is_connected() if self.mt5 else False,
            "market_data": self.market_data,
            "positions": self.positions,
            "stats": self.stats,
            "ai_decisions": self.ai_decisions[-5:] if self.ai_decisions else [],
            "timeframes": self.timeframes,
            "trend_direction": self.trend_direction,
            "timestamp": datetime.now()
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import yaml
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except:
        config = {
            "ai": {"provider": "claude", "api_key": None},
            "trading": {"symbols": ["EURUSD"]},
            "timeframes": {"trend": "H4", "entry": "M5", "confirm": "M15"},
            "engine": {"scalp_loop_seconds": 5, "trend_loop_seconds": 60}
        }
        
    engine = TradingEngine(config)
    
    if engine.initialize():
        print("Engine initialized")
        print("Status:", engine.get_status())
        engine.stop()
    else:
        print("Failed to initialize engine")

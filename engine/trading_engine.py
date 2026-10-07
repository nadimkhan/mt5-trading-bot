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
from engine.indicators import analyze_market
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


def db_insert_trade(symbol, action, lot_size, entry_price):
    """Insert a new trade record into DB"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, action, lot_size, entry_price, status, opened_at)
            VALUES (?, ?, ?, ?, 'OPEN', ?)
        """, (symbol, action, lot_size, entry_price, datetime.now().isoformat()))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to insert trade to DB: {e}")
        return None


def db_update_trade(trade_id, exit_price, pnl):
    """Update trade record when closed"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades 
            SET exit_price = ?, pnl = ?, status = 'CLOSED', closed_at = ?
            WHERE id = ?
        """, (exit_price, pnl, datetime.now().isoformat(), trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update trade in DB: {e}")


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

            # Initialize AI
            ai_config = self.config.get("ai", {})
            self.ai = AIAnalyzer(
                provider=ai_config.get("provider", "claude"),
                api_key=ai_config.get("api_key"),
                model=ai_config.get("model"),
                max_tokens=ai_config.get("max_tokens", 1000),
                temperature=ai_config.get("temperature", 0.7)
            )
            
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
        
        while self.running:
            try:
                now = time.time()
                
                # SCALP LOOP (every 5 seconds) - Entry signals
                if now - last_scalp_check >= self.scalp_interval:
                    self._scalp_loop()
                    last_scalp_check = now
                
                # TREND LOOP (every 60 seconds) - H4 trend update
                if now - last_trend_update >= self.trend_interval:
                    self._trend_loop()
                    last_trend_update = now
                    
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

    def _scalp_loop(self):
        """Fast loop - Check M5 for entry signals"""
        # Reset daily stats if new day
        self._check_daily_reset()
        
        # Check risk limits
        if self._check_risk_limits():
            self.status = "RISK_LIMIT_REACHED"
            logger.warning("Risk limit reached, skipping trading")
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
                
        # Send to AI for decision
        if self.config.get("ai", {}).get("api_key"):
            decision = self._get_ai_decision()

            # Handle multiple decisions from AI (one per symbol)
            if decision and "_all_decisions" in decision:
                all_decisions = decision.pop("_all_decisions")
                # Execute ALL decisions
                for dec in all_decisions:
                    self._execute_decision(dec)
                decision["_all_decisions"] = all_decisions
            else:
                # Execute single decision
                self._execute_decision(decision)
        else:
            decision = {"error": "AI not configured", "decision": "SKIP"}
        
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
        
        # Get current price
        price_info = self.mt5.get_current_price(symbol)
        if price_info:
            closes.append(price_info["bid"])
            
        # Calculate indicators and analysis
        analysis = analyze_market(closes, highs, lows, timeframe)
        
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
                        reasoning=d.get("reasoning", d.get("raw_response", "")[:500] if d.get("raw_response") else ""),
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
            
        # Check spread - removed, spread is broker-defined
            
        # Execute trade
        lot_size = decision.get("lot_size", self.config.get("trading", {}).get("default_lot_size", 0.01))
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
        
        if result:
            self.stats["trades_today"] += 1
            self.stats["total_trades"] += 1
            logger.info(f"ORDER SENT: {action} {lot_size} {symbol} @ {current_price}")
            
            # Insert trade to DB
            trade_id = db_insert_trade(symbol, action, lot_size, current_price)
            
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

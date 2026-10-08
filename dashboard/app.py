"""
Dashboard - Web interface to monitor the trading bot with real-time WebSocket updates
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import logging
import sqlite3
from datetime import datetime
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.constants import DB_PATH

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'mt5-trading-bot-secret'
# Disable Jinja2 template cache so HTML changes appear on browser refresh
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

engine = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            enabled INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            lot_size REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl REAL,
            status TEXT DEFAULT 'OPEN',
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            lot_size REAL,
            reasoning TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    # Migration: enable any legacy disabled symbols
    db_enable_existing_symbols()
    _init_default_symbols()


def _init_default_symbols():
    """Initialize default symbols - enable by default so they're immediately tradable"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM symbols")
    if cursor.fetchone()[0] == 0:
        # First-time setup: insert defaults as ENABLED
        for symbol in ['XAUUSD', 'EURUSD', 'GBPUSD', 'BRNUSD']:
            cursor.execute("INSERT OR IGNORE INTO symbols (symbol, enabled) VALUES (?, 1)", (symbol,))
        conn.commit()
        logger.info("Initialized 4 default symbols as enabled")
    conn.close()


def db_enable_existing_symbols():
    """One-time migration: enable any existing symbols that were initialized as disabled.
    This handles users who had the old code that inserted with enabled=0."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Enable all symbols that have no explicit user action recorded
    # (We do this only if there are no enabled symbols yet - safety check)
    cursor.execute("SELECT COUNT(*) FROM symbols WHERE enabled = 1")
    enabled_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM symbols")
    total_count = cursor.fetchone()[0]
    # If we have symbols but none enabled, this is a legacy state - auto-enable them
    if total_count > 0 and enabled_count == 0:
        cursor.execute("UPDATE symbols SET enabled = 1")
        conn.commit()
        logger.info(f"Migration: Enabled {total_count} legacy symbols (previously inserted as disabled)")
    conn.close()


def db_get_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_trades = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
    closed_trades = cursor.fetchone()[0]
    open_trades = 0
    if engine and engine.mt5:
        try:
            positions = engine.mt5.get_positions()
            open_trades = len(positions) if positions else 0
        except Exception:
            pass
    cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0 AND status = 'CLOSED'")
    win_count = cursor.fetchone()[0]
    win_rate = (win_count / closed_trades * 100) if closed_trades > 0 else 0
    cursor.execute("SELECT SUM(pnl) FROM trades WHERE status = 'CLOSED'")
    total_pnl = cursor.fetchone()[0] or 0
    conn.close()
    return {'total_trades': total_trades, 'closed_trades_count': closed_trades,
            'open_trades_count': open_trades, 'win_rate': round(win_rate, 1), 'total_pnl': round(total_pnl, 2)}


@app.after_request
def add_no_cache_headers(response):
    """Prevent caching of dashboard files so UI changes appear without restart"""
    if response.content_type and ('html' in response.content_type or 'css' in response.content_type or 'javascript' in response.content_type):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    response = app.make_response(render_template('dashboard.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/status')
def api_status():
    status = "STOPPED"
    if engine:
        status = engine.status
    return jsonify({"status": status})


@app.route('/api/market')
def api_market():
    """Get current market data for all symbols"""
    global engine
    market_data = {}
    if engine and engine.mt5:
        for symbol in (engine.symbols or []):
            try:
                tick = engine.mt5.get_current_price(symbol)
                spread = engine.mt5.get_spread(symbol)
                trend = 'SIDEWAYS'
                if hasattr(engine, 'trend_direction') and symbol in engine.trend_direction:
                    trend = engine.trend_direction[symbol]
                market_data[symbol] = {
                    'bid': tick.get('bid', 0) if tick else 0,
                    'ask': tick.get('ask', 0) if tick else 0,
                    'spread': spread or 0,
                    'trend_direction': trend
                }
            except Exception as e:
                market_data[symbol] = {'bid': 0, 'ask': 0, 'spread': 0, 'trend_direction': 'SIDEWAYS'}
    return jsonify(market_data)


@app.route('/api/positions')
def api_positions():
    if engine and engine.mt5:
        try:
            positions = engine.mt5.get_positions() or []
            # Convert datetime to string for JSON
            for p in positions:
                if 'time' in p and hasattr(p['time'], 'isoformat'):
                    p['time'] = p['time'].isoformat()
            return jsonify(positions)
        except Exception:
            return jsonify([])
    return jsonify([])


@app.route('/api/analytics')
def api_analytics():
    """Get comprehensive trade analytics"""
    from engine.trade_counter import get_trade_stats
    stats = get_trade_stats()
    basic = db_get_analytics()
    basic.update(stats)
    return jsonify(basic)


@app.route('/api/decisions')
def api_decisions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ai_decisions ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        # Convert datetime to string
        for k, v in d.items():
            if hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
        result.append(d)
    return jsonify(result)


def db_get_trades(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for k, v in d.items():
            if hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.route('/api/history')
def api_history():
    """Get closed trades - sync from MT5 history + local DB"""
    global engine
    trades = []
    if engine and engine.mt5:
        try:
            from datetime import datetime, timedelta
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            deals = engine.mt5.get_history_deals(from_date, to_date) or []

            # MT5: entry=0 = IN (open), entry=1 = OUT (close)
            in_deals = [d for d in deals if d.get('entry') == 0]
            out_deals = [d for d in deals if d.get('entry') == 1]

            # Match IN with OUT by symbol (each close has the profit)
            for out_deal in out_deals:
                symbol = out_deal.get('symbol', 'UNKNOWN')
                # Find the matching IN deal
                in_deal = None
                for d in in_deals:
                    if d.get('symbol') == symbol:
                        in_deal = d
                        break
                if not in_deal:
                    continue
                # IN type 0=BUY, type 1=SELL
                entry_type = in_deal.get('type', 0)
                action = 'BUY' if entry_type == 0 else 'SELL'
                ts = out_deal.get('time', '')
                ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                in_ts = in_deal.get('time', '')
                in_ts_str = in_ts.isoformat() if hasattr(in_ts, 'isoformat') else str(in_ts)
                trades.append({
                    'symbol': symbol,
                    'action': action,
                    'entry_price': in_deal.get('price', 0),
                    'exit_price': out_deal.get('price', 0),
                    'pnl': out_deal.get('profit', 0),
                    'volume': out_deal.get('volume', 0),
                    'opened_at': in_ts_str,
                    'closed_at': ts_str,
                    'status': 'CLOSED'
                })
        except Exception as e:
            logger.error(f"Failed to get MT5 history: {e}")
    if not trades:
        trades = db_get_trades(50)
    trades.sort(key=lambda t: t.get('closed_at', ''), reverse=True)
    return jsonify(trades[:20])


@app.route('/api/strategies')
def api_get_strategies():
    """Get all strategies with their configurations"""
    from strategies.strategy_config import load_configs
    configs = load_configs()
    current = 'scalp'
    if engine and hasattr(engine, 'strategy_manager'):
        current = engine.strategy_manager.active_strategy
    return jsonify({"available": list(configs.keys()), "current": current, "configs": configs})


@app.route('/api/strategies/config', methods=['GET'])
def api_get_strategy_configs():
    """Get all strategy configurations"""
    from strategies.strategy_config import load_configs
    return jsonify(load_configs())


@app.route('/api/strategies/config', methods=['POST'])
def api_update_strategy_config():
    """Update a strategy parameter or timeframe"""
    from strategies.strategy_config import load_configs, save_configs, update_strategy_param
    data = request.get_json()
    strategy = data.get('strategy')
    param = data.get('param')
    value = data.get('value')

    configs = load_configs()
    success, msg = update_strategy_param(configs, strategy, param, value)
    if success:
        save_configs(configs)
        # Apply to engine if running (no restart needed)
        if engine and hasattr(engine, 'strategy_manager'):
            strategy_obj = engine.strategy_manager.strategies.get(strategy)
            if strategy_obj and hasattr(strategy_obj, 'reload_config'):
                strategy_obj.reload_config(configs)
            # If timeframe changed, apply to engine immediately
            if param in ['trend', 'entry', 'confirm', 'scalp_interval', 'trend_interval']:
                if hasattr(engine, '_apply_strategy_timeframes'):
                    engine._apply_strategy_timeframes()
                    logger.info(f"Timeframes applied: {engine.timeframes}")
        return jsonify({"status": "ok", "strategy": strategy, "param": param, "value": value})
    return jsonify({"error": msg}), 400


@app.route('/api/strategies/reset', methods=['POST'])
def api_reset_strategies():
    """Reset all strategies to default configs"""
    from strategies.strategy_config import reset_configs
    configs = reset_configs()
    if configs:
        return jsonify({"status": "ok", "configs": configs})
    return jsonify({"error": "Failed to reset"}), 500


@app.route('/api/strategies/reload', methods=['POST'])
def api_reload_strategies():
    """Hot-reload strategy configs into the running engine (no restart needed)"""
    global engine
    try:
        from strategies.strategy_config import load_configs
        configs = load_configs()
        if not engine or not hasattr(engine, 'strategy_manager'):
            return jsonify({"error": "Engine not running"}), 400
        # Reload each strategy
        for name, strategy in engine.strategy_manager.strategies.items():
            if hasattr(strategy, 'reload_config'):
                strategy.reload_config(configs)
        # Apply timeframes
        if hasattr(engine, '_apply_strategy_timeframes'):
            engine._apply_strategy_timeframes()
        return jsonify({
            "status": "ok",
            "active_strategy": engine.strategy_manager.active_strategy,
            "timeframes": engine.timeframes
        })
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/set', methods=['POST'])
def api_set_strategy():
    global engine
    data = request.get_json()
    strategy = data.get('strategy', 'scalp')
    if engine and hasattr(engine, 'set_active_strategy'):
        engine.set_active_strategy(strategy)
    elif engine and hasattr(engine, 'strategy_manager'):
        engine.strategy_manager.active_strategy = strategy
    return jsonify({"status": "ok", "strategy": strategy})


@app.route('/api/ml/search', methods=['POST'])
def api_ml_search():
    """Run genetic algorithm search to discover optimal strategy parameters"""
    global engine
    try:
        from strategies.ml_searcher import GeneticSearcher
        data = request.get_json() or {}
        config = {
            'population_size': data.get('population_size', 50),
            'generations': data.get('generations', 20),
            'min_profit_factor': data.get('min_profit_factor', 1.3),
            'min_trades': data.get('min_trades', 50),
        }
        # Get backtester from engine if available
        backtester = None
        if engine and hasattr(engine, 'backtester'):
            backtester = engine.backtester
        elif engine and engine.mt5:
            from backtester.backtester import Backtester
            backtester = Backtester(engine.mt5, config)
        searcher = GeneticSearcher(backtester=backtester, config=config)
        # Get all enabled symbols from engine
        symbols = []
        if engine and hasattr(engine, 'symbols') and engine.symbols:
            symbols = engine.symbols
        else:
            symbols = ["EURUSD"]
        # Get timeframes (use entry timeframe as default)
        timeframes = ["H1"]  # H1 is the most reliable for ML search
        if engine and hasattr(engine, 'timeframes'):
            entry_tf = engine.timeframes.get("entry", "M15")
            # Only use higher TFs for search (need more data per bar)
            tf_map = {"M1": "M15", "M5": "H1", "M15": "H1", "M30": "H1",
                      "H1": "H1", "H4": "H4", "D1": "D1"}
            timeframes = [tf_map.get(entry_tf, "H1")]
        # Run search
        days = data.get('days', 365)
        logger.info(f"ML search starting for symbols={symbols} timeframes={timeframes} days={days}")
        validated = searcher.run_search(symbols=symbols, timeframes=timeframes, days=days)
        # Return results
        best_pf = max((g.profit_factor for g in validated), default=0)
        return jsonify({
            "status": "ok",
            "validated_count": len(validated),
            "best_pf": round(best_pf, 2),
            "symbols_searched": symbols,
            "timeframes_used": timeframes,
            "data_source": "MT5 historical" if backtester else "synthetic",
            "genomes": [g.to_dict() for g in validated]
        })
    except Exception as e:
        logger.error(f"ML search failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/ml/genomes', methods=['GET'])
def api_ml_genomes():
    """Get list of validated genomes from DB"""
    try:
        import sqlite3
        import os
        from engine.constants import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, symbol, ema_fast, ema_slow, profit_factor, total_trades, "
            "win_rate, max_drawdown, sharpe, generation, validated_at "
            "FROM ml_genomes ORDER BY profit_factor DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        cols = ['id', 'symbol', 'ema_fast', 'ema_slow', 'profit_factor',
                'total_trades', 'win_rate', 'max_drawdown', 'sharpe',
                'generation', 'validated_at']
        return jsonify([dict(zip(cols, row)) for row in rows])
    except Exception as e:
        return jsonify([])


@app.route('/api/start', methods=['POST'])
def api_start():
    global engine
    if engine and not engine.running:
        thread = threading.Thread(target=engine.start)
        thread.daemon = True
        thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})


@app.route('/api/symbols/available', methods=['GET'])
def api_symbols_available():
    """Get list of all available symbols from MT5"""
    try:
        if engine and engine.mt5:
            symbols = engine.mt5.get_symbols_list()
            if symbols:
                return jsonify(symbols[:200])  # Limit to 200
        # Fallback: return common defaults
        return jsonify([
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'GBPCHF',
            'XAUUSD', 'XAGUSD', 'BRNUSD', 'USOUSD',
            'BTCUSD', 'ETHUSD',
            'US500', 'US100', 'US30', 'DE40', 'UK100'
        ])
    except Exception as e:
        logger.error(f"Failed to get available symbols: {e}")
        return jsonify([])


@app.route('/api/symbols', methods=['GET'])
def api_symbols():
    """Get all configured symbols with enabled status"""
    from engine.trading_engine import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, enabled FROM symbols ORDER BY symbol")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"symbol": r["symbol"], "enabled": bool(r["enabled"])} for r in rows])


@app.route('/api/symbols/selected', methods=['GET'])
def api_symbols_selected():
    """Get currently enabled symbols (simple list)"""
    from engine.trading_engine import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM symbols WHERE enabled = 1 ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(symbols)


@app.route('/api/symbols/select', methods=['POST'])
def api_symbols_select():
    """Update which symbols are enabled"""
    data = request.get_json()
    symbols = data.get('symbols', [])
    from engine.trading_engine import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE symbols SET enabled = 0")
    for symbol in symbols:
        cursor.execute("UPDATE symbols SET enabled = 1 WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()
    global engine
    if engine and hasattr(engine, 'symbols'):
        engine.symbols = symbols
    return jsonify({"status": "ok", "symbols": symbols})


@app.route('/api/symbols/add', methods=['POST'])
def api_symbols_add():
    """Add a new symbol to the watchlist"""
    from engine.trading_engine import get_db_connection
    data = request.get_json()
    symbol = data.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    # Insert or update - default enabled=1 (user wants to trade it)
    cursor.execute(
        "INSERT INTO symbols (symbol, enabled) VALUES (?, 1) "
        "ON CONFLICT(symbol) DO UPDATE SET enabled = 1",
        (symbol,)
    )
    conn.commit()
    conn.close()
    # Update engine symbols list
    global engine
    if engine and hasattr(engine, 'symbols') and symbol not in engine.symbols:
        engine.symbols.append(symbol)
    return jsonify({"status": "ok", "symbol": symbol})


@app.route('/api/symbols/remove', methods=['POST'])
def api_symbols_remove():
    """Remove a symbol from the watchlist (or disable it)"""
    from engine.trading_engine import get_db_connection
    data = request.get_json()
    symbol = data.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    # Just disable, don't delete (preserves history)
    cursor.execute("UPDATE symbols SET enabled = 0 WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()
    global engine
    if engine and hasattr(engine, 'symbols') and symbol in engine.symbols:
        engine.symbols.remove(symbol)
    return jsonify({"status": "ok", "symbol": symbol})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global engine
    if engine:
        engine.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Engine not initialized"})


@socketio.on('connect')
def on_connect():
    logger.info("Client connected")


@socketio.on('disconnect')
def on_disconnect():
    logger.info("Client disconnected")


def broadcast_status():
    global engine
    if engine:
        socketio.emit('status_update', {"status": engine.status})


def broadcast_decisions(decisions):
    socketio.emit('decisions_update', decisions)


def broadcast_positions(positions):
    socketio.emit('positions_update', positions)


def broadcast_history(history):
    socketio.emit('history_update', history)


def broadcast_analytics(analytics):
    socketio.emit('analytics_update', analytics)


def run_dashboard(trading_engine):
    global engine
    engine = trading_engine
    init_database()
    # Note: socketio.run() is called separately in main.py after engine is initialized


# Alias for main.py compatibility
init_dashboard = run_dashboard


if __name__ == '__main__':
    init_database()
    start_socketio()

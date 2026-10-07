"""
Dashboard - Web interface to monitor the trading bot with real-time WebSocket updates
"""
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import logging
import time
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mt5-trading-bot-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global engine reference
engine = None

# Database path
DB_PATH = "E:/projects/mt5-trading-bot/bot.db"
CONFIG_PATH = "E:/projects/mt5-trading-bot/config.yaml"


def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize SQLite database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create symbols table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            enabled INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create trades table
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
    
    # Create ai_decisions table
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
    
    # Initialize default symbols from config if DB is empty
    _init_default_symbols()


def _init_default_symbols():
    """Initialize symbols table - seeds with empty table if DB is new"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM symbols")
    count = cursor.fetchone()[0]

    # Only seed if table is completely empty (first run)
    # User will add symbols via dashboard
    if count == 0:
        # Seed with a few common symbols as examples - user can change these
        default_symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'BRNUSD']
        for symbol in default_symbols:
            cursor.execute(
                "INSERT OR IGNORE INTO symbols (symbol, enabled) VALUES (?, 0)",
                (symbol,)
            )
        conn.commit()

    conn.close()


def get_enabled_symbols():
    """Get list of enabled trading symbols from DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM symbols WHERE enabled = 1 ORDER BY symbol")
    symbols = [row['symbol'] for row in cursor.fetchall()]
    conn.close()
    return symbols


def set_enabled_symbols(symbols_list):
    """Update enabled symbols in DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Disable all first
    cursor.execute("UPDATE symbols SET enabled = 0")
    
    # Enable selected
    for symbol in symbols_list:
        cursor.execute(
            "INSERT OR REPLACE INTO symbols (symbol, enabled) VALUES (?, 1)",
            (symbol,)
        )
    
    conn.commit()
    conn.close()


def db_insert_trade(symbol, action, lot_size, entry_price):
    """Insert a new trade record into DB"""
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


def db_update_trade(trade_id, exit_price, pnl):
    """Update trade record when closed"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE trades 
        SET exit_price = ?, pnl = ?, status = 'CLOSED', closed_at = ?
        WHERE id = ?
    """, (exit_price, pnl, datetime.now().isoformat(), trade_id))
    conn.commit()
    conn.close()


def db_insert_ai_decision(symbol, action, lot_size, reasoning, confidence):
    """Insert AI decision into DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ai_decisions (symbol, action, lot_size, reasoning, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, action, lot_size, reasoning, confidence, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def db_get_trades(limit=50):
    """Get trade history from DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, action, lot_size, entry_price, exit_price, pnl, status, opened_at, closed_at
        FROM trades
        ORDER BY opened_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def db_get_ai_decisions(limit=10):
    """Get recent AI decisions from DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, action, lot_size, reasoning, confidence, created_at
        FROM ai_decisions
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def db_get_analytics():
    """Get analytics data from DB + live MT5 positions"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total closed trades from DB
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
    closed_trades = cursor.fetchone()[0]
    
    # Get open positions from MT5 directly
    open_positions = []
    if engine and engine.mt5:
        open_positions = engine.mt5.get_positions()
    
    open_trades = len(open_positions)
    
    # Win rate (only for closed trades)
    cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0 AND status = 'CLOSED'")
    wins = cursor.fetchone()[0]
    total_closed = closed_trades
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    
    # Total P&L from closed trades
    cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'CLOSED'")
    closed_pnl = cursor.fetchone()[0]
    
    # Add current unrealized P&L from MT5
    unrealized_pnl = sum(p.get('profit', 0) for p in open_positions)
    total_pnl = closed_pnl + unrealized_pnl
    
    conn.close()
    
    return {
        "total_trades": closed_trades + open_trades,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "closed_trades_count": closed_trades,
        "open_trades_count": open_trades
    }


def init_dashboard(trading_engine):
    """Initialize dashboard with trading engine"""
    global engine
    engine = trading_engine

# Initialize database at module load time (before engine needs it)
init_database()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/status')
def api_status():
    """Get engine status"""
    if engine is None:
        return jsonify({"error": "Engine not initialized", "status": "STOPPED"})
    try:
        status = engine.get_status()
        return jsonify(status)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "status": "ERROR", "trace": traceback.format_exc()[:500]})


@app.route('/api/market/<symbol>')
def api_market(symbol):
    """Get market data for symbol"""
    if engine and symbol in engine.market_data:
        return jsonify(engine.market_data[symbol])
    return jsonify({"error": "No data for symbol"})


@app.route('/api/positions')
def api_positions():
    """Get open positions - fresh from MT5"""
    if engine and engine.mt5:
        return jsonify(engine.mt5.get_positions())
    if engine:
        return jsonify(engine.positions)
    return jsonify([])


@app.route('/api/history')
def api_history():
    """Get trade history from DB"""
    trades = db_get_trades(20)
    return jsonify(trades)


@app.route('/api/decisions')
def api_decisions():
    """Get recent AI decisions from DB"""
    decisions = db_get_ai_decisions(10)
    return jsonify(decisions)


@app.route('/api/symbols', methods=['GET'])
def api_symbols():
    """Get all symbols in the DB (user-added symbols)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, enabled FROM symbols ORDER BY symbol")
    rows = cursor.fetchall()
    conn.close()
    # Return as dict with enabled status so frontend can check them
    symbols = [{'symbol': row['symbol'], 'enabled': row['enabled']} for row in rows]
    return jsonify(symbols)


@app.route('/api/symbols/selected', methods=['GET'])
def api_symbols_selected():
    """Get user's selected (enabled) symbols"""
    symbols = get_enabled_symbols()
    return jsonify(symbols)


@app.route('/api/symbols/select', methods=['POST'])
def api_symbols_select():
    """Save user's selected symbols"""
    data = request.get_json()
    if not data or 'symbols' not in data:
        return jsonify({"error": "Missing symbols array"})
    set_enabled_symbols(data['symbols'])
    return jsonify({"status": "saved", "symbols": data['symbols']})


@app.route('/api/symbols/available', methods=['GET'])
def api_symbols_available():
    """Get all available MT5 symbols for browsing"""
    if engine and engine.mt5:
        try:
            symbols = engine.mt5.get_symbols_list()
            return jsonify(symbols)
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return jsonify({"error": str(e)})
    return jsonify({"error": "MT5 not connected"})


@app.route('/api/symbols/add', methods=['POST'])
def api_symbols_add():
    """Add new symbol to DB"""
    data = request.get_json()
    if not data or 'symbol' not in data:
        return jsonify({"error": "Missing symbol"})
    symbol = data['symbol'].upper()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO symbols (symbol, enabled) VALUES (?, 1)", (symbol,))
    conn.commit()
    conn.close()
    return jsonify({"status": "added", "symbol": symbol})


@app.route('/api/trades', methods=['GET'])
def api_trades():
    """Get trade history from DB"""
    trades = db_get_trades(50)
    return jsonify(trades)


@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    """Get analytics data"""
    analytics = db_get_analytics()
    return jsonify(analytics)


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading"""
    if engine:
        if not engine.running:
            thread = threading.Thread(target=engine.start)
            thread.daemon = True
            thread.start()
            return jsonify({"status": "started"})
        return jsonify({"status": "already running"})
    return jsonify({"error": "Engine not initialized"})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    if engine:
        engine.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Engine not initialized"})


@socketio.on('connect')
def handle_connect():
    """Client connected"""
    logger.info("Client connected")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    logger.info("Client disconnected")


def broadcast_updates():
    """Background thread to broadcast updates to all clients"""
    import json

    last_status = {}
    last_decisions = []
    last_positions = []
    last_history = []
    last_analytics = {}

    def make_serializable(obj, _seen=None):
        """Convert datetime objects to strings for JSON serialization, with cycle detection"""
        if _seen is None:
            _seen = set()
        if id(obj) in _seen:
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            _seen.add(id(obj))
            return {k: make_serializable(v, _seen) for k, v in obj.items()}
        elif isinstance(obj, list):
            _seen.add(id(obj))
            return [make_serializable(i, _seen) for i in obj]
        return obj

    # Track last forced broadcast for periodic refresh
    last_forced_broadcast = time.time()
    last_positions_broadcast = time.time()

    while True:
        if engine:
            try:
                current_time = time.time()

                # Get current state
                status = engine.get_status()
                # Get fresh positions directly from MT5 (not cached)
                positions = engine.mt5.get_positions() if engine.mt5 else []
                
                # Get decisions and history from DB
                try:
                    decisions = db_get_ai_decisions(10)
                    history = db_get_trades(20)
                    analytics = db_get_analytics()
                except Exception:
                    decisions = []
                    history = []
                    analytics = {}

                # Broadcast status update
                market_hash = str(len(status.get('market_data', {})))
                status_hash = (str(status.get('status', '')) +
                              str(status.get('stats', {}).get('pnl_today', 0)) +
                              market_hash)

                # Force broadcast every 5 minutes
                force_broadcast = (current_time - last_forced_broadcast) >= 300

                if status_hash != last_status.get('status_hash', '') or force_broadcast:
                    socketio.emit('status_update', make_serializable(status))
                    last_status['status_hash'] = status_hash
                    last_forced_broadcast = current_time

                # Broadcast AI decisions
                decisions_hash = str([d.get('action', '') for d in decisions])
                if decisions_hash != last_decisions:
                    socketio.emit('decisions_update', make_serializable(decisions))
                    last_decisions = decisions_hash

                # Broadcast positions every 5 seconds (prices change constantly)
                positions_broadcast = (current_time - last_positions_broadcast) >= 5
                if positions_broadcast:
                    socketio.emit('positions_update', make_serializable(positions))
                    last_positions_broadcast = current_time
                    logger.info(f"Broadcast positions: {len(positions)} items, first profit={positions[0].get('profit') if positions else 'N/A'}")

                # Broadcast history
                history_hash = str([t.get('id', '') for t in history])
                if history_hash != last_history:
                    socketio.emit('history_update', make_serializable(history))
                    last_history = history_hash
                    
                # Broadcast analytics
                analytics_hash = str(analytics)
                if analytics_hash != last_analytics:
                    socketio.emit('analytics_update', make_serializable(analytics))
                    last_analytics = analytics_hash

            except Exception as e:
                logger.error(f"Broadcast error: {e}")

        socketio.sleep(1)


# Start broadcast thread
threading.Thread(target=broadcast_updates, daemon=True).start()


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MT5 Trading Bot</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', Arial, sans-serif;
            background: #4a5d6e;
            color: #eee;
            padding: 20px;
        }
        h1, h2 { font-family: 'Poppins', Arial, sans-serif; color: #00d4ff; margin-bottom: 20px; }
        h2 { font-size: 16px; margin: 15px 0 10px; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #0f3460;
        }

        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status.ready { background: #00ff88; color: #000; }
        .status.running { background: #00d4ff; color: #000; }
        .status.error { background: #ff4444; color: #fff; }
        .status.stopped { background: #666; color: #fff; }

        table { width: 100%; border-collapse: collapse; }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }
        th { color: #00d4ff; font-size: 12px; }

        .symbol {
            font-weight: bold;
            color: #00d4ff;
        }
        .buy { color: #00ff88; }
        .sell { color: #ff4444; }
        .profit { color: #00ff88; }
        .loss { color: #ff4444; }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
        }
        .btn-start { background: #00ff88; color: #000; }
        .btn-stop { background: #ff4444; color: #fff; }
        .btn-primary { background: #00d4ff; color: #000; }

        .stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .stat {
            background: #0f3460;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 24px; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 12px; color: #888; }

        .market-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #0f3460;
        }
        .market-row:last-child { border-bottom: none; }

        .ai-decision {
            background: #0f3460;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-size: 13px;
        }

        .confidence {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        }
        .confidence.high { background: #00ff88; color: #000; }
        .confidence.medium { background: #ffaa00; color: #000; }
        .confidence.low { background: #ff4444; color: #fff; }

        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal.show {
            display: flex;
        }
        .modal-content {
            background: #0f3460;
            padding: 30px;
            border-radius: 10px;
            max-width: 500px;
            width: 90%;
            text-align: center;
        }
        .modal-content h3 {
            margin-bottom: 15px;
            color: #00ff88;
        }
        .modal-content p {
            margin-bottom: 20px;
            color: #eee;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        .symbol-selector {
            margin-bottom: 15px;
        }
        .symbol-selector select {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #0f3460;
            color: #eee;
            font-family: 'Inter', Arial, sans-serif;
            font-size: 14px;
        }
        .symbol-selector select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .add-symbol-form {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .add-symbol-form input {
            flex: 1;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #0f3460;
            color: #eee;
            font-family: 'Inter', Arial, sans-serif;
        }
        .symbol-item {
            display: flex;
            align-items: center;
            padding: 8px;
            border-bottom: 1px solid #0f3460;
        }
        .symbol-item:last-child { border-bottom: none; }
        .symbol-item input { margin-right: 10px; }
        .symbol-item label { flex: 1; cursor: pointer; }
        .symbol-item.enabled label { color: #00ff88; font-weight: bold; }
    </style>
</head>
<body>
    <h1><span class="live-indicator"></span>MT5 Trading Bot</h1>

    <div class="grid">
        <div class="card">
            <h2>Symbol Selection</h2>
            <div class="symbol-selector">
                <div id="symbol-list">Loading symbols...</div>
                <div class="add-symbol-form">
                    <input type="text" id="add-symbol-input" placeholder="Enter symbol (e.g. XAUUSD)">
                    <button class="btn btn-primary" onclick="addNewSymbol()">Add</button>
                    <button class="btn" onclick="browseMT5Symbols()">Browse MT5</button>
                </div>
                <div style="margin-top: 15px;">
                    <button class="btn btn-primary" onclick="saveSelectedSymbols()">Save Selection</button>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Engine Status</h2>
            <p>
                Status: <span id="engine-status" class="status">Loading...</span>
            </p>
            <p style="margin-top: 10px;">
                <button class="btn btn-start" onclick="startEngine()">Start</button>
                <button class="btn btn-stop" onclick="stopEngine()">Stop</button>
            </p>
        </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
        <h2>Analytics</h2>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="analytics-total-trades">0</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="analytics-win-rate">0%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="analytics-pnl">$0</div>
                <div class="stat-label">Total P&L</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="analytics-open">0</div>
                <div class="stat-label">Open Positions</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="analytics-closed">0</div>
                <div class="stat-label">Closed Trades</div>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Market Analysis</h2>
            <div id="market-data">Loading...</div>
        </div>

        <div class="card">
            <h2>Open Positions</h2>
            <div id="positions-data">No positions</div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>AI Decisions</h2>
            <div id="ai-decisions">Waiting for decisions...</div>
        </div>

        <div class="card">
            <h2>Trade History</h2>
            <div id="trade-history">No trades yet</div>
        </div>
    </div>

    <script>
        const socket = io();
        let selectedSymbols = [];

        socket.on('connect', () => {
            console.log('Connected to server');
            // Request initial data on connect
            fetch('/api/status').then(r => r.json()).then(data => updateStatus(data));
            fetch('/api/positions').then(r => r.json()).then(data => updatePositions(data));
            fetch('/api/analytics').then(r => r.json()).then(data => updateAnalytics(data));
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        socket.on('status_update', (data) => {
            updateStatus(data);
        });

        socket.on('decisions_update', (decisions) => {
            updateDecisions(decisions);
        });

        socket.on('positions_update', (positions) => {
            console.log('Received positions_update:', positions);
            updatePositions(positions);
        });

        socket.on('history_update', (history) => {
            updateHistory(history);
        });

        socket.on('analytics_update', (analytics) => {
            updateAnalytics(analytics);
        });

        async function loadSymbols() {
            try {
                const response = await fetch('/api/symbols');
                const list = document.getElementById('symbol-list');
                if (response.ok) {
                    const data = await response.json();
                    if (Array.isArray(data) && data.length > 0) {
                        let html = '';
                        data.forEach(item => {
                            const checked = item.enabled ? 'checked' : '';
                            const enabledClass = item.enabled ? 'enabled' : '';
                            html += '<div class="symbol-item ' + enabledClass + '">' +
                                '<input type="checkbox" id="sym_' + item.symbol + '" value="' + item.symbol + '" ' + checked + '>' +
                                '<label for="sym_' + item.symbol + '">' + item.symbol + (item.enabled ? ' (enabled)' : '') + '</label>' +
                                '</div>';
                        });
                        list.innerHTML = html;
                    } else {
                        list.innerHTML = '<div style="padding:10px;color:#888;">No symbols added yet. Add symbols below.</div>';
                    }
                } else {
                    list.innerHTML = '<div style="padding:10px;color:#f44;">Error: HTTP ' + response.status + '</div>';
                }
            } catch (e) {
                console.error('Failed to load symbols:', e);
                document.getElementById('symbol-list').innerHTML = '<div style="padding:10px;color:#f44;">Connection error: ' + e.message + '</div>';
            }
        }

        // Auto-reload after 3 seconds if still showing "Loading..."
        setTimeout(() => {
            const el = document.getElementById('symbol-list');
            if (el && el.textContent === 'Loading symbols...') {
                loadSymbols();
            }
        }, 3000);

        async function addNewSymbol() {
            const input = document.getElementById('add-symbol-input');
            const symbol = input.value.trim().toUpperCase();
            if (!symbol) return;

            try {
                const response = await fetch('/api/symbols/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: symbol })
                });
                if (response.ok) {
                    input.value = '';
                    loadSymbols();
                    document.getElementById('modal-message').textContent = symbol + ' added';
                    document.getElementById('symbolModal').classList.add('show');
                }
            } catch (e) {
                console.error('Failed to add symbol:', e);
            }
        }

        async function browseMT5Symbols() {
            try {
                const response = await fetch('/api/symbols/available');
                const list = document.getElementById('symbol-list');
                if (response.ok) {
                    const symbols = await response.json();
                    if (Array.isArray(symbols)) {
                        let html = '<div style="max-height:300px;overflow-y:auto;" id="mt5-symbols-list">';
                        symbols.slice(0, 100).forEach(s => {
                            html += '<div class="symbol-item" data-symbol="' + s + '" style="cursor:pointer;">' +
                                '<label>' + s + ' (click to add)</label>' +
                                '</div>';
                        });
                        html += '</div>';
                        list.innerHTML = html;
                    }
                }
            } catch (e) {
                console.error('Failed to browse MT5 symbols:', e);
            }
        }

        // Event delegation for dynamically added MT5 symbols
        document.addEventListener('click', function(e) {
            const item = e.target.closest('.symbol-item[data-symbol]');
            if (item) {
                const symbol = item.getAttribute('data-symbol');
                window.addMT5Symbol(symbol);
            }
        });

        window.addMT5Symbol = function(symbol) {
            fetch('/api/symbols/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol })
            }).then(() => loadSymbols());
        };

        async function saveSelectedSymbols() {
            const checkboxes = document.querySelectorAll('#symbol-list input[type="checkbox"]');
            const selected = Array.from(checkboxes).filter(cb => cb.checked).map(cb => cb.value);
            if (selected.length === 0) {
                document.getElementById('modal-message').textContent = 'Please select at least one symbol';
                document.getElementById('symbolModal').classList.add('show');
                return;
            }
            try {
                const response = await fetch('/api/symbols/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbols: selected })
                });
                if (response.ok) {
                    document.getElementById('modal-message').textContent = 'Symbols saved for trading';
                    document.getElementById('symbolModal').classList.add('show');
                    loadSymbols();
                }
            } catch (e) {
                console.error('Failed to save symbols:', e);
            }
        }

        function closeModal() {
            document.getElementById('symbolModal').classList.remove('show');
        }

        document.getElementById('symbolModal').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });

        function updateAnalytics(analytics) {
            if (!analytics) return;
            document.getElementById('analytics-total-trades').textContent = analytics.total_trades || 0;
            document.getElementById('analytics-win-rate').textContent = (analytics.win_rate || 0) + '%';
            const pnl = analytics.total_pnl || 0;
            document.getElementById('analytics-pnl').textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
            document.getElementById('analytics-pnl').className = pnl >= 0 ? 'stat-value' : 'stat-value loss';
            document.getElementById('analytics-open').textContent = analytics.open_trades_count || 0;
            document.getElementById('analytics-closed').textContent = analytics.closed_trades_count || 0;
        }

        function updateStatus(data) {
            console.log('updateStatus called with:', data.status);
            const statusEl = document.getElementById('engine-status');
            statusEl.textContent = data.status || 'UNKNOWN';
            statusEl.className = 'status ' + (data.status || 'stopped').toLowerCase();

            if (data.market_data) {
                let marketHtml = '';
                for (const [symbol, info] of Object.entries(data.market_data)) {
                    if (selectedSymbols.length === 0 || selectedSymbols.includes(symbol)) {
                        if (info) {
                            const m5Data = info.M5 || {};
                            const h4Data = info.H4 || {};
                            const trend = m5Data.overall || h4Data.overall || 'N/A';
                            const trendDir = info.trend_direction || 'SIDEWAYS';
                            const trendClass = trend.includes('BULL') ? 'buy' : trend.includes('BEAR') ? 'sell' : '';
                            const bid = m5Data.bid || h4Data.bid || 'N/A';
                            const rsi = m5Data.rsi || h4Data.rsi || 'N/A';
                            const spread = m5Data.spread || 'N/A';
                            const hasPos = info.has_position ? ' <span style="color:#ffaa00;">(Position)</span>' : '';
                            marketHtml += '<div class="market-row">' +
                                '<div><span class="symbol">' + symbol + '</span> <span class="' + trendClass + '">' + trend + '</span> ' +
                                '<span style="font-size:10px;color:#888;">H4:' + trendDir + '</span>' + hasPos + '</div>' +
                                '<div style="text-align:right;"><div>' + (typeof bid === 'number' ? bid.toFixed(5) : bid) + '</div>' +
                                '<div style="font-size:11px;color:#888;">RSI: ' + (typeof rsi === 'number' ? rsi.toFixed(1) : rsi) + ' | Spread: ' + spread + ' pips</div></div>' +
                                '</div>';
                        }
                    }
                }
                document.getElementById('market-data').innerHTML = marketHtml || 'No data for selected symbols';
            }
        }

        function updateDecisions(decisions) {
            let html = '';
            if (decisions && decisions.length > 0) {
                decisions.slice(-5).reverse().forEach(d => {
                    const action = d.action || 'HOLD';
                    const conf = d.confidence || 50;
                    const confClass = conf >= 70 ? 'high' : conf >= 50 ? 'medium' : 'low';
                    const actionClass = action === 'BUY' ? 'buy' : action === 'SELL' ? 'sell' : '';
                    const time = new Date(d.created_at || Date.now()).toLocaleTimeString();
                    // Clean up reasoning - remove JSON formatting if present
                    let reasoning = d.reasoning || '';
                    if (reasoning.includes('"entry_reason"')) {
                        const match = reasoning.match(/"entry_reason":\s*"([^"]+)"/);
                        reasoning = match ? match[1] : reasoning.substring(0, 100);
                    }
                    if (reasoning.length > 150) reasoning = reasoning.substring(0, 150) + '...';
                    html += '<div class="ai-decision"><strong>' + (d.symbol || 'N/A') + '</strong>: ' +
                        '<span class="' + actionClass + '">' + action + '</span> ' +
                        '<span class="confidence ' + confClass + '">' + conf + '%</span> ' +
                        '<span style="font-size:10px;color:#888;">' + time + '</span><br>' +
                        '<small>' + reasoning + '</small></div>';
                });
            }
            document.getElementById('ai-decisions').innerHTML = html || 'No decisions yet';
        }

        function updatePositions(positions) {
            let posHtml = '';
            if (positions && positions.length > 0) {
                posHtml = '<table><tr><th>Symbol</th><th>Type</th><th>Lot</th><th>Entry</th><th>Current</th><th>P&L</th></tr>';
                positions.forEach(p => {
                    const pnlClass = p.profit >= 0 ? 'profit' : 'loss';
                    const typeClass = p.type === 'BUY' ? 'buy' : 'sell';
                    const currentPrice = p.price_current ? p.price_current.toFixed(5) : 'N/A';
                    posHtml += '<tr><td class="symbol">' + p.symbol + '</td><td class="' + typeClass + '">' + p.type + '</td>' +
                        '<td>' + p.volume + '</td><td>' + p.price_open.toFixed(5) + '</td><td>' + currentPrice + '</td>' +
                        '<td class="' + pnlClass + '">' + (p.profit >= 0 ? '+' : '') + p.profit.toFixed(2) + '</td></tr>';
                });
                posHtml += '</table>';
            }
            document.getElementById('positions-data').innerHTML = posHtml || 'No positions';
        }

        function updateHistory(history) {
            let html = '';
            if (history && history.length > 0) {
                html = '<table><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Lot</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Status</th></tr>';
                history.slice(-10).reverse().forEach(t => {
                    const pnlClass = (t.pnl || 0) >= 0 ? 'profit' : 'loss';
                    const time = new Date(t.opened_at || Date.now()).toLocaleString();
                    html += '<tr><td>' + time + '</td>' +
                        '<td class="symbol">' + t.symbol + '</td>' +
                        '<td class="' + (t.action === 'BUY' ? 'buy' : 'sell') + '">' + t.action + '</td>' +
                        '<td>' + t.lot_size + '</td>' +
                        '<td>' + (typeof t.entry_price === 'number' ? t.entry_price.toFixed(5) : t.entry_price) + '</td>' +
                        '<td>' + (t.exit_price ? (typeof t.exit_price === 'number' ? t.exit_price.toFixed(5) : t.exit_price) : '-') + '</td>' +
                        '<td class="' + pnlClass + '">' + ((t.pnl || 0) >= 0 ? '+' : '') + (t.pnl || 0).toFixed(2) + '</td>' +
                        '<td>' + t.status + '</td></tr>';
                });
                html += '</table>';
            }
            document.getElementById('trade-history').innerHTML = html || 'No trades yet';
        }

        function startEngine() {
            fetch('/api/start', { method: 'POST' }).then(r => r.json()).then(d => alert(d.status || d.error));
        }

        function stopEngine() {
            fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(d => alert(d.status || d.error));
        }

        // Wait for socket connection then load data
        setTimeout(() => {
            loadSymbols();
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    console.log('Initial status loaded:', data.status);
                    updateStatus(data);
                })
                .catch(e => console.error('Failed to load status:', e));
            fetch('/api/positions')
                .then(r => r.json())
                .then(data => {
                    updatePositions(data);
                })
                .catch(e => console.error('Failed to load positions:', e));
            fetch('/api/analytics')
                .then(r => r.json())
                .then(data => {
                    updateAnalytics(data);
                })
                .catch(e => console.error('Failed to load analytics:', e));
        }, 500);
    </script>

    <!-- Modal for symbol selection confirmation -->
    <div id="symbolModal" class="modal">
        <div class="modal-content">
            <h3>Symbols Updated</h3>
            <p id="modal-message">Symbols have been added to trading.</p>
            <div class="modal-buttons">
                <button class="btn btn-primary" onclick="closeModal()">OK</button>
            </div>
        </div>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    # Initialize database
    init_database()
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)

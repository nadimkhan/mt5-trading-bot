# MT5 Trading Bot

An autonomous AI-powered trading bot that connects to MetaTrader 5, analyzes markets using cloud AI (Claude, OpenAI, MiniMax, Groq), and executes trades automatically.

## Features

- 🤖 **AI-Powered Decisions** - Uses cloud AI to analyze markets and make trading decisions
- 📊 **Technical Analysis** - Built-in RSI, MACD, Bollinger Bands, ATR indicators
- 🔄 **Multiple AI Providers** - Support for Claude, OpenAI, MiniMax, and Groq
- 💻 **Live Dashboard** - Web interface showing real-time market data, positions, and AI decisions
- 📈 **Risk Management** - Configurable daily loss limits, drawdown protection
- 📜 **Trade Logging** - Complete history of all trades with AI reasoning

## Requirements

- Windows PC with MetaTrader 5 installed
- Python 3.8+ (Python 3.14 recommended)
- MT5 Demo or Live account
- AI API key (optional - bot can run with indicators only)

## Quick Start

### 1. Install Python Dependencies

```bash
cd E:/projects/mt5-trading-bot
pip install -r requirements.txt
```

### 2. Configure

Copy the example config and edit with your settings:

```bash
copy config.yaml.example config.yaml
```

Edit `config.yaml` with your settings:

```yaml
# AI Provider (get from your provider)
ai:
  provider: "claude"  # claude, openai, minimax, or groq
  api_key: "your-api-key-here"

# MT5 Connection
mt5:
  login: 57452518479      # Your MT5 login
  password: ""            # Your password
  server: "GTCGlobalTrade-Demo"  # Your server

# Trading Settings
trading:
  symbols:
    - "EURUSD"
    - "GBPUSD"
    - "XAUUSD"
    - "USDJPY"
  timeframe: "H1"
  max_positions: 5
```

### 3. Test Connection

```bash
python main.py --test
```

This will verify MT5 connection and show current prices.

### 4. Run the Bot

**With Dashboard:**
```bash
python main.py
```
Then open http://127.0.0.1:5000 in your browser.

**Headless (Console Only):**
```bash
python main.py --headless
```

## Dashboard

The dashboard shows:
- Engine status (Running/Stopped/Error)
- Today's trade statistics
- Real-time market analysis for each symbol
- Open positions with live P&L
- Recent AI decisions with reasoning
- Complete trade history

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      TRADING LOOP                            │
│                                                              │
│  1. GET MARKET DATA ──► MT5 (prices, OHLCV)               │
│                                                              │
│  2. CALCULATE INDICATORS ──► RSI, MACD, Bollinger, ATR    │
│                                                              │
│  3. AI ANALYSIS ──► Send to Claude/OpenAI/etc.             │
│     └─► "Should I buy or sell EURUSD?"                     │
│                                                              │
│  4. EXECUTE DECISION ──► MT5 (place order, SL, TP)        │
│                                                              │
│  5. LOG RESULT ──► Trade history + AI reasoning            │
│                                                              │
│  6. REPEAT ──► Every 60 seconds (configurable)             │
└─────────────────────────────────────────────────────────────┘
```

## AI Decision Process

The AI receives:
- Current prices and spreads
- Technical indicators (RSI, MACD, Bollinger, ATR)
- Support/Resistance levels
- Trend direction
- Open positions
- Recent trade history

The AI outputs:
- Action: BUY / SELL / HOLD
- Lot size (based on risk rules)
- Stop loss and take profit levels
- Confidence score
- Reasoning for the decision

## Risk Management

Default settings:
- Max 2% risk per trade
- Max 5% daily loss limit
- Max 10% drawdown limit
- Max 5 open positions
- No trades if spread > 15 pips

## Project Structure

```
mt5-trading-bot/
├── main.py              # Entry point
├── config.yaml          # Your settings
├── config.yaml.example  # Template
├── requirements.txt     # Dependencies
├── engine/
│   ├── mt5_connector.py    # MT5 connection
│   ├── trading_engine.py   # Main trading logic
│   └── indicators.py       # Technical indicators
├── ai/
│   └── ai_analyzer.py      # AI provider integration
├── dashboard/
│   └── app.py              # Web dashboard
└── logs/                   # Trade logs
```

## Trading Strategies

The bot uses these strategies (AI decides which to use):

| Strategy | Description | Best For |
|----------|-------------|----------|
| SCALP | Quick trades, small targets | Low volatility |
| PULLBACK | Buy/sell at support/resistance | Trending markets |
| TREND_FOLLOW | Ride the trend | Strong trends |
| BREAKOUT | Trade range breakouts | Volatile markets |
| REVERSAL | Catch reversals | Overbought/Oversold |

## Trade Logging

All trades are saved to `logs/trades_YYYYMMDD.json`:

```json
{
  "ticket": 12345678,
  "symbol": "EURUSD",
  "type": "BUY",
  "volume": 0.05,
  "entry_price": 1.12150,
  "exit_price": 1.12400,
  "pnl": 125.00,
  "strategy": "PULLBACK",
  "ai_confidence": 78,
  "ai_reasoning": "RSI oversold, price near support, bullish MACD cross",
  "market_analysis": {
    "rsi": 32,
    "trend": "BULL_TREND",
    "spread": 10
  }
}
```

## Troubleshooting

**MT5 Connection Failed:**
- Make sure MT5 terminal is running
- Check login credentials in config.yaml
- Verify server name matches MT5

**AI Not Making Decisions:**
- Check api_key is set in config.yaml
- Verify API key is valid
- Check internet connection

**No Trades Executed:**
- Check if AI is configured (api_key required)
- Verify risk limits haven't been reached
- Check if spread is too wide (>15 pips)

## Safety Features

⚠️ **IMPORTANT**: This is a demo bot with paper trading. Always:
- Use a DEMO account first
- Review AI decisions before going live
- Set appropriate risk limits
- Monitor the bot regularly

## License

MIT - Use at your own risk. Trading involves substantial risk of loss.

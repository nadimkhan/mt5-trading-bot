"""
MT5 Trading Bot - Main Entry Point
"""
import argparse
import logging
import yaml
import sys
from pathlib import Path

from engine.trading_engine import TradingEngine
from dashboard.app import app, init_dashboard, socketio


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/bot.log', mode='a')
        ]
    )


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    path = Path(config_path)
    
    if not path.exists():
        # Try example file
        example_path = Path("config.yaml.example")
        if example_path.exists():
            print(f"⚠️  {config_path} not found. Copying from example...")
            import shutil
            shutil.copy(example_path, path)
            print(f"📝 Created {config_path} - please edit it with your settings")
            return None
        else:
            print(f"❌ Config file not found: {config_path}")
            return None
            
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MT5 Trading Bot")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Start with web dashboard"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without dashboard (console only)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test MT5 connection without trading"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config(args.config)
    if config is None:
        sys.exit(1)
        
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize engine
    engine = TradingEngine(config)
    
    if args.test:
        # Just test MT5 connection
        print("🔌 Testing MT5 connection...")
        if engine.initialize():
            print("✅ MT5 Connected successfully!")
            status = engine.get_status()
            print(f"\n📊 Account Info:")
            if engine.mt5:
                account = engine.mt5.get_account_info()
                if account:
                    print(f"   Login: {account['login']}")
                    print(f"   Balance: ${account['balance']:.2f}")
                    print(f"   Equity: ${account['equity']:.2f}")
                    print(f"   Margin: ${account['margin']:.2f}")
                    print(f"   Free Margin: ${account['free_margin']:.2f}")
                    
            print(f"\n📈 Market Data:")
            for symbol in config.get("trading", {}).get("symbols", ["EURUSD"]):
                price = engine.mt5.get_current_price(symbol)
                if price:
                    spread = round((price['ask'] - price['bid']) * 10000, 1)
                    print(f"   {symbol}: Bid {price['bid']:.5f} | Ask {price['ask']:.5f} | Spread {spread} pips")
                    
            print(f"\n💼 Open Positions: {len(engine.positions)}")
            engine.stop()
            print("\n✅ Test complete!")
        else:
            print("❌ MT5 connection failed")
            sys.exit(1)
        return
        
    if args.headless:
        # Run headless (console only)
        logger.info("Starting MT5 Trading Bot (Headless Mode)")

        if not engine.initialize():
            logger.error("Failed to initialize engine")
            sys.exit(1)

        try:
            engine.start()
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
            engine.stop()

    else:
        # Run with dashboard
        logger.info("Dashboard mode started - Dashboard at http://127.0.0.1:5000")

        # Initialize dashboard FIRST (creates DB tables)
        init_dashboard(engine)

        # THEN initialize engine (will find DB tables ready)
        if not engine.initialize():
            logger.error("Failed to initialize engine")
            sys.exit(1)
        
        print("""
======================================================================
           MT5 Trading Bot Dashboard
======================================================================
  Dashboard: http://127.0.0.1:5000
  Press Ctrl+C to stop
======================================================================
        """)
        
        # Start engine in background thread
        import threading
        engine_thread = threading.Thread(target=engine.start)
        engine_thread.daemon = True
        engine_thread.start()
        
        try:
            # Run Flask dashboard
            dashboard_config = config.get("dashboard", {})
            host = dashboard_config.get("host", "127.0.0.1")
            port = dashboard_config.get("port", 5000)
            socketio.run(app, host=host, port=port, debug=False)
        except KeyboardInterrupt:
            logger.info("⏹️  Shutting down...")
            engine.stop()
            logger.info("👋 Goodbye!")


if __name__ == "__main__":
    main()

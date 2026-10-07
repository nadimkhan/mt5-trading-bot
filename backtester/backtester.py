"""
Backtester - Strategy validation using MT5 historical data
- Uses MT5 historical data with realistic spread, commission, slippage
- Calculates profit factor, win rate, Sharpe ratio
- Walk-forward testing on unseen data
- Rejects setups with profit factor < 1.3 or < 100 trades
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies using MT5 historical data"""
    
    def __init__(self, mt5_connector, config: dict = None):
        self.mt5 = mt5_connector
        self.config = config or {}
        
        # Backtest settings
        bt_config = self.config.get("backtest", {})
        self.initial_balance = bt_config.get("initial_balance", 100000)
        self.spread_multiplier = bt_config.get("spread_multiplier", 1.2)  # Add 20% for realistic spread
        self.slippage_pips = bt_config.get("slippage_pips", 0.5)
        self.commission_per_lot = bt_config.get("commission_per_lot", 7.0)  # $7 per lot typical
        
        # Acceptance criteria
        self.min_profit_factor = bt_config.get("min_profit_factor", 1.3)
        self.min_trades = bt_config.get("min_trades", 100)
        self.max_drawdown_percent = bt_config.get("max_drawdown_percent", 20)
        
    def download_historical_data(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> List[dict]:
        """Download historical data from MT5 for backtesting"""
        try:
            # Ensure MT5 is initialized
            if not mt5.initialize():
                logger.error("MT5 not initialized")
                return []
            
            # Convert to MT5 date format
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_ts, end_ts)
            if rates is None or len(rates) == 0:
                logger.error(f"No historical data for {symbol} {timeframe}")
                return []
            
            bars = []
            for rate in rates:
                bars.append({
                    "time": datetime.fromtimestamp(rate['time']),
                    "open": rate['open'],
                    "high": rate['high'],
                    "low": rate['low'],
                    "close": rate['close'],
                    "tick_volume": rate['tick_volume'],
                    "spread": rate['spread'],
                    "real_volume": rate['real_volume']
                })
            
            logger.info(f"Downloaded {len(bars)} bars for {symbol} {timeframe}")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to download historical data: {e}")
            return []
    
    def calculate_indicators(self, bars: List[dict]) -> List[dict]:
        """Calculate indicators for backtesting"""
        # This is a simplified version - in production would use proper indicator library
        closes = [b['close'] for b in bars]
        highs = [b['high'] for b in bars]
        lows = [b['low'] for b in bars]
        
        # Add indicators to each bar
        for i, bar in enumerate(bars):
            bar['ema_9'] = self._calculate_ema(closes[:i+1], 9)
            bar['ema_21'] = self._calculate_ema(closes[:i+1], 21)
            bar['ema_50'] = self._calculate_ema(closes[:i+1], 50)
            bar['atr'] = self._calculate_atr(highs[:i+1], lows[:i+1], closes[:i+1])
            bar['rsi'] = self._calculate_rsi(closes[:i+1])
        
        return bars
    
    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _calculate_atr(self, highs, lows, closes, period=14) -> float:
        """Calculate ATR"""
        if len(highs) < period + 1:
            return 0
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return sum(trs[-period:]) / period if trs else 0
    
    def _calculate_rsi(self, prices, period=14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def run_backtest(self, symbol: str, strategy_func, 
                     start_date: datetime, end_date: datetime,
                     timeframe: str = "H1") -> Dict:
        """
        Run backtest for a strategy function.
        
        strategy_func(bars, index, position) -> 'BUY', 'SELL', 'HOLD'
        """
        logger.info(f"Starting backtest: {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Download data
        bars = self.download_historical_data(symbol, timeframe, start_date, end_date)
        if len(bars) < 100:
            return {"error": "Insufficient data", "trades": []}
        
        # Calculate indicators
        bars = self.calculate_indicators(bars)
        
        # Initialize backtest state
        balance = self.initial_balance
        equity = balance
        positions = []  # Current open positions
        closed_trades = []  # Historical trades
        equity_curve = []  # For drawdown calculation
        
        # Track state
        current_position = None
        entry_price = 0
        entry_time = None
        
        for i, bar in enumerate(bars):
            current_price = bar['close']
            atr = bar.get('atr', 0)
            
            # Update equity curve
            position_pnl = 0
            if current_position:
                if current_position['type'] == 'BUY':
                    position_pnl = (current_price - entry_price) * current_position['volume'] * 100000
                else:
                    position_pnl = (entry_price - current_price) * current_position['volume'] * 100000
            equity = balance + position_pnl
            equity_curve.append({"time": bar['time'], "equity": equity})
            
            # Get signal from strategy
            signal = strategy_func(bars, i, current_position)
            
            # Apply signal
            if signal == 'BUY' and current_position is None:
                # Open long
                sl_pips = atr * 3 if atr else 30
                tp_pips = atr * 5 if atr else 50
                
                current_position = {
                    "type": "BUY",
                    "entry_price": current_price,
                    "sl_price": current_price - (sl_pips * 0.0001),
                    "tp_price": current_price + (tp_pips * 0.0001),
                    "volume": self._calculate_lot_size(balance, sl_pips),
                    "entry_time": bar['time'],
                    "spread": bar.get('spread', 0),
                    "atr": atr
                }
                
            elif signal == 'SELL' and current_position is None:
                # Open short
                sl_pips = atr * 3 if atr else 30
                tp_pips = atr * 5 if atr else 50
                
                current_position = {
                    "type": "SELL",
                    "entry_price": current_price,
                    "sl_price": current_price + (sl_pips * 0.0001),
                    "tp_price": current_price - (tp_pips * 0.0001),
                    "volume": self._calculate_lot_size(balance, sl_pips),
                    "entry_time": bar['time'],
                    "spread": bar.get('spread', 0),
                    "atr": atr
                }
            
            elif current_position:
                # Check SL/TP
                hit_sl = False
                hit_tp = False
                
                if current_position['type'] == 'BUY':
                    if current_price <= current_position['sl_price']:
                        hit_sl = True
                    elif current_price >= current_position['tp_price']:
                        hit_tp = True
                else:
                    if current_price >= current_position['sl_price']:
                        hit_sl = True
                    elif current_price <= current_position['tp_price']:
                        hit_tp = True
                
                if hit_sl or hit_tp:
                    # Close position
                    if current_position['type'] == 'BUY':
                        pnl = (current_price - entry_price) * current_position['volume'] * 100000
                    else:
                        pnl = (entry_price - current_price) * current_position['volume'] * 100000
                    
                    # Apply costs
                    commission = self.commission_per_lot * current_position['volume']
                    spread_cost = current_position['spread'] * 0.0001 * current_position['volume'] * 100000 * self.spread_multiplier
                    slippage_cost = self.slippage_pips * 0.0001 * current_position['volume'] * 100000
                    
                    net_pnl = pnl - commission - spread_cost - slippage_cost
                    balance += net_pnl
                    
                    closed_trades.append({
                        "symbol": symbol,
                        "type": current_position['type'],
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "exit_time": bar['time'],
                        "pnl": net_pnl,
                        "gross_pnl": pnl,
                        "commission": commission,
                        "spread_cost": spread_cost,
                        "slippage_cost": slippage_cost,
                        "volume": current_position['volume'],
                        "atr_at_entry": current_position['atr'],
                        "duration_bars": i - (bar['time'] - current_position['entry_time']).total_seconds() / 3600
                    })
                    
                    current_position = None
        
        # Calculate metrics
        metrics = self._calculate_metrics(closed_trades, equity_curve, start_date, end_date)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": balance,
            "metrics": metrics,
            "trades": closed_trades,
            "equity_curve": equity_curve,
            "status": "PASS" if metrics['passed'] else "FAIL"
        }
    
    def _calculate_lot_size(self, balance: float, sl_pips: float) -> float:
        """Calculate lot size based on risk"""
        risk_amount = balance * 0.01  # 1% risk
        pip_value = 10  # $10 per pip per standard lot
        lot = risk_amount / (sl_pips * pip_value)
        return round(min(max(lot, 0.01), 1.0), 2)
    
    def _calculate_metrics(self, trades: List[dict], equity_curve: List[dict],
                          start_date: datetime, end_date: datetime) -> Dict:
        """Calculate backtest performance metrics"""
        if not trades:
            return {
                "passed": False,
                "reason": "No trades generated",
                "total_trades": 0
            }
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Calculate max drawdown
        peak = self.initial_balance
        max_drawdown = 0
        for point in equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            drawdown = (peak - point['equity']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t['pnl'] / self.initial_balance for t in trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Determine if passed criteria
        passed = True
        reasons = []
        
        if total_trades < self.min_trades:
            passed = False
            reasons.append(f"Only {total_trades} trades (min: {self.min_trades})")
        
        if profit_factor < self.min_profit_factor:
            passed = False
            reasons.append(f"Profit factor {profit_factor:.2f} < {self.min_profit_factor}")
        
        if max_drawdown > self.max_drawdown_percent:
            passed = False
            reasons.append(f"Max drawdown {max_drawdown:.1f}% > {self.max_drawdown_percent}%")
        
        return {
            "passed": passed,
            "reasons": reasons,
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_percent": round(max_drawdown, 1),
            "sharpe_ratio": round(sharpe, 2),
            "total_pnl": round(sum(t['pnl'] for t in trades), 2),
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    
    def run_walk_forward(self, symbol: str, strategy_func,
                         train_months: int = 3, test_months: int = 1,
                         total_months: int = 12, timeframe: str = "H1") -> List[Dict]:
        """
        Run walk-forward test - train on period, test on unseen period.
        This validates that the strategy isn't curve-fitted.
        """
        results = []
        end_date = datetime.now()
        current_date = end_date - timedelta(days=total_months * 30)
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + timedelta(days=train_months * 30)
            
            # Test period
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)
            
            if test_end > end_date:
                test_end = end_date
            
            logger.info(f"Walk-forward: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
            
            # Run backtest on training period
            train_result = self.run_backtest(symbol, strategy_func, train_start, train_end, timeframe)
            
            # Run backtest on test period (unseen data)
            test_result = self.run_backtest(symbol, strategy_func, test_start, test_end, timeframe)
            
            results.append({
                "train_period": {"start": train_start.isoformat(), "end": train_end.isoformat()},
                "test_period": {"start": test_start.isoformat(), "end": test_end.isoformat()},
                "train_metrics": train_result.get("metrics", {}),
                "test_metrics": test_result.get("metrics", {}),
                "train_passed": train_result.get("metrics", {}).get("passed", False),
                "test_passed": test_result.get("metrics", {}).get("passed", False)
            })
            
            # Move to next window
            current_date = test_end
        
        return results
    
    def compare_strategies(self, symbol: str, strategies: Dict[str, callable],
                           start_date: datetime, end_date: datetime,
                           timeframe: str = "H1") -> Dict:
        """Compare multiple strategies on the same data"""
        results = {}
        
        for name, strategy_func in strategies.items():
            logger.info(f"Testing strategy: {name}")
            result = self.run_backtest(symbol, strategy_func, start_date, end_date, timeframe)
            results[name] = result
        
        # Find best strategy
        best_name = max(results.keys(), key=lambda k: results[k].get("metrics", {}).get("profit_factor", 0))
        
        return {
            "results": results,
            "best_strategy": best_name,
            "best_metrics": results[best_name].get("metrics", {})
        }

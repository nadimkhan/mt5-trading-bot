"""
MT5 Connector - Direct connection to MetaTrader 5
"""
import MetaTrader5 as mt5
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MT5Connector:
    def __init__(self, login=None, password=None, server=None, path=None):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        
    def connect(self):
        """Initialize MT5 connection"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
                
            # Login if credentials provided
            if self.login:
                if not mt5.login(self.login, password=self.password, server=self.server):
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    return False
                    
            self.connected = True
            account_info = mt5.account_info()
            logger.info(f"MT5 Connected. Account: {account_info.login}, Balance: ${account_info.balance}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Disconnected")
            
    def is_connected(self):
        """Check if connected"""
        if not self.connected:
            return False
        # Verify connection is still alive
        try:
            mt5.account_info()
            return True
        except Exception:
            self.connected = False
            return False
    
    def reconnect(self):
        """Attempt to reconnect to MT5"""
        try:
            # Shutdown and reinitialize
            self.disconnect()
            time.sleep(1)
            return self.connect()
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            return False
            
    def get_account_info(self):
        """Get account information"""
        try:
            info = mt5.account_info()
            return {
                "login": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "profit": info.profit,
                "leverage": info.leverage
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def get_spread(self, symbol):
        """Get current spread for a symbol in pips"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0
            # Spread from symbol_info is in points
            spread_points = symbol_info.spread
            # Convert points to pips based on digits
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                return spread_points / 10  # 5/3-digit broker
            return spread_points
        except Exception as e:
            logger.error(f"Failed to get spread for {symbol}: {e}")
            return 0

    def get_symbol_info(self, symbol):
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                "symbol": info.name,
                "bid": info.bid,
                "ask": info.ask,
                "spread": info.spread,
                "digits": info.digits,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step
            }
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
            
    def get_current_price(self, symbol):
        """Get current bid/ask price"""
        try:
            info = mt5.symbol_info_tick(symbol)
            if info is None:
                return None
            return {
                "symbol": symbol,
                "bid": info.bid,
                "ask": info.ask,
                "time": datetime.fromtimestamp(info.time)
            }
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
            
    def get_ohlcv(self, symbol, timeframe, count=100):
        """Get OHLCV bars"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }
        
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                return []
                
            bars = []
            for rate in rates:
                bars.append({
                    "time": datetime.fromtimestamp(rate['time']),
                    "open": rate['open'],
                    "high": rate['high'],
                    "low": rate['low'],
                    "close": rate['close'],
                    "volume": rate['tick_volume']
                })
            return bars
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return []
            
    def get_ticks(self, symbol, count=100):
        """Get recent tick data"""
        try:
            ticks = mt5.copy_ticks_from(symbol, None, count, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                return []
                
            tick_list = []
            for tick in ticks[-count:]:
                tick_list.append({
                    "time": datetime.fromtimestamp(tick['time']),
                    "bid": tick['bid'],
                    "ask": tick['ask'],
                    "volume": tick['volume']
                })
            return tick_list
        except Exception as e:
            logger.error(f"Failed to get ticks for {symbol}: {e}")
            return []
            
    def get_positions(self):
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
                
            pos_list = []
            for pos in positions:
                pos_list.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "magic": pos.magic,
                    "time": datetime.fromtimestamp(pos.time)
                })
            return pos_list
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
            
    def get_orders(self):
        """Get all pending orders"""
        try:
            orders = mt5.orders_get()
            if orders is None:
                return []
                
            order_list = []
            for order in orders:
                order_list.append({
                    "ticket": order.ticket,
                    "symbol": order.symbol,
                    "type": order.type,
                    "volume": order.volume_current,
                    "price_open": order.price_open,
                    "price_current": order.price_current,
                    "state": order.state,
                    "time": datetime.fromtimestamp(order.time_setup)
                })
            return order_list
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def get_history_deals(self, from_date, to_date):
        """Get closed deals from MT5 history"""
        try:
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None:
                return []
            result = []
            for deal in deals:
                result.append({
                    'ticket': deal.ticket,
                    'symbol': deal.symbol,
                    'entry': deal.entry,
                    'type': deal.type,
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'time': datetime.fromtimestamp(deal.time)
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get history deals: {e}")
            return []

    def get_symbols_list(self):
        """Get all available trading symbols from MT5"""
        try:
            # Get total number of symbols
            total = mt5.symbols_total()
            if total <= 0:
                logger.warning("No symbols found in MT5")
                return []
            
            # Get all symbols
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            
            # Filter to only forex and commodities (common extensions)
            symbol_list = []
            for s in symbols:
                name = s.name
                # Include: forex pairs, XAUUSD, XAGUSD, BRN, WTI, etc.
                # Exclude: indices, stocks, crypto, etc. that might not be tradeable
                if any(ext in name for ext in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'XAU', 'XAG', 'BRN', 'WTI', 'OIL']):
                    symbol_list.append(name)
            
            # Remove duplicates and sort
            symbol_list = sorted(set(symbol_list))
            logger.info(f"Found {len(symbol_list)} tradeable symbols")
            return symbol_list
        except Exception as e:
            logger.error(f"Failed to get symbols list: {e}")
            return []
            
    def send_order(self, symbol, order_type, volume, sl=None, tp=None, comment="", hidden_sl_tp=False):
        """Send a market order.

        If hidden_sl_tp=True, sends sl=0 and tp=0 to MT5 (no SL/TP visible in terminal).
        The SL/TP values are returned to the caller for in-memory management.
        Use this when you want to manage exits yourself (e.g., regime-change exits).
        """
        try:
            # Get symbol info (needed for both modes)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Unknown symbol: {symbol}")
                return None

            # If hidden_sl_tp, skip SL/TP validation and send zeros
            if hidden_sl_tp:
                logger.info(f"{symbol}: Sending order with HIDDEN SL/TP (managed externally)")
                sl = 0.0
                tp = 0.0

            # Get current prices
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get price for {symbol}")
                return None

            # Prepare request
            if order_type.upper() == "BUY":
                price = tick.ask
                order_mt5 = mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                order_mt5 = mt5.ORDER_TYPE_SELL

            # Validate SL/TP against minimum stop level
            # MT5 requires SL/TP to be at LEAST trade_stops_level away from price
            # We add 2x safety multiplier to handle spread, commission, and rounding
            point = symbol_info.point
            min_stop_distance = symbol_info.trade_stops_level * point * 2
            if sl is not None:
                if order_mt5 == mt5.ORDER_TYPE_BUY:
                    # SL must be below price by at least min_stop_distance
                    if sl > price - min_stop_distance:
                        sl = round(price - min_stop_distance - min_stop_distance * 0.1, symbol_info.digits)
                        logger.info(f"SL adjusted to respect min stop level: {sl}")
                else:
                    # SL must be above price by at least min_stop_distance
                    if sl < price + min_stop_distance:
                        sl = round(price + min_stop_distance + min_stop_distance * 0.1, symbol_info.digits)
                        logger.info(f"SL adjusted to respect min stop level: {sl}")
            if tp is not None:
                if order_mt5 == mt5.ORDER_TYPE_BUY:
                    if tp < price + min_stop_distance:
                        tp = round(price + min_stop_distance + min_stop_distance * 0.1, symbol_info.digits)
                        logger.info(f"TP adjusted to respect min stop level: {tp}")
                else:
                    if tp > price - min_stop_distance:
                        tp = round(price - min_stop_distance - min_stop_distance * 0.1, symbol_info.digits)
                        logger.info(f"TP adjusted to respect min stop level: {tp}")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_mt5,
                "price": price,
                "deviation": 10,
                "magic": 123456,  # Magic number to identify our orders
                "comment": comment,
                "type_filling": mt5.ORDER_FILLING_IOC  # IOC is the only mode supported by this demo broker
            }

            # Add SL/TP if provided (after validation)
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp

            # Send order
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment} (retcode={result.retcode})")
                return None

            logger.info(f"Order sent: {order_type} {volume} {symbol} @ {price}, Ticket: {result.order}")
            return {
                "ticket": result.order,
                "retcode": result.retcode,
                "deal": result.deal,
                "price": result.price,
                "comment": result.comment
            }

        except Exception as e:
            logger.error(f"Failed to send order: {e}")
            return None

    def modify_sl(self, ticket, new_sl, new_tp=None):
        """Modify stop loss (and optionally take profit) of an open position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position or len(position) == 0:
                logger.error(f"Position {ticket} not found for SL modification")
                return False
            pos = position[0]
            # Skip if SL is already at the target value (avoid "No changes" roundtrip)
            if pos.sl and abs(pos.sl - new_sl) < 0.00001:
                return True
            # Get symbol's minimum stop level
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info:
                # stop_level is in points, convert to price
                # MT5 requires SL to be at LEAST trade_stops_level away from price
                # We add a 2x safety multiplier to account for spread, commission, and rounding
                point = symbol_info.point
                min_stop_distance = symbol_info.trade_stops_level * point * 2
                current_price = pos.price_current
                # Ensure SL is far enough from current price
                if pos.type == 0:  # BUY
                    if new_sl > current_price - min_stop_distance:
                        new_sl = round(current_price - min_stop_distance, symbol_info.digits)
                        logger.debug(f"SL adjusted to respect min stop level: {new_sl}")
                else:  # SELL
                    if new_sl < current_price + min_stop_distance:
                        new_sl = round(current_price + min_stop_distance, symbol_info.digits)
                        logger.debug(f"SL adjusted to respect min stop level: {new_sl}")
                # Final check: don't even try if SL is unchanged
                if pos.sl and abs(pos.sl - new_sl) < 0.00001:
                    return True
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": new_sl,
                "tp": new_tp if new_tp else pos.tp,
            }
            result = mt5.order_send(request)
            # 10009 = success, 10025 = "No changes" (also success - SL already at target)
            if result and result.retcode in (mt5.TRADE_RETCODE_DONE, 10025):
                if result.retcode == 10025:
                    logger.debug(f"SL already at target for {ticket}: {new_sl}")
                else:
                    logger.info(f"SL modified for {ticket}: new SL={new_sl}")
                return True
            else:
                retcode = result.retcode if result else 'no result'
                comment = result.comment if result else ''
                logger.error(f"Failed to modify SL: retcode={retcode} comment={comment}")
                return False
        except Exception as e:
            logger.error(f"Error modifying SL: {e}")
            return False

    def close_position(self, ticket, volume=None):
        """Close a position"""
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                logger.error(f"Position {ticket} not found")
                return False
                
            pos = positions[0]
            symbol = pos.symbol
            order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            volume_to_close = volume if volume else pos.volume
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_to_close,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Closed by trading bot"
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close failed: {result.comment}")
                return False
                
            logger.info(f"Position {ticket} closed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position {ticket}: {e}")
            return False
            
    def calculate_lot_size(self, symbol, risk_percent, sl_pips):
        """Calculate lot size based on risk"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01
                
            account = mt5.account_info()
            if account is None:
                return 0.01
                
            # Calculate risk amount in currency
            risk_amount = account.balance * (risk_percent / 100)
            
            # Get contract size (usually 100000 for forex)
            contract_size = symbol_info.trade_contract_size
            
            # Calculate pip value
            pip_value = (symbol_info.point * contract_size) / symbol_info.bid
            
            # Calculate lot size
            lot_size = risk_amount / (sl_pips * pip_value)
            
            # Round to step
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
            
            # Apply limits
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            lot_size = min(lot_size, 0.10)  # Cap at 0.10 for safety
            
            return round(lot_size, 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate lot size: {e}")
            return 0.01


# Test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    connector = MT5Connector()
    
    if connector.connect():
        print("✓ MT5 Connected")
        
        # Test get account info
        info = connector.get_account_info()
        print(f"Account: {info}")
        
        # Test get price
        price = connector.get_current_price("EURUSD")
        print(f"EURUSD: {price}")
        
        # Test get OHLCV
        bars = connector.get_ohlcv("EURUSD", "H1", 10)
        print(f"Got {len(bars)} bars")
        
        # Test get positions
        positions = connector.get_positions()
        print(f"Open positions: {len(positions)}")
        
        connector.disconnect()
    else:
        print("✗ Failed to connect to MT5")

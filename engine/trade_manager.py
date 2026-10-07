"""
Trade Management Module
Handles SL/TP server-side, breakeven moves, partial take-profit, and ATR trailing stops.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class TradeManager:
    """Manages open positions with server-side SL/TP and trailing stops"""
    
    def __init__(self, mt5_connector, config: dict):
        self.mt5 = mt5_connector
        self.config = config
        self.magic = config.get("mt5", {}).get("magic_number", 123456)
        
        # Risk settings
        risk_config = config.get("risk", {})
        self.risk_per_trade_pct = risk_config.get("max_risk_per_trade", 1.0)  # 1% default
        self.daily_loss_limit_pct = risk_config.get("daily_loss_limit", 3.0)  # 3% default
        self.spread_cap_pips = risk_config.get("spread_cap_pips", 20)
        
        # ATR settings for trailing
        atr_config = config.get("trailing", {})
        self.use_atr_trailing = atr_config.get("enabled", True)
        self.atr_multiplier = atr_config.get("atr_multiplier", 3.0)
        self.atr_period = atr_config.get("period", 14)
        self.trailing_threshold_pips = atr_config.get("threshold_pips", 15)
        
        # Breakeven settings
        be_config = config.get("breakeven", {})
        self.breakeven_enabled = be_config.get("enabled", True)
        self.breakeven_trigger_pips = be_config.get("trigger_pips", 20)
        
        # Partial TP settings
        ptp_config = config.get("partial_tp", {})
        self.partial_tp_enabled = ptp_config.get("enabled", True)
        self.partial_tp_pct = ptp_config.get("percentage", 50)  # Close 50% at TP1
        self.partial_tp_at_pips = ptp_config.get("at_pips", 30)
        
        # Track managed positions
        self._managed_positions: Dict[int, dict] = {}
        
        # Daily P&L tracking
        self._daily_pnl = 0.0
        self._last_reset_date = datetime.now().date()
    
    def calculate_atr(self, symbol: str, timeframe: str = "H1") -> float:
        """Get ATR value for a symbol"""
        try:
            bars = self.mt5.get_ohlcv(symbol, timeframe, self.atr_period + 2)
            if len(bars) < self.atr_period + 1:
                return None
                
            # Calculate True Range
            trs = []
            for i in range(1, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                trs.append(tr)
            
            if len(trs) < self.atr_period:
                return None
                
            atr = sum(trs[-self.atr_period:]) / self.atr_period
            return atr
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, sl_pips: float) -> float:
        """Calculate lot size based on risk and ATR-based stop"""
        try:
            account = self.mt5.get_account_info()
            if not account:
                return 0.01
            
            symbol_info = self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Risk amount in account currency
            risk_amount = account['balance'] * (self.risk_per_trade_pct / 100)
            
            # Pip value for this symbol
            contract_size = symbol_info['trade_contract_size']
            pip_size = symbol_info['point'] * 10 if 'JPY' not in symbol else symbol_info['point'] * 100
            pip_value = (pip_size * contract_size) / symbol_info['bid']
            
            # Lot size from risk
            if sl_pips <= 0 or pip_value <= 0:
                return 0.01
                
            lot = risk_amount / (sl_pips * pip_value)
            
            # Round to step
            lot = round(lot / symbol_info['volume_step']) * symbol_info['volume_step']
            
            # Apply limits
            lot = max(symbol_info['volume_min'], min(lot, symbol_info['volume_max']))
            lot = min(lot, 0.10)  # Cap at 0.10
            
            return round(lot, 2)
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.01
    
    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify position SL server-side via MT5"""
        try:
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "magic": self.magic
            })
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Modified SL for ticket {ticket} to {new_sl}")
                return True
            else:
                logger.warning(f"Failed to modify SL: {result.comment}")
                return False
        except Exception as e:
            logger.error(f"Failed to modify SL: {e}")
            return False
    
    def move_breakeven(self, position: dict) -> bool:
        """Move SL to breakeven if profit exceeds trigger"""
        if not self.breakeven_enabled:
            return False
            
        ticket = position['ticket']
        current_sl = position.get('sl', 0)
        entry_price = position['price_open']
        current_price = position['price_current']
        pos_type = position['type']  # 0=BUY, 1=SELL
        
        # Calculate profit in pips
        if pos_type == 0:  # BUY
            profit_pips = (current_price - entry_price) / self._get_pip_size(position['symbol'])
        else:  # SELL
            profit_pips = (entry_price - current_price) / self._get_pip_size(position['symbol'])
        
        # Check if we should move to breakeven
        if profit_pips >= self.breakeven_trigger_pips:
            # Calculate new SL (breakeven + small buffer)
            pip_size = self._get_pip_size(position['symbol'])
            buffer = pip_size * 2  # 2 pip buffer
            
            if pos_type == 0:
                new_sl = entry_price + buffer
            else:
                new_sl = entry_price - buffer
            
            # Only move if new SL is better
            if pos_type == 0 and new_sl > current_sl:
                return self.modify_position_sl(ticket, new_sl)
            elif pos_type == 1 and (current_sl == 0 or new_sl < current_sl):
                return self.modify_position_sl(ticket, new_sl)
        
        return False
    
    def apply_partial_tp(self, position: dict) -> bool:
        """Close partial position at TP level"""
        if not self.partial_tp_enabled:
            return False
            
        ticket = position['ticket']
        entry_price = position['price_open']
        current_price = position['price_current']
        pos_type = position['type']
        volume = position['volume']
        symbol = position['symbol']
        
        # Calculate profit in pips
        pip_size = self._get_pip_size(symbol)
        if pos_type == 0:
            profit_pips = (current_price - entry_price) / pip_size
        else:
            profit_pips = (entry_price - current_price) / pip_size
        
        # Check if we hit partial TP
        if profit_pips >= self.partial_tp_at_pips:
            partial_volume = round(volume * (self.partial_tp_pct / 100), 2)
            if partial_volume >= volume_min_for_symbol(symbol):
                success = self.mt5.close_position(ticket, partial_volume)
                if success:
                    logger.info(f"Partial TP hit: closed {partial_volume} of {volume} on {ticket}")
                    return True
        return False
    
    def update_trailing_stop(self, position: dict) -> bool:
        """Update trailing stop based on ATR"""
        if not self.use_atr_trailing:
            return False
            
        ticket = position['ticket']
        entry_price = position['price_open']
        current_price = position['price_current']
        current_sl = position.get('sl', 0)
        pos_type = position['type']
        symbol = position['symbol']
        
        pip_size = self._get_pip_size(symbol)
        
        # Calculate profit in pips
        if pos_type == 0:
            profit_pips = (current_price - entry_price) / pip_size
        else:
            profit_pips = (entry_price - current_price) / pip_size
        
        # Only trail if profit exceeds threshold
        if profit_pips < self.trailing_threshold_pips:
            return False
        
        # Get ATR
        atr = self.calculate_atr(symbol)
        if atr is None:
            return False
        
        atr_pips = atr / pip_size
        stop_distance_pips = atr_pips * self.atr_multiplier
        
        # New trailing SL
        if pos_type == 0:
            new_sl = current_price - (stop_distance_pips * pip_size)
            # Only move up
            if new_sl <= current_sl:
                return False
        else:
            new_sl = current_price + (stop_distance_pips * pip_size)
            # Only move down
            if current_sl != 0 and new_sl >= current_sl:
                return False
        
        return self.modify_position_sl(ticket, new_sl)
    
    def check_and_manage_position(self, position: dict) -> dict:
        """Run all trade management checks on a position"""
        ticket = position['ticket']
        
        # Initialize tracking if new
        if ticket not in self._managed_positions:
            self._managed_positions[ticket] = {
                'breakeven_moved': False,
                'partial_tp_hit': False,
                'trailing_active': False,
                'entry_sl': position.get('sl', 0)
            }
        
        results = {
            'breakeven': False,
            'partial_tp': False,
            'trailing': False
        }
        
        # Check breakeven
        if not self._managed_positions[ticket].get('breakeven_moved'):
            if self.move_breakeven(position):
                results['breakeven'] = True
                self._managed_positions[ticket]['breakeven_moved'] = True
        
        # Check partial TP
        if not self._managed_positions[ticket].get('partial_tp_hit'):
            if self.apply_partial_tp(position):
                results['partial_tp'] = True
                self._managed_positions[ticket]['partial_tp_hit'] = True
        
        # Check trailing stop
        if self.update_trailing_stop(position):
            results['trailing'] = True
        
        return results
    
    def manage_all_positions(self) -> dict:
        """Check and manage all open positions"""
        positions = self.mt5.get_positions()
        results = {
            'managed': 0,
            'actions': []
        }
        
        for pos in positions:
            # Only manage our positions (by magic number)
            if pos.get('magic') == self.magic:
                action_results = self.check_and_manage_position(pos)
                if any(action_results.values()):
                    results['managed'] += 1
                    results['actions'].append({
                        'ticket': pos['ticket'],
                        'symbol': pos['symbol'],
                        **action_results
                    })
        
        return results
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol"""
        if 'JPY' in symbol:
            return 0.01
        return 0.0001
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is breached"""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._last_reset_date = today
        
        account = self.mt5.get_account_info()
        if account:
            # Daily loss limit as percentage of account
            max_loss = account['balance'] * (self.daily_loss_limit_pct / 100)
            if self._daily_pnl <= -max_loss:
                logger.warning(f"Daily loss limit breached: {self._daily_pnl:.2f}")
                return True
        return False
    
    def close_all_positions(self, reason: str = "Kill switch") -> dict:
        """Emergency close all positions"""
        logger.warning(f"Closing ALL positions: {reason}")
        closed = []
        positions = self.mt5.get_positions()
        
        for pos in positions:
            if pos.get('magic') == self.magic:
                success = self.mt5.close_position(pos['ticket'])
                closed.append({
                    'ticket': pos['ticket'],
                    'symbol': pos['symbol'],
                    'success': success
                })
        
        return {'closed': closed, 'reason': reason}
    
    def update_daily_pnl(self, closed_pnl: float):
        """Track daily P&L"""
        self._daily_pnl += closed_pnl


def volume_min_for_symbol(symbol: str) -> float:
    """Get minimum volume step for a symbol"""
    try:
        info = mt5.symbol_info(symbol)
        if info:
            return info.volume_min
    except:
        pass
    return 0.01

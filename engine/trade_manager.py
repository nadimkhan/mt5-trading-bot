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

        # Hidden SL/TP mode (SL/TP stored in our DB, not sent to MT5)
        self.hidden_sl_tp = config.get("trade_management", {}).get("hidden_sl_tp", False)

        # Regime-change exit: close positions when regime changes from entry
        self.regime_exit_enabled = config.get("trade_management", {}).get("regime_exit", True)
        # Track position regime at entry: {ticket: regime_at_entry}
        self._position_regimes = {}

        # Hidden SL/TP mode: {ticket: {'sl': price, 'tp': price, 'symbol': str}}
        # Only used when hidden_sl_tp=True in config
        self._internal_sl_tps = {}

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
        """Modify position SL server-side via MT5 - delegates to connector for validation"""
        if not self.mt5:
            return False
        return self.mt5.modify_sl(ticket, new_sl)
    
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
        """
        Close partial position at TP1 (default 1:1 RR).
        Also moves stop to breakeven after partial TP is hit.
        """
        if not self.partial_tp_enabled:
            return False

        ticket = position['ticket']
        entry_price = position['price_open']
        current_price = position['price_current']
        pos_type = position['type']
        volume = position['volume']
        symbol = position['symbol']
        current_sl = position.get('sl', 0)

        # Calculate profit in pips
        pip_size = self._get_pip_size(symbol)
        if pos_type == 0:  # BUY
            profit_pips = (current_price - entry_price) / pip_size
        else:  # SELL
            profit_pips = (entry_price - current_price) / pip_size

        # Get current SL distance
        sl_distance_pips = 0
        if current_sl and current_sl > 0:
            if pos_type == 0:
                sl_distance_pips = (entry_price - current_sl) / pip_size
            else:
                sl_distance_pips = (current_sl - entry_price) / pip_size

        # TP1 is at 1:1 RR by default (at the same distance as SL)
        tp1_pips = sl_distance_pips if sl_distance_pips > 0 else self.partial_tp_at_pips

        # Check if we hit TP1
        if profit_pips >= tp1_pips:
            partial_volume = round(volume * (self.partial_tp_pct / 100), 2)
            if partial_volume >= volume_min_for_symbol(symbol):
                success = self.mt5.close_position(ticket, partial_volume)
                if success:
                    logger.info(f"Partial TP hit at {profit_pips:.1f} pips (TP1={tp1_pips:.1f}): closed {partial_volume} of {volume} on {ticket}")
                    # Move SL to breakeven + small buffer
                    if current_sl and current_sl > 0:
                        buffer = pip_size * 2  # 2 pip buffer
                        if pos_type == 0:  # BUY
                            new_sl = entry_price + buffer
                            if new_sl > current_sl:  # Only move SL up
                                self.mt5.modify_sl(ticket, new_sl)
                                logger.info(f"SL moved to breakeven+buffer: {new_sl}")
                        else:  # SELL
                            new_sl = entry_price - buffer
                            if new_sl < current_sl:  # Only move SL down
                                self.mt5.modify_sl(ticket, new_sl)
                                logger.info(f"SL moved to breakeven+buffer: {new_sl}")
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

    def record_position_regime(self, ticket: int, regime: str):
        """Record the regime at position entry for later regime-change detection."""
        self._position_regimes[ticket] = regime

    def record_internal_sl_tp(self, ticket: int, sl: float, tp: float, symbol: str):
        """Store SL/TP in memory (when hidden_sl_tp=True)."""
        self._internal_sl_tps[ticket] = {'sl': sl, 'tp': tp, 'symbol': symbol}

    def clear_internal_sl_tp(self, ticket: int):
        """Remove internal SL/TP tracking for a closed position."""
        self._internal_sl_tps.pop(ticket, None)

    def get_entry_regime(self, ticket: int) -> str:
        """Get the regime at which a position was opened."""
        return self._position_regimes.get(ticket, "UNKNOWN")

    def clear_position_regime(self, ticket: int):
        """Remove regime tracking for a closed position."""
        self._position_regimes.pop(ticket, None)

    def check_and_close_on_sl_tp_hit(self, internal_sl_tps: Dict[int, Dict]) -> list:
        """Check if any positions hit their internal (hidden) SL/TP and close them.

        Args:
            internal_sl_tps: {ticket: {'sl': price, 'tp': price, 'symbol': str}}

        Returns:
            List of {ticket, symbol, reason} for closed positions
        """
        closed = []
        if not self.mt5:
            return closed
        try:
            positions = self.mt5.get_positions() or []
            for pos in positions:
                if pos.get('magic') != self.magic:
                    continue
                ticket = pos.get('ticket')
                if ticket not in internal_sl_tps:
                    continue
                levels = internal_sl_tps[ticket]
                sl = levels.get('sl')
                tp = levels.get('tp')
                current = pos.get('price_current', 0)
                pos_type = pos.get('type', 0)  # 0=buy, 1=sell
                # For BUY positions: SL below current, TP above current
                # For SELL positions: SL above current, TP below current
                hit_sl = False
                hit_tp = False
                if pos_type == 0:  # BUY
                    if sl and current > 0 and current <= sl:
                        hit_sl = True
                    if tp and current > 0 and current >= tp:
                        hit_tp = True
                else:  # SELL
                    if sl and current > 0 and current >= sl:
                        hit_sl = True
                    if tp and current > 0 and current <= tp:
                        hit_tp = True
                if hit_sl or hit_tp:
                    reason = 'SL hit' if hit_sl else 'TP hit'
                    logger.info(f"{pos.get('symbol')} ticket {ticket}: {reason} (hidden), closing position")
                    result = self.mt5.close_position(ticket)
                    if result:
                        closed.append({
                            'ticket': ticket,
                            'symbol': pos.get('symbol'),
                            'reason': f'Hidden {reason}'
                        })
        except Exception as e:
            logger.error(f"Hidden SL/TP check error: {e}")
        return closed

    def check_and_close_on_regime_change(self, current_regimes: Dict[str, str]) -> list:
        """Close positions where the regime has changed from entry.

        Args:
            current_regimes: {symbol: current_regime} from regime detection

        Returns:
            List of {ticket, symbol, reason} for closed positions
        """
        closed = []
        if not self.regime_exit_enabled:
            return closed
        if not self.mt5:
            return closed
        try:
            positions = self.mt5.get_positions() or []
            for pos in positions:
                if pos.get('magic') != self.magic:
                    continue
                ticket = pos.get('ticket')
                symbol = pos.get('symbol')
                entry_regime = self.get_entry_regime(ticket)
                current = current_regimes.get(symbol, "UNKNOWN")
                # Close if regime flipped (BULL<->BEAR) or went to SIDEWAYS
                if entry_regime != "UNKNOWN" and current != "UNKNOWN":
                    flipped = (
                        (entry_regime == "BULL" and current in ("BEAR", "SIDEWAYS")) or
                        (entry_regime == "BEAR" and current in ("BULL", "SIDEWAYS")) or
                        (entry_regime == "SIDEWAYS" and current in ("BULL", "BEAR"))
                    )
                    if flipped:
                        logger.info(f"{symbol} ticket {ticket}: regime changed {entry_regime} -> {current}, closing position")
                        result = self.mt5.close_position(ticket)
                        if result:
                            closed.append({
                                'ticket': ticket,
                                'symbol': symbol,
                                'reason': f'Regime changed: {entry_regime} -> {current}'
                            })
                            self.clear_position_regime(ticket)
        except Exception as e:
            logger.error(f"Regime-change check error: {e}")
        return closed


def volume_min_for_symbol(symbol: str) -> float:
    """Get minimum volume step for a symbol"""
    try:
        info = mt5.symbol_info(symbol)
        if info:
            return info.volume_min
    except Exception:
        pass
    return 0.01

"""
Operational Safeguards
- Automatic reconnection with exponential backoff
- Magic number filtering (only manage our own trades)
- Drawdown kill switch
- Position re-sync on disconnect
"""
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class Safeguards:
    """Operational safety features"""
    
    def __init__(self, mt5_connector, config: dict):
        self.mt5 = mt5_connector
        self.config = config
        self.magic = config.get("mt5", {}).get("magic_number", 123456)
        
        # Reconnect settings
        reconnect_config = config.get("reconnect", {})
        self.max_reconnect_attempts = reconnect_config.get("max_attempts", 5)
        self.initial_reconnect_delay = reconnect_config.get("initial_delay_seconds", 2)
        self.max_reconnect_delay = reconnect_config.get("max_delay_seconds", 60)
        
        # Drawdown settings
        drawdown_config = config.get("drawdown", {})
        self.max_drawdown_percent = drawdown_config.get("max_percent", 5.0)
        self.daily_drawdown_limit = drawdown_config.get("daily_limit_percent", 3.0)
        
        # Track state
        self._reconnect_attempts = 0
        self._last_position_sync = None
        self._connection_lost_at = None
        
        # Account snapshot at start
        self._account_balance_start = None
        self._account_equity_start = None
    
    def initialize(self, account_info: dict):
        """Initialize safeguards with account info"""
        self._account_balance_start = account_info.get("balance", 0)
        self._account_equity_start = account_info.get("equity", account_info.get("balance", 0))
        logger.info(f"Safeguards initialized: Balance={self._account_balance_start}, Equity={self._account_equity_start}")
    
    def filter_our_positions(self, positions: list) -> list:
        """Filter positions to only our magic number"""
        our_positions = []
        for pos in positions:
            if pos.get("magic") == self.magic:
                our_positions.append(pos)
            else:
                logger.debug(f"Filtered out position {pos.get('ticket')} - magic {pos.get('magic')} != {self.magic}")
        return our_positions
    
    def check_connection(self) -> bool:
        """Check if MT5 connection is alive"""
        if not self.mt5:
            return False
        
        try:
            return self.mt5.is_connected()
        except Exception:
            return False
    
    def handle_disconnect(self) -> dict:
        """Handle disconnection with reconnection logic"""
        if self._connection_lost_at is None:
            self._connection_lost_at = datetime.now()
            logger.warning("MT5 connection lost - initiating reconnection...")
        
        # Calculate delay with exponential backoff
        delay = min(
            self.initial_reconnect_delay * (2 ** self._reconnect_attempts),
            self.max_reconnect_delay
        )
        
        logger.info(f"Reconnect attempt {self._reconnect_attempts + 1}/{self.max_reconnect_attempts} in {delay}s")
        
        time.sleep(delay)
        
        # Attempt reconnect
        if self.mt5.reconnect():
            self._reconnect_attempts = 0
            self._connection_lost_at = None
            logger.info("MT5 reconnected successfully")
            
            # Sync positions after reconnect
            positions = self.mt5.get_positions()
            self._last_position_sync = datetime.now()
            
            return {
                "reconnected": True,
                "positions_synced": len(positions),
                "positions": positions
            }
        else:
            self._reconnect_attempts += 1
            
            if self._reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(f"Max reconnect attempts ({self.max_reconnect_attempts}) reached")
                return {
                    "reconnected": False,
                    "reason": "Max attempts reached",
                    "attempts": self._reconnect_attempts
                }
        
        return {
            "reconnected": False,
            "attempts": self._reconnect_attempts,
            "delay": delay
        }
    
    def check_drawdown(self, account_info: dict) -> dict:
        """
        Check if drawdown limits are breached.
        Returns dict with 'breached' bool and details.
        """
        current_equity = account_info.get("equity", 0)
        current_balance = account_info.get("balance", 0)
        
        results = {
            "breached": False,
            "max_drawdown_breached": False,
            "daily_drawdown_breached": False,
            "details": {}
        }
        
        # Check max drawdown from starting equity
        if self._account_equity_start and self._account_equity_start > 0:
            drawdown = (self._account_equity_start - current_equity) / self._account_equity_start * 100
            
            if drawdown > self.max_drawdown_percent:
                results["breached"] = True
                results["max_drawdown_breached"] = True
                results["details"]["max_drawdown"] = {
                    "limit": self.max_drawdown_percent,
                    "current": round(drawdown, 2),
                    "start_equity": self._account_equity_start,
                    "current_equity": current_equity
                }
                logger.warning(f"Max drawdown breached: {drawdown:.2f}% (limit: {self.max_drawdown_percent}%)")
        
        # Check daily drawdown (from day's high or balance)
        if self._account_balance_start and self._account_balance_start > 0:
            daily_drawdown = (self._account_balance_start - current_balance) / self._account_balance_start * 100
            
            if daily_drawdown > self.daily_drawdown_limit:
                results["breached"] = True
                results["daily_drawdown_breached"] = True
                results["details"]["daily_drawdown"] = {
                    "limit": self.daily_drawdown_limit,
                    "current": round(daily_drawdown, 2),
                    "start_balance": self._account_balance_start,
                    "current_balance": current_balance
                }
                logger.warning(f"Daily drawdown breached: {daily_drawdown:.2f}% (limit: {self.daily_drawdown_limit}%)")
        
        return results
    
    def should_kill_switch(self, account_info: dict) -> tuple:
        """
        Determine if kill switch should be activated.
        Returns (should_kill: bool, reason: str)
        """
        # Check drawdown
        drawdown_result = self.check_drawdown(account_info)
        if drawdown_result["breached"]:
            reason = ""
            if drawdown_result.get("max_drawdown_breached"):
                details = drawdown_result["details"]["max_drawdown"]
                reason = f"Max drawdown {details['current']}% > {details['limit']}%"
            elif drawdown_result.get("daily_drawdown_breached"):
                details = drawdown_result["details"]["daily_drawdown"]
                reason = f"Daily drawdown {details['current']}% > {details['limit']}%"
            return True, reason
        
        return False, ""
    
    def sync_positions(self) -> list:
        """Force sync positions from MT5"""
        if not self.mt5:
            return []
        
        try:
            all_positions = self.mt5.get_positions()
            our_positions = self.filter_our_positions(all_positions)
            self._last_position_sync = datetime.now()
            logger.info(f"Position sync: {len(our_positions)} our positions out of {len(all_positions)} total")
            return our_positions
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return []
    
    def get_status(self) -> dict:
        """Get safeguards status for dashboard"""
        return {
            "magic_number": self.magic,
            "max_drawdown_percent": self.max_drawdown_percent,
            "daily_drawdown_limit": self.daily_drawdown_limit,
            "account_balance_start": self._account_balance_start,
            "account_equity_start": self._account_equity_start,
            "last_position_sync": self._last_position_sync.isoformat() if self._last_position_sync else None,
            "connection_lost_at": self._connection_lost_at.isoformat() if self._connection_lost_at else None,
            "reconnect_attempts": self._reconnect_attempts
        }

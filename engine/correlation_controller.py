"""
Correlation Controller - Caps exposure per currency
EURUSD, GBPUSD, USDJPY = all USD exposure
If all 3 are long, that's really one big USD bet
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Currency mapping - which currency each pair exposes
CURRENCY_EXPOSURE = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"],
    "USDCAD": ["USD", "CAD"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "XAUUSD": ["XAU", "USD"],  # Gold/USD
    "BRNUSD": ["BRN", "USD"],  # Oil/USD
}


class CorrelationController:
    """Controls correlation exposure per currency"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Correlation settings
        corr_config = self.config.get("correlation", {})
        self.max_per_currency = corr_config.get("max_per_currency", 3)  # Max positions per single currency
        self.max_usd_exposure = corr_config.get("max_usd_exposure", 3)  # Max USD-correlated positions
        
        # Track positions by currency
        self._currency_weights: Dict[str, float] = {}
    
    def get_currencies(self, symbol: str) -> List[str]:
        """Get currencies exposed by a symbol"""
        return CURRENCY_EXPOSURE.get(symbol, [])
    
    def calculate_exposure(self, positions: List[dict]) -> Dict[str, float]:
        """
        Calculate current exposure per currency.
        Returns dict like {"USD": 2.5, "EUR": 1.0, "GBP": 0.5}
        """
        exposure = {}
        
        for pos in positions:
            symbol = pos.get("symbol", "")
            volume = float(pos.get("volume", 0))
            pos_type = pos.get("type", 0)  # 0=BUY, 1=SELL
            
            currencies = self.get_currencies(symbol)
            
            for i, currency in enumerate(currencies):
                # Adjust weight based on position direction
                # For USD pairs: if BUY, we gain USD; if SELL, we lose USD
                if currency == "USD":
                    weight = volume if pos_type == 0 else -volume * 0.5  # SELL on USD pair = partial USD exposure
                elif currency == "EUR":
                    weight = volume if pos_type == 0 else -volume
                elif currency == "GBP":
                    weight = volume if pos_type == 0 else -volume
                else:
                    weight = volume * 0.5  # Other currencies weighted less
                
                exposure[currency] = exposure.get(currency, 0) + weight
        
        self._currency_weights = exposure
        return exposure
    
    def can_open_position(self, symbol: str, positions: List[dict], lot_size: float = 0.01) -> dict:
        """
        Check if we can open a new position without exceeding correlation limits.
        
        Returns:
            {"allowed": bool, "reason": str, "current_exposure": dict}
        """
        # Get currencies for this symbol
        currencies = self.get_currencies(symbol)
        if not currencies:
            return {"allowed": True, "reason": "Unknown symbol - no currency check"}
        
        # Calculate current exposure
        current_exposure = self.calculate_exposure(positions)
        
        # Check USD exposure (most important)
        usd_corr_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
        
        # Count USD-correlated positions
        usd_count = 0
        for pos in positions:
            if pos.get("symbol") in usd_corr_pairs:
                usd_count += 1
        
        if symbol in usd_corr_pairs:
            usd_count += 1  # Adding another USD pair
        
        if usd_count > self.max_usd_exposure:
            return {
                "allowed": False,
                "reason": f"USD exposure cap reached ({usd_count}/{self.max_usd_exposure}). "
                         f"EURUSD, GBPUSD, USDJPY all move together on USD news.",
                "current_exposure": current_exposure
            }
        
        # Check individual currency exposure
        for currency in currencies:
            current = current_exposure.get(currency, 0)
            # Estimate additional exposure
            additional = lot_size if currency == "USD" else lot_size * 0.7
            projected = current + additional
            
            if projected > self.max_per_currency:
                return {
                    "allowed": False,
                    "reason": f"{currency} exposure would be {projected:.2f} (max: {self.max_per_currency})",
                    "current_exposure": current_exposure
                }
        
        return {
            "allowed": True,
            "reason": "Within correlation limits",
            "current_exposure": current_exposure
        }
    
    def get_exposure_report(self, positions: List[dict]) -> dict:
        """Get full exposure report for dashboard"""
        exposure = self.calculate_exposure(positions)
        
        usd_corr_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
        usd_count = sum(1 for p in positions if p.get("symbol") in usd_corr_pairs)
        
        return {
            "by_currency": exposure,
            "usd_correlated_positions": usd_count,
            "max_usd_exposure": self.max_usd_exposure,
            "max_per_currency": self.max_per_currency,
            "status": "OK" if usd_count <= self.max_usd_exposure else "WARNING"
        }
    
    def get_related_pairs(self, symbol: str) -> List[str]:
        """Get pairs that correlate with this symbol"""
        currencies = self.get_currencies(symbol)
        related = []
        
        for pair, pair_currencies in CURRENCY_EXPOSURE.items():
            if pair == symbol:
                continue
            # Check if any currency overlaps
            for c in currencies:
                if c in pair_currencies:
                    related.append(pair)
                    break
        
        return related

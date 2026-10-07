"""
Rule-Based Strategies - Deterministic trading setups
AI acts as a filter, not the decision-maker
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ScalpStrategy:
    """
    Scalping strategy: H4 trend filter + M15 pullback + M1 confirmation
    AI only vetoes based on news/regime.
    
    Entry rules:
    1. H4/M15: Trend direction (EMA 50/200 crossover or price above/below)
    2. M15: Pullback to EMA 20 (entry zone)
    3. M1: Confirmation candle in direction of trend
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Load saved values from disk if available
        try:
            from strategies.strategy_config import load_configs, get_strategy_param
            saved = load_configs().get("scalp", {})
            self.strategy_config = saved.get("parameters", {})
        except Exception:
            self.strategy_config = {}

        # Strategy parameters - read from config or use defaults
        self.ema_fast = int(self._get_param("ema_fast", 9))
        self.ema_medium = int(self._get_param("ema_medium", 20))
        self.ema_slow = int(self._get_param("ema_slow", 50))
        self.ema_trend = int(self._get_param("ema_trend", 200))

        # Entry thresholds
        self.pullback_max_pips = int(self._get_param("pullback_max_pips", 15))
        self.min_confidence = int(self._get_param("min_confidence", 60))
        self.atr_sl_multiplier = float(self._get_param("atr_sl_multiplier", 3.0))
        self.atr_tp_multiplier = float(self._get_param("atr_tp_multiplier", 5.0))
        self.min_risk_reward = float(self._get_param("min_risk_reward", 1.5))
        self.rsi_overbought = int(self._get_param("rsi_overbought", 70))
        self.rsi_oversold = int(self._get_param("rsi_oversold", 30))
        self.max_spread_pips = float(self._get_param("max_spread_pips", 3.0))
        self.max_open_trades = int(self._get_param("max_open_trades", 2))
        self.daily_loss_limit_pct = float(self._get_param("daily_loss_limit_pct", 3.0))

    def _get_param(self, name, default):
        """Get parameter from strategy config or fallback to default"""
        if name in self.strategy_config:
            return self.strategy_config[name].get("value", default)
        return default

    def reload_config(self, all_configs):
        """Reload configuration from saved configs"""
        saved = all_configs.get("scalp", {})
        self.strategy_config = saved.get("parameters", {})
        self.ema_fast = int(self._get_param("ema_fast", 9))
        self.ema_medium = int(self._get_param("ema_medium", 20))
        self.ema_slow = int(self._get_param("ema_slow", 50))
        self.ema_trend = int(self._get_param("ema_trend", 200))
        self.pullback_max_pips = int(self._get_param("pullback_max_pips", 15))
        self.min_confidence = int(self._get_param("min_confidence", 60))
        self.atr_sl_multiplier = float(self._get_param("atr_sl_multiplier", 3.0))
        self.atr_tp_multiplier = float(self._get_param("atr_tp_multiplier", 5.0))
        self.min_risk_reward = float(self._get_param("min_risk_reward", 1.5))
        self.rsi_overbought = int(self._get_param("rsi_overbought", 70))
        self.rsi_oversold = int(self._get_param("rsi_oversold", 30))
        self.max_spread_pips = float(self._get_param("max_spread_pips", 3.0))
        self.max_open_trades = int(self._get_param("max_open_trades", 2))
        self.daily_loss_limit_pct = float(self._get_param("daily_loss_limit_pct", 3.0))
        
    def check_setup(self, market_data: Dict) -> Dict:
        """
        Check if there's a valid setup based on rule-based criteria.
        
        Args:
            market_data: Dict with timeframe data containing 'H4', 'M15', 'M5', 'M1'
            
        Returns:
            Dict with 'signal' (BUY/SELL/HOLD), 'confidence', 'reason'
        """
        h4 = market_data.get('H4', {})
        m15 = market_data.get('M15', {})
        m5 = market_data.get('M5', {})
        m1 = market_data.get('M1', {})
        
        if not all([h4, m15, m5, m1]):
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        # Step 1: H4 Trend Check
        h4_trend = self._check_trend(h4, 'H4')
        if h4_trend == "NONE":
            return {"signal": "HOLD", "confidence": 0, "reason": "H4 trend unclear"}
        
        # Step 2: M15 Entry Zone (pullback to EMA 20)
        m15_entry = self._check_entry_zone(m15, h4_trend)
        if not m15_entry["valid"]:
            return {"signal": "HOLD", "confidence": m15_entry["confidence"], "reason": m15_entry["reason"]}
        
        # Step 3: M1 Confirmation
        m1_confirm = self._check_confirmation(m1, h4_trend)
        if not m1_confirm["valid"]:
            return {"signal": "HOLD", "confidence": m1_confirm["confidence"], "reason": m1_confirm["reason"]}
        
        # All checks passed
        confidence = min(100, 40 + m15_entry["confidence"] + m1_confirm["confidence"])
        
        return {
            "signal": h4_trend,  # BUY if bull, SELL if bear
            "confidence": confidence,
            "reason": f"{h4_trend} setup confirmed: {m15_entry['reason']}, {m1_confirm['reason']}",
            "entry_zone": m15_entry.get("entry_price"),
            "stop_loss": m15_entry.get("sl_price"),
            "take_profit": m15_entry.get("tp_price")
        }
    
    def _check_trend(self, data: Dict, timeframe: str) -> str:
        """Check trend direction on given timeframe using EMA"""
        ema_fast = data.get("ema_cross", {}).get("ema_9")
        ema_slow = data.get("ema_cross", {}).get("ema_21")
        ema_trend = data.get("ema_trend", {}).get("ema_200") if isinstance(data.get("ema_trend"), dict) else data.get("ema_200")
        current_price = data.get("current_price")
        
        if not all([ema_fast, ema_slow, current_price]):
            return "NONE"
        
        # Bull trend: price above EMA 200, EMA 9 > EMA 21
        if current_price > ema_trend and ema_fast > ema_slow:
            return "BUY"
        
        # Bear trend: price below EMA 200, EMA 9 < EMA 21
        if current_price < ema_trend and ema_fast < ema_slow:
            return "SELL"
        
        return "NONE"
    
    def _check_entry_zone(self, m15_data: Dict, trend_direction: str) -> Dict:
        """
        Check if price is in entry zone (pullback to EMA 20)
        """
        ema_20 = m15_data.get("ema_cross", {}).get("ema_20") if isinstance(m15_data.get("ema_cross"), dict) else m15_data.get("ema_20")
        current_price = m15_data.get("current_price")
        atr = m15_data.get("atr")
        
        if not all([ema_20, current_price]):
            return {"valid": False, "confidence": 0, "reason": "No EMA 20 data"}
        
        # Calculate distance from EMA 20 in pips
        price_diff = abs(current_price - ema_20)
        digits = 5 if current_price < 100 else 3
        diff_pips = price_diff * (10 ** (digits - 4))
        
        # For BUY: price should be close to or slightly below EMA 20 (pullback)
        # For SELL: price should be close to or slightly above EMA 20 (pullback)
        if trend_direction == "BUY":
            # Price should be at or below EMA 20 for buy setup
            if current_price > ema_20:
                return {"valid": False, "confidence": 20, "reason": f"Price not in pullback (above EMA20)"}
            
            # Check if pullback is not too deep
            if diff_pips > self.pullback_max_pips:
                return {"valid": False, "confidence": 10, "reason": f"Pullback too deep ({diff_pips:.1f} pips)"}
        
        elif trend_direction == "SELL":
            # Price should be at or above EMA 20 for sell setup
            if current_price < ema_20:
                return {"valid": False, "confidence": 20, "reason": f"Price not in pullback (below EMA20)"}
            
            if diff_pips > self.pullback_max_pips:
                return {"valid": False, "confidence": 10, "reason": f"Pullback too deep ({diff_pips:.1f} pips)"}
        
        # Calculate SL and TP based on ATR
        sl_distance = atr * 3 if atr else 0.0030
        tp_distance = atr * 5 if atr else 0.0050
        
        if trend_direction == "BUY":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance
        
        confidence = max(0, 40 - (diff_pips * 2))  # Closer to EMA = higher confidence
        
        return {
            "valid": True,
            "confidence": confidence,
            "reason": f"Entry zone confirmed ({diff_pips:.1f} pips from EMA20)",
            "entry_price": current_price,
            "sl_price": sl_price,
            "tp_price": tp_price
        }
    
    def _check_confirmation(self, m1_data: Dict, trend_direction: str) -> Dict:
        """
        Check M1 candle for confirmation
        """
        current_price = m1_data.get("current_price")
        m1_ema_9 = m1_data.get("ema_cross", {}).get("ema_9") if isinstance(m1_data.get("ema_cross"), dict) else m1_data.get("ema_9")
        m1_ema_21 = m1_data.get("ema_cross", {}).get("ema_21") if isinstance(m1_data.get("ema_cross"), dict) else m1_data.get("ema_21")
        
        if not all([current_price, m1_ema_9, m1_ema_21]):
            return {"valid": False, "confidence": 0, "reason": "No M1 data"}
        
        # Check if M1 EMAs align with trend
        if trend_direction == "BUY":
            if m1_ema_9 < m1_ema_21:
                return {"valid": False, "confidence": 15, "reason": "M1 bearish alignment"}
            # Check for bullish candle (close > open simulation)
            if m1_data.get("close", current_price) < m1_data.get("open", current_price):
                return {"valid": False, "confidence": 10, "reason": "M1 candle not bullish"}
        
        elif trend_direction == "SELL":
            if m1_ema_9 > m1_ema_21:
                return {"valid": False, "confidence": 15, "reason": "M1 bullish alignment"}
            if m1_data.get("close", current_price) > m1_data.get("open", current_price):
                return {"valid": False, "confidence": 10, "reason": "M1 candle not bearish"}
        
        return {
            "valid": True,
            "confidence": 30,
            "reason": f"M1 confirmation: {trend_direction} aligned"
        }


class TrendFollowingStrategy:
    """
    Simple trend following using EMA crossover
    AI filters out ranging markets
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        try:
            from strategies.strategy_config import load_configs
            saved = load_configs().get("trend", {})
            self.strategy_config = saved.get("parameters", {})
        except Exception:
            self.strategy_config = {}

        # Load configurable params
        self.ema_fast = int(self._get_param("ema_fast", 9))
        self.ema_slow = int(self._get_param("ema_slow", 21))
        self.min_confidence = int(self._get_param("min_confidence", 50))
        self.atr_sl_multiplier = float(self._get_param("atr_sl_multiplier", 2.5))
        self.atr_tp_multiplier = float(self._get_param("atr_tp_multiplier", 6.0))
        self.min_risk_reward = float(self._get_param("min_risk_reward", 2.0))
        self.rsi_overbought = int(self._get_param("rsi_overbought", 75))
        self.rsi_oversold = int(self._get_param("rsi_oversold", 25))
        self.max_spread_pips = float(self._get_param("max_spread_pips", 5.0))
        self.max_open_trades = int(self._get_param("max_open_trades", 3))
        self.daily_loss_limit_pct = float(self._get_param("daily_loss_limit_pct", 3.0))

    def _get_param(self, name, default):
        if name in self.strategy_config:
            return self.strategy_config[name].get("value", default)
        return default

    def reload_config(self, all_configs):
        saved = all_configs.get("trend", {})
        self.strategy_config = saved.get("parameters", {})
        self.ema_fast = int(self._get_param("ema_fast", 9))
        self.ema_slow = int(self._get_param("ema_slow", 21))
        self.min_confidence = int(self._get_param("min_confidence", 50))
        self.atr_sl_multiplier = float(self._get_param("atr_sl_multiplier", 2.5))
        self.atr_tp_multiplier = float(self._get_param("atr_tp_multiplier", 6.0))
        self.min_risk_reward = float(self._get_param("min_risk_reward", 2.0))
        self.rsi_overbought = int(self._get_param("rsi_overbought", 75))
        self.rsi_oversold = int(self._get_param("rsi_oversold", 25))
        self.max_spread_pips = float(self._get_param("max_spread_pips", 5.0))
        self.max_open_trades = int(self._get_param("max_open_trades", 3))
        self.daily_loss_limit_pct = float(self._get_param("daily_loss_limit_pct", 3.0))

    def check_setup(self, market_data: Dict) -> Dict:
        """Check for EMA crossover setup"""
        m15 = market_data.get('M15', {})
        m5 = market_data.get('M5', {})
        
        if not m15 or not m5:
            return {"signal": "HOLD", "confidence": 0, "reason": "No data"}
        
        m15_cross = m15.get("ema_cross", {})
        m5_cross = m5.get("ema_cross", {})
        
        m15_bull = m15_cross.get("signal") == "BULLISH" if m15_cross else False
        m15_bear = m15_cross.get("signal") == "BEARISH" if m15_cross else False
        m5_bull = m5_cross.get("signal") == "BULLISH" if m5_cross else False
        m5_bear = m5_cross.get("signal") == "BEARISH" if m5_cross else False
        
        # Strong buy: M15 bull + M5 bull
        if m15_bull and m5_bull:
            return {"signal": "BUY", "confidence": 80, "reason": "M15+M5 bullish"}
        
        # Strong sell: M15 bear + M5 bear
        if m15_bear and m5_bear:
            return {"signal": "SELL", "confidence": 80, "reason": "M15+M5 bearish"}
        
        # Weak signals
        if m15_bull or m5_bull:
            return {"signal": "BUY", "confidence": 40, "reason": "Partial bullish signal"}
        if m15_bear or m5_bear:
            return {"signal": "SELL", "confidence": 40, "reason": "Partial bearish signal"}
        
        return {"signal": "HOLD", "confidence": 0, "reason": "No signal"}


class StrategyManager:
    """Manages multiple strategies and selects best"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.strategies = {
            "scalp": ScalpStrategy(config),
            "trend": TrendFollowingStrategy(config)
        }
        self.active_strategy = self.config.get("strategy", {}).get("active", "scalp")
    
    def get_signal(self, market_data: Dict) -> Dict:
        """Get signal from active strategy"""
        strategy = self.strategies.get(self.active_strategy, self.strategies["scalp"])
        return strategy.check_setup(market_data)
    
    def get_all_signals(self, market_data: Dict) -> Dict:
        """Get signals from all strategies for analysis"""
        signals = {}
        for name, strategy in self.strategies.items():
            signals[name] = strategy.check_setup(market_data)
        return signals

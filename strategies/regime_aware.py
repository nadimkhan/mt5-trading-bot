"""
Regime-Aware Trading Strategy
==========================================
Detects market regime (BULL/BEAR/SIDEWAYS) and applies pre-researched
parameter sets optimized for each regime.

Uses the engine's existing indicator functions which compute the LATEST value
from a list of bars.
"""
import logging
from typing import Dict, List

from engine.indicators import (
    calculate_adx, calculate_atr, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_ema
)

logger = logging.getLogger(__name__)


# =============================================================================
# PRE-RESEARCHED PARAMETER SETS
# =============================================================================

REGIME_PARAMS = {
    "BULL": {
        "name": "Bull Trend Pullback",
        "description": "Buy pullbacks to fast EMA in uptrends",
        "ema_fast": 21,
        "ema_slow": 50,
        "rsi_period": 14,
        "rsi_buy_min": 40,
        "rsi_buy_max": 60,
        "use_macd": True,
        "adx_min": 25,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 4.0,
        "volume_filter": True,
        "volume_mult": 1.0,
        "expected_win_rate": 50,
        "expected_rr": 2.0,
    },
    "BEAR": {
        "name": "Bear Trend Pullback",
        "description": "Sell rallies to fast EMA in downtrends",
        "ema_fast": 21,
        "ema_slow": 50,
        "rsi_period": 14,
        "rsi_sell_min": 40,
        "rsi_sell_max": 60,
        "use_macd": True,
        "adx_min": 25,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 4.0,
        "volume_filter": True,
        "volume_mult": 1.0,
        "expected_win_rate": 50,
        "expected_rr": 2.0,
    },
    "SIDEWAYS": {
        "name": "Mean Reversion",
        "description": "Fade extremes in ranging markets using Bollinger Bands",
        "bb_period": 20,
        "bb_stddev": 2.0,
        "rsi_period": 14,
        "rsi_buy_max": 30,
        "rsi_sell_min": 70,
        "adx_max": 20,
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 1.5,
        "volume_filter": False,
        "expected_win_rate": 65,
        "expected_rr": 1.5,
    },
    "VOLATILE": {
        "name": "No Trade",
        "description": "Skip - high volatility, no edge for retail",
        "skip": True,
    },
}


class RegimeAwareStrategy:
    """Strategy that detects market regime and applies pre-researched params."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.current_regime = "UNKNOWN"
        logger.info("RegimeAwareStrategy initialized")

    def _detect_regime(self, closes: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Detect current market regime from indicator values."""
        if len(closes) < 30:
            return {"regime": "UNKNOWN", "reason": "Insufficient data"}

        adx_data = calculate_adx(highs, lows, closes, 14)
        if not adx_data:
            return {"regime": "UNKNOWN", "reason": "ADX not ready"}

        adx = adx_data.get("adx", 0)
        plus_di = adx_data.get("plus_di", 0)
        minus_di = adx_data.get("minus_di", 0)

        current_atr = calculate_atr(highs, lows, closes, 14)
        if current_atr is None or current_atr <= 0:
            return {"regime": "UNKNOWN", "reason": "ATR not ready"}

        # Trending?
        if adx >= 25:
            if plus_di > minus_di:
                return {
                    "regime": "BULL", "direction": "up",
                    "reason": f"ADX={adx:.1f} (trending up), +DI={plus_di:.1f} > -DI={minus_di:.1f}",
                    "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
                }
            else:
                return {
                    "regime": "BEAR", "direction": "down",
                    "reason": f"ADX={adx:.1f} (trending down), -DI={minus_di:.1f} > +DI={plus_di:.1f}",
                    "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
                }

        # Ranging
        if adx < 20:
            return {
                "regime": "SIDEWAYS", "direction": None,
                "reason": f"ADX={adx:.1f} < 20 (no trend)",
                "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            }

        # Transitional
        if plus_di > minus_di:
            return {
                "regime": "BULL", "direction": "up",
                "reason": f"ADX={adx:.1f} (transitional), +DI dominant",
                "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            }
        return {
            "regime": "BEAR", "direction": "down",
            "reason": f"ADX={adx:.1f} (transitional), -DI dominant",
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
        }

    def _evaluate_bull(self, closes, highs, lows, volumes, p) -> Dict:
        """BUY signal in BULL regime (buy pullback to EMA)."""
        min_bars = max(p["ema_slow"], p["rsi_period"]) + 5
        if len(closes) < min_bars:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

        ema_fast = calculate_ema(closes, p["ema_fast"])
        ema_slow = calculate_ema(closes, p["ema_slow"])
        rsi = calculate_rsi(closes, p["rsi_period"])
        atr = calculate_atr(highs, lows, closes, 14)

        if not all([ema_fast, ema_slow, rsi, atr]):
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        current_price = closes[-1]

        if current_price <= ema_slow:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"Price {current_price:.5f} below slow EMA {ema_slow:.5f}"}

        distance = abs(current_price - ema_fast) / atr
        if distance > 1.5:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"Too far from EMA ({distance:.1f} ATRs)"}

        if rsi < p["rsi_buy_min"] or rsi > p["rsi_buy_max"]:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI={rsi:.1f} not in pullback zone ({p['rsi_buy_min']}-{p['rsi_buy_max']})"}

        if p.get("use_macd"):
            macd = calculate_macd(closes)
            if macd and macd.get("macd", 0) <= macd.get("signal", 0):
                return {"signal": "HOLD", "confidence": 0, "reason": "MACD not bullish"}

        if p.get("volume_filter") and volumes and len(volumes) >= 20:
            vol_avg = sum(volumes[-20:]) / 20
            if vol_avg > 0 and volumes[-1] < vol_avg * p.get("volume_mult", 1.0):
                return {"signal": "HOLD", "confidence": 0, "reason": "Volume too low"}

        sl = current_price - atr * p["atr_sl_mult"]
        tp = current_price + atr * p["atr_tp_mult"]
        return {
            "signal": "BUY",
            "confidence": min(85, 50 + int(p.get("expected_win_rate", 50) / 2)),
            "sl": sl, "tp": tp, "entry_zone": current_price,
            "reason": f"BULL pullback: RSI={rsi:.1f}, EMA dist={distance:.1f} ATR"
        }

    def _evaluate_bear(self, closes, highs, lows, volumes, p) -> Dict:
        """SELL signal in BEAR regime (sell rally to EMA)."""
        min_bars = max(p["ema_slow"], p["rsi_period"]) + 5
        if len(closes) < min_bars:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

        ema_fast = calculate_ema(closes, p["ema_fast"])
        ema_slow = calculate_ema(closes, p["ema_slow"])
        rsi = calculate_rsi(closes, p["rsi_period"])
        atr = calculate_atr(highs, lows, closes, 14)

        if not all([ema_fast, ema_slow, rsi, atr]):
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        current_price = closes[-1]

        if current_price >= ema_slow:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"Price {current_price:.5f} above slow EMA {ema_slow:.5f}"}

        distance = abs(current_price - ema_fast) / atr
        if distance > 1.5:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"Too far from EMA ({distance:.1f} ATRs)"}

        if rsi < p["rsi_sell_min"] or rsi > p["rsi_sell_max"]:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI={rsi:.1f} not in rally zone ({p['rsi_sell_min']}-{p['rsi_sell_max']})"}

        if p.get("use_macd"):
            macd = calculate_macd(closes)
            if macd and macd.get("macd", 0) >= macd.get("signal", 0):
                return {"signal": "HOLD", "confidence": 0, "reason": "MACD not bearish"}

        if p.get("volume_filter") and volumes and len(volumes) >= 20:
            vol_avg = sum(volumes[-20:]) / 20
            if vol_avg > 0 and volumes[-1] < vol_avg * p.get("volume_mult", 1.0):
                return {"signal": "HOLD", "confidence": 0, "reason": "Volume too low"}

        sl = current_price + atr * p["atr_sl_mult"]
        tp = current_price - atr * p["atr_tp_mult"]
        return {
            "signal": "SELL",
            "confidence": min(85, 50 + int(p.get("expected_win_rate", 50) / 2)),
            "sl": sl, "tp": tp, "entry_zone": current_price,
            "reason": f"BEAR rally: RSI={rsi:.1f}, EMA dist={distance:.1f} ATR"
        }

    def _evaluate_sideways(self, closes, highs, lows, volumes, p) -> Dict:
        """Mean reversion signal in SIDEWAYS regime (Bollinger extremes)."""
        if len(closes) < p["bb_period"] + 5:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

        bb = calculate_bollinger_bands(closes, p["bb_period"], p["bb_stddev"])
        rsi = calculate_rsi(closes, p["rsi_period"])
        atr = calculate_atr(highs, lows, closes, 14)

        if not all([bb, rsi, atr]):
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        bb_upper = bb.get("upper", 0)
        bb_lower = bb.get("lower", 0)
        current_price = closes[-1]

        # Buy at lower band
        if current_price <= bb_lower and rsi < p["rsi_buy_max"]:
            sl = current_price - atr * p["atr_sl_mult"]
            tp = current_price + atr * p["atr_tp_mult"]
            return {
                "signal": "BUY",
                "confidence": min(85, 50 + int(p.get("expected_win_rate", 65) / 2)),
                "sl": sl, "tp": tp, "entry_zone": current_price,
                "reason": f"SIDEWAYS BUY at lower BB, RSI={rsi:.1f}"
            }

        # Sell at upper band
        if current_price >= bb_upper and rsi > p["rsi_sell_min"]:
            sl = current_price + atr * p["atr_sl_mult"]
            tp = current_price - atr * p["atr_tp_mult"]
            return {
                "signal": "SELL",
                "confidence": min(85, 50 + int(p.get("expected_win_rate", 65) / 2)),
                "sl": sl, "tp": tp, "entry_zone": current_price,
                "reason": f"SIDEWAYS SELL at upper BB, RSI={rsi:.1f}"
            }

        return {"signal": "HOLD", "confidence": 0,
                "reason": f"SIDEWAYS but not at BB extreme (price={current_price:.5f})"}

    def check_entry(self, market_data: Dict) -> Dict:
        """Main entry: detect regime, then evaluate with regime-specific params."""
        closes = market_data.get("closes", [])
        highs = market_data.get("highs", [])
        lows = market_data.get("lows", [])
        volumes = market_data.get("volumes", market_data.get("tick_volumes", []))

        if len(closes) < 30:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data",
                    "regime": "UNKNOWN"}

        # Step 1: Detect regime
        regime_info = self._detect_regime(closes, highs, lows)
        regime = regime_info["regime"]
        self.current_regime = regime

        # Step 2: Check volatile (skip)
        if regime == "VOLATILE":
            return {
                "signal": "HOLD", "confidence": 0,
                "reason": "Volatile market - skipping",
                "regime": regime, "regime_reason": regime_info["reason"],
                "adx": regime_info.get("adx", 0)
            }

        if regime == "UNKNOWN":
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"Regime unknown", "regime": regime}

        # Step 3: Get params for this regime
        params = REGIME_PARAMS.get(regime, {})

        # Step 4: Evaluate
        if regime == "BULL":
            result = self._evaluate_bull(closes, highs, lows, volumes, params)
        elif regime == "BEAR":
            result = self._evaluate_bear(closes, highs, lows, volumes, params)
        elif regime == "SIDEWAYS":
            result = self._evaluate_sideways(closes, highs, lows, volumes, params)
        else:
            result = {"signal": "HOLD", "confidence": 0, "reason": f"No logic for {regime}"}

        # Add metadata
        result["regime"] = regime
        result["regime_reason"] = regime_info["reason"]
        result["adx"] = regime_info.get("adx", 0)
        result["plus_di"] = regime_info.get("plus_di", 0)
        result["minus_di"] = regime_info.get("minus_di", 0)
        result["params_used"] = params.get("name", "Unknown")
        return result

    # Alias for engine compatibility
    def check_setup(self, market_data: Dict) -> Dict:
        return self.check_entry(market_data)

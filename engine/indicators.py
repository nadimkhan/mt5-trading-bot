"""
Technical Indicators - Calculate RSI, MACD, Bollinger, ATR, etc.
"""
import numpy as np
from datetime import datetime


def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return None
        
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow + signal:
        return None
        
    # Calculate EMAs
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None
        
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    # Simplified: use last 9 values
    macd_values = []
    for i in range(-signal, 0):
        if len(prices) >= slow + signal + abs(i):
            e_f = calculate_ema(prices[:i], fast)
            e_s = calculate_ema(prices[:i], slow)
            if e_f and e_s:
                macd_values.append(e_f - e_s)
    
    if len(macd_values) < 3:
        signal_line = macd_line
    else:
        signal_line = np.mean(macd_values[-3:])
    
    histogram = macd_line - signal_line
    
    return {
        "macd": round(macd_line, 5),
        "signal": round(signal_line, 5),
        "histogram": round(histogram, 5),
        "trend": "BULLISH" if histogram > 0 else "BEARISH"
    }


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
        
    prices_array = np.array(prices)
    multiplier = 2 / (period + 1)
    
    # Start with SMA
    ema = np.mean(prices_array[:period])
    
    # Calculate EMA
    for price in prices_array[period:]:
        ema = (price - ema) * multiplier + ema
        
    return ema


def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return round(np.mean(prices[-period:]), 5)


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None
        
    prices_array = np.array(prices[-period:])
    sma = np.mean(prices_array)
    std = np.std(prices_array)
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return {
        "upper": round(upper, 5),
        "middle": round(sma, 5),
        "lower": round(lower, 5),
        "width": round(upper - lower, 5)
    }


def calculate_atr(highs, lows, closes, period=14):
    """Calculate ATR (Average True Range)"""
    if len(highs) < period + 1:
        return None
        
    tr_list = []
    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return None
        
    atr = np.mean(tr_list[-period:])
    return round(atr, 5)


def calculate_support_resistance(prices, lookback=20):
    """Find support and resistance levels"""
    if len(prices) < lookback:
        return {"support": None, "resistance": None}
        
    recent_prices = prices[-lookback:]
    
    # Simple approach: use min/max
    support = round(min(recent_prices), 5)
    resistance = round(max(recent_prices), 5)
    
    # Find pivot points for more accuracy
    pivots = []
    for i in range(2, len(recent_prices) - 2):
        if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i-2]:
            if recent_prices[i] > recent_prices[i+1] and recent_prices[i] > recent_prices[i+2]:
                pivots.append(("R", recent_prices[i]))
        elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i-2]:
            if recent_prices[i] < recent_prices[i+1] and recent_prices[i] < recent_prices[i+2]:
                pivots.append(("S", recent_prices[i]))
    
    # Get most recent resistance and support pivots
    resistance_levels = [p[1] for p in pivots if p[0] == "R"]
    support_levels = [p[1] for p in pivots if p[0] == "S"]
    
    current_price = prices[-1]
    
    # Find nearest resistance above and support below
    nearest_resistance = None
    nearest_support = None
    
    for level in resistance_levels:
        if level > current_price:
            if nearest_resistance is None or level < nearest_resistance:
                nearest_resistance = round(level, 5)
                
    for level in support_levels:
        if level < current_price:
            if nearest_support is None or level > nearest_support:
                nearest_support = round(level, 5)
    
    return {
        "support": nearest_support if nearest_support else support,
        "resistance": nearest_resistance if nearest_resistance else resistance,
        "pivots": pivots[-5:] if pivots else []  # Last 5 pivot points
    }


def calculate_trend(prices, short_period=20, long_period=50):
    """Determine trend direction"""
    if len(prices) < long_period:
        return "UNKNOWN"

    short_ma = calculate_sma(prices, short_period)
    long_ma = calculate_sma(prices, long_period)
    current_price = prices[-1]

    if current_price > short_ma and short_ma > long_ma:
        return "BULL_TREND"
    elif current_price < short_ma and short_ma < long_ma:
        return "BEAR_TREND"
    else:
        return "SIDEWAYS"


def calculate_ema_cross(prices_closes):
    """Calculate EMA 9 and 21 crossover signals for scalping"""
    if len(prices_closes) < 21:
        return None

    ema_9 = calculate_ema(prices_closes, 9)
    ema_21 = calculate_ema(prices_closes, 21)

    if ema_9 is None or ema_21 is None:
        return None

    # Current position
    if ema_9 > ema_21:
        signal = "BULLISH"
    elif ema_9 < ema_21:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "ema_9": round(ema_9, 5),
        "ema_21": round(ema_21, 5),
        "signal": signal,
        "distance_pips": round(abs(ema_9 - ema_21) * 10000, 1) if ema_9 and ema_21 else 0
    }


def calculate_adx(highs, lows, closes, period=14):
    """Calculate ADX (Average Directional Index) - measures trend strength"""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    
    # Calculate True Range and Directional Movement
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    
    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i-1]
        prev_low = lows[i-1]
        prev_close = closes[i-1]
        
        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
        
        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            plus_dm_list.append(up_move)
        else:
            plus_dm_list.append(0)
            
        if down_move > up_move and down_move > 0:
            minus_dm_list.append(down_move)
        else:
            minus_dm_list.append(0)
    
    if len(tr_list) < period:
        return None
    
    # Smooth with EMA
    atr = np.mean(tr_list[-period:])
    
    plus_dm_smooth = np.mean(plus_dm_list[-period:])
    minus_dm_smooth = np.mean(minus_dm_list[-period:])
    
    if atr == 0:
        return None
    
    # Calculate DI
    plus_di = (plus_dm_smooth / atr) * 100
    minus_di = (minus_dm_smooth / atr) * 100
    
    # Calculate DX
    di_sum = plus_di + minus_di
    if di_sum == 0:
        dx = 0
    else:
        dx = abs(plus_di - minus_di) / di_sum * 100
    
    # ADX is the smoothed DX
    adx = dx  # Simplified - in production would use Wilder smoothing
    
    return {
        "adx": round(adx, 2),
        "plus_di": round(plus_di, 2),
        "minus_di": round(minus_di, 2),
        "trend_strength": "STRONG" if adx > 25 else "WEAK" if adx < 20 else "MODERATE"
    }


def calculate_atr_percentile(symbol, atr_current, lookback=100):
    """
    Calculate ATR percentile - what % of recent ATR is current ATR?
    High percentile = high volatility, Low = low volatility
    Used to filter out dead markets
    """
    # This would need historical ATR data
    # Placeholder - returns percentile based on current volatility
    return 50  # Default to middle


def detect_market_regime(prices, highs, lows, closes, adx_period=14, atr_period=14):
    """
    Detect market regime: TRENDING, RANGING, or VOLATILE
    - TRENDING: ADX > 25, clear direction
    - RANGING: ADX < 20, no clear trend
    - VOLATILE: ATR is high percentile
    """
    adx_data = calculate_adx(highs, lows, closes, adx_period)
    atr_data = calculate_atr(highs, lows, closes, atr_period)
    
    if not adx_data:
        return {"regime": "UNKNOWN", "reason": "Insufficient data"}
    
    adx = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]
    
    # Determine regime based on ADX
    if adx < 20:
        regime = "RANGING"
        reason = f"ADX={adx} < 20 - no trend"
    elif adx < 25:
        regime = "TRANSITIONAL"
        reason = f"ADX={adx} < 25 - weak trend"
    else:
        regime = "TRENDING"
        reason = f"ADX={adx} > 25 - strong trend"
    
    # Determine direction if trending
    direction = None
    if regime == "TRENDING":
        if plus_di > minus_di:
            direction = "BULL"
        else:
            direction = "BEAR"
    
    return {
        "regime": regime,
        "direction": direction,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "atr": atr_data,
        "reason": reason
    }


def is_tradeable_regime(regime_data, strategy_type="trend"):
    """
    Filter if market is tradeable based on regime and strategy type.
    
    For TREND strategies: require TRENDING regime
    For MEAN_REVERSION: require RANGING regime
    """
    regime = regime_data.get("regime", "UNKNOWN")
    adx = regime_data.get("adx", 0)
    
    if strategy_type == "trend":
        # Trend strategies need trending market
        if regime == "RANGING":
            return False, "Market is ranging - trend strategies disabled"
        if regime == "TRANSITIONAL":
            return False, "Market is transitional - waiting for confirmation"
        if adx < 25:
            return False, f"ADX={adx} too weak for trend trading"
        return True, "Trending market confirmed"
    
    elif strategy_type == "mean_reversion":
        # Mean reversion needs ranging market
        if regime == "TRENDING":
            return False, "Market is trending - mean reversion disabled"
        if adx > 30:
            return False, f"ADX={adx} too strong for mean reversion"
        return True, "Ranging market confirmed"
    
    return True, "Regime check passed"
    """Calculate price change over periods"""
    if len(prices) < periods + 1:
        return None
        
    change = prices[-1] - prices[-(periods + 1)]
    percent = (change / prices[-(periods + 1)]) * 100
    
    return {
        "change": round(change, 5),
        "percent": round(percent, 2),
        "direction": "UP" if change > 0 else "DOWN"
    }


def analyze_market(prices, highs, lows, timeframe="H1"):
    """Complete market analysis"""
    if len(prices) < 50:
        return None

    closes = prices

    # RSI
    rsi = calculate_rsi(closes, 14)
    rsi_7 = calculate_rsi(closes, 7)  # Faster RSI for scalping

    # MACD
    macd = calculate_macd(closes)

    # Bollinger Bands
    bollinger = calculate_bollinger_bands(closes, period=20, std_dev=2)

    # ATR
    atr = calculate_atr(highs, lows, closes)

    # Support/Resistance
    sr = calculate_support_resistance(closes)

    # Trend
    trend = calculate_trend(closes)

    # EMA Crossover for scalping
    ema_cross = calculate_ema_cross(closes)

    # Price change
    change_1h = calculate_price_change(closes, 1)
    change_4h = calculate_price_change(closes, 4)

    # Current candle info
    current_bar = {
        "open": prices[-1],
        "high": max(highs[-5:]),
        "low": min(lows[-5:]),
        "close": prices[-1]
    }

    # Determine momentum
    if rsi and rsi > 70:
        rsi_signal = "OVERBOUGHT"
    elif rsi and rsi < 30:
        rsi_signal = "OVERSOLD"
    elif rsi and rsi > 55:
        rsi_signal = "BULLISH"
    elif rsi and rsi < 45:
        rsi_signal = "BEARISH"
    else:
        rsi_signal = "NEUTRAL"

    # Overall assessment
    bullish_signals = 0
    bearish_signals = 0

    if trend == "BULL_TREND":
        bullish_signals += 2
    elif trend == "BEAR_TREND":
        bearish_signals += 2

    if macd and macd["histogram"] > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if rsi_signal in ["OVERSOLD", "BULLISH"]:
        bullish_signals += 1
    elif rsi_signal in ["OVERBOUGHT", "BEARISH"]:
        bearish_signals += 1

    # EMA cross adds signal weight
    if ema_cross:
        if ema_cross["signal"] == "BULLISH":
            bullish_signals += 2
        elif ema_cross["signal"] == "BEARISH":
            bearish_signals += 2

    if bullish_signals > bearish_signals:
        overall = "BULLISH"
        confidence = min(100, 50 + ((bullish_signals - bearish_signals) * 15))
    elif bearish_signals > bullish_signals:
        overall = "BEARISH"
        confidence = min(100, 50 + ((bearish_signals - bullish_signals) * 15))
    else:
        overall = "NEUTRAL"
        confidence = 50

    return {
        "timeframe": timeframe,
        "current_price": round(prices[-1], 5),
        "trend": trend,
        "rsi": rsi,
        "rsi_7": rsi_7,
        "rsi_signal": rsi_signal,
        "macd": macd,
        "bollinger": bollinger,
        "atr": atr,
        "ema_cross": ema_cross,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "change_1h": change_1h,
        "change_4h": change_4h,
        "overall": overall,
        "confidence": confidence,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "timestamp": datetime.now()
    }


# Test
if __name__ == "__main__":
    # Generate sample data
    prices = [1.12000 + i * 0.0001 for i in range(100)]
    highs = [p + 0.0002 for p in prices]
    lows = [p - 0.0002 for p in prices]
    
    print("RSI:", calculate_rsi(prices))
    print("MACD:", calculate_macd(prices))
    print("Bollinger:", calculate_bollinger_bands(prices))
    print("ATR:", calculate_atr(highs, lows, prices))
    print("Support/Resistance:", calculate_support_resistance(prices))
    print("Trend:", calculate_trend(prices))
    print("\nFull Analysis:", analyze_market(prices, highs, lows))

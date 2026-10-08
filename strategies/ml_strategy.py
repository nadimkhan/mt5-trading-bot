"""
ML-Discovered Strategy - Uses parameters from genetic algorithm search.

This strategy uses FIXED parameters discovered through backtesting
(no AI tokens, deterministic, can be re-validated on new data).

Workflow:
1. Run GeneticSearcher.run_search() to discover parameters
2. Best genome is saved to DB
3. MLStrategy loads the best genome at startup
4. Uses those fixed parameters to generate signals
"""
import logging
from typing import Dict
from strategies.ml_searcher import GeneticSearcher, StrategyGenome

logger = logging.getLogger(__name__)


class MLStrategy:
    """
    Strategy using parameters discovered via genetic algorithm.
    Falls back to default parameters if no genome is found.
    """

    def __init__(self, config: dict = None, symbol: str = None, timeframes: dict = None):
        self.config = config or {}
        self.symbol = symbol
        # Get timeframes from config or use sensible defaults for ML
        if timeframes is None:
            timeframes = self.config.get('timeframes', {})
        self.timeframes = {
            'trend': timeframes.get('trend', 'H4'),
            'entry': timeframes.get('entry', 'M15'),
            'confirm': timeframes.get('confirm', 'M5')
        }
        # Try to load best genome from DB
        self.genome = GeneticSearcher.load_best_genome(symbol)
        if self.genome:
            logger.info(f"Loaded ML genome for {symbol}: PF={self.genome.profit_factor:.2f}, "
                        f"trades={self.genome.total_trades}, "
                        f"WR={self.genome.win_rate:.1f}%, "
                        f"params=[EMA{self.genome.ema_fast}/{self.genome.ema_slow}]")
            self.use_ml_params = True
        else:
            logger.info(f"No ML genome found for {symbol}, using defaults")
            self.genome = StrategyGenome()
            self.use_ml_params = False

    def check_setup(self, market_data: Dict) -> Dict:
        """
        Check for entry using ML-discovered parameters.
        Uses the configured timeframes (trend, entry, confirm) instead of hardcoded ones.
        Returns: signal dict with 'signal', 'confidence', 'reason'
        """
        # Use the configured ENTRY timeframe for signals
        entry_tf = self.timeframes['entry']
        confirm_tf = self.timeframes['confirm']
        trend_tf = self.timeframes['trend']

        # Get data from configured timeframes
        trend_data = market_data.get(trend_tf, {})
        entry_data = market_data.get(entry_tf, {})
        confirm_data = market_data.get(confirm_tf, {})

        if not entry_data:
            return {"signal": "HOLD", "confidence": 0, "reason": f"No {entry_tf} data"}

        closes = entry_data.get('closes', [])
        highs = entry_data.get('highs', [])
        lows = entry_data.get('lows', [])

        if len(closes) < max(self.genome.ema_slow, self.genome.rsi_period) + 10:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

        # Compute indicators on ENTRY timeframe
        ema_fast = self._ema(closes, self.genome.ema_fast)
        ema_slow = self._ema(closes, self.genome.ema_slow)
        rsi = self._rsi(closes, self.genome.rsi_period)
        atr = self._atr(highs, lows, closes)
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, self.genome.bb_period, self.genome.bb_stddev)
        # MACD
        macd_line, macd_signal, macd_hist = self._macd(closes, self.genome.macd_fast, self.genome.macd_slow, self.genome.macd_signal)
        # Volume
        volumes = entry_data.get('tick_volumes', entry_data.get('volumes', []))
        vol_sma = self._volume_sma(volumes, 20) if volumes else [None] * len(closes)

        i = len(closes) - 1
        if i < 1 or ema_fast[i] is None or ema_slow[i] is None or rsi[i] is None or atr[i] is None:
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        current_price = closes[i]
        current_rsi = rsi[i]
        current_atr = atr[i]

        # Check trend: EMA fast above slow = bullish, below = bearish
        trend = "BULL" if ema_fast[i] > ema_slow[i] else "BEAR"

        # ATR floor check (don't trade dead markets)
        if current_atr < 0.0001:  # very low ATR
            return {"signal": "HOLD", "confidence": 0, "reason": "Low volatility"}

        # Volume filter
        vol_ok = True
        if self.genome.use_volume_filter and vol_sma[i] is not None and vol_sma[i] > 0 and len(volumes) > 0:
            vol_ok = volumes[i] >= vol_sma[i] * self.genome.volume_min_multiplier
            if not vol_ok:
                return {"signal": "HOLD", "confidence": 0, "reason": f"Low volume ({volumes[i]}/{vol_sma[i]:.0f})"}

        # Check RSI not extreme
        if current_rsi >= self.genome.rsi_overbought:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI overbought ({current_rsi:.1f} >= {self.genome.rsi_overbought})"}
        if current_rsi <= self.genome.rsi_oversold:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI oversold ({current_rsi:.1f} <= {self.genome.rsi_oversold})"}

        # Check for entry signals
        long_signal = False
        short_signal = False
        signal_reasons = []

        # 1. EMA cross + MACD cross
        if self.genome.use_ema_cross and self.genome.use_macd_cross:
            if (ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i]
                and macd_line[i-1] <= macd_signal[i-1] and macd_line[i] > macd_signal[i]
                and trend == "BULL"):
                long_signal = True
                signal_reasons.append(f"EMA{self.genome.ema_fast}/{self.genome.ema_slow}+MACD")
            elif (ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i]
                  and macd_line[i-1] >= macd_signal[i-1] and macd_line[i] < macd_signal[i]
                  and trend == "BEAR"):
                short_signal = True
                signal_reasons.append(f"EMA{self.genome.ema_fast}/{self.genome.ema_slow}+MACD")
        # 2. EMA cross only
        elif self.genome.use_ema_cross:
            if ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i] and trend == "BULL":
                long_signal = True
                signal_reasons.append("EMA-cross")
            elif ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i] and trend == "BEAR":
                short_signal = True
                signal_reasons.append("EMA-cross")
        # 3. MACD cross only
        elif self.genome.use_macd_cross:
            if macd_line[i-1] <= macd_signal[i-1] and macd_line[i] > macd_signal[i] and macd_line[i] > 0:
                long_signal = True
                signal_reasons.append("MACD-cross")
            elif macd_line[i-1] >= macd_signal[i-1] and macd_line[i] < macd_signal[i] and macd_line[i] < 0:
                short_signal = True
                signal_reasons.append("MACD-cross")
        # 4. Bollinger band bounce (mean reversion)
        elif self.genome.use_bb_bounce:
            if current_price <= bb_lower[i] and current_rsi < 35:
                long_signal = True
                signal_reasons.append("BB-bounce")
            elif current_price >= bb_upper[i] and current_rsi > 65:
                short_signal = True
                signal_reasons.append("BB-bounce")

        if long_signal:
            sl = current_price - current_atr * self.genome.atr_sl_multiplier
            tp = current_price + current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "BUY",
                "confidence": confidence,
                "reason": f"ML({','.join(signal_reasons)}): PF={self.genome.profit_factor:.2f}, RSI={current_rsi:.1f}",
                "entry_zone": current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "ml_genome": self.genome.id
            }
        elif short_signal:
            sl = current_price + current_atr * self.genome.atr_sl_multiplier
            tp = current_price - current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "SELL",
                "confidence": confidence,
                "reason": f"ML({','.join(signal_reasons)}): PF={self.genome.profit_factor:.2f}, RSI={current_rsi:.1f}",
                "entry_zone": current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "ml_genome": self.genome.id
            }
        return {"signal": "HOLD", "confidence": 0, "reason": "No crossover"}

    def _ema(self, prices, period):
        if len(prices) < period:
            return [None] * len(prices)
        result = [None] * len(prices)
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        result[period - 1] = ema
        for i in range(period, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema
            result[i] = ema
        return result

    def _rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return [None] * len(prices)
        result = [None] * len(prices)
        gains = []
        losses = []
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            gains.append(max(0, diff))
            losses.append(max(0, -diff))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        result[period] = 100 - (100 / (1 + rs))
        for i in range(period + 1, len(prices)):
            diff = prices[i] - prices[i-1]
            gain = max(0, diff)
            loss = max(0, -diff)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            result[i] = 100 - (100 / (1 + rs))
        return result

    def _atr(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return [None] * len(highs)
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1]))
            trs.append(tr)
        result = [None] * len(highs)
        if len(trs) >= period:
            atr = sum(trs[:period]) / period
            result[period] = atr
            for i in range(period + 1, len(trs)):
                atr = (atr * (period - 1) + trs[i]) / period
                result[i + 1] = atr
        return result

    def _bollinger_bands(self, prices, period, stddev):
        if len(prices) < period:
            return [None] * len(prices), [None] * len(prices), [None] * len(prices)
        upper = [None] * len(prices)
        middle = [None] * len(prices)
        lower = [None] * len(prices)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            mean = sum(window) / period
            variance = sum((p - mean) ** 2 for p in window) / period
            std = variance ** 0.5
            middle[i] = mean
            upper[i] = mean + stddev * std
            lower[i] = mean - stddev * std
        return upper, middle, lower

    def _macd(self, prices, fast_period, slow_period, signal_period):
        ema_fast = self._ema(prices, fast_period)
        ema_slow = self._ema(prices, slow_period)
        macd_line = [None] * len(prices)
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line[i] = ema_fast[i] - ema_slow[i]
        macd_values = [v if v is not None else 0 for v in macd_line]
        signal_line_raw = self._ema(macd_values, signal_period)
        signal_line = [None] * len(prices)
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line_raw[i] is not None:
                signal_line[i] = signal_line_raw[i]
        return macd_line, signal_line, [None] * len(prices)

    def _volume_sma(self, volumes, period=20):
        if not volumes or len(volumes) < period:
            return [None] * len(volumes) if volumes else [None]
        result = [None] * len(volumes)
        for i in range(period - 1, len(volumes)):
            result[i] = sum(volumes[i - period + 1:i + 1]) / period
        return result

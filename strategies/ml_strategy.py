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

        # Compute indicators on ENTRY timeframe (simplified 8-param)
        ema_fast = self._ema(closes, self.genome.ema_fast)
        ema_slow = self._ema(closes, self.genome.ema_slow)
        rsi = self._rsi(closes, self.genome.rsi_period)
        atr = self._atr(highs, lows, closes)
        # Volume
        volumes = entry_data.get('tick_volumes', entry_data.get('volumes', []))
        vol_sma = self._volume_sma(volumes, 20) if volumes else [None] * len(closes)

        i = len(closes) - 1
        if i < 1 or ema_fast[i] is None or ema_slow[i] is None or rsi[i] is None or atr[i] is None:
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        current_price = closes[i]
        current_rsi = rsi[i]
        current_atr = atr[i]

        # ATR floor check (don't trade dead markets)
        if current_atr < 0.0001:
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

        # ENTRY: EMA crossover (only signal)
        if ema_fast[i-1] is None or ema_slow[i-1] is None:
            return {"signal": "HOLD", "confidence": 0, "reason": "EMA not ready"}
        bullish_ema = (ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i])
        bearish_ema = (ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i])

        if bullish_ema and vol_ok:
            sl = current_price - current_atr * self.genome.atr_sl_multiplier
            tp = current_price + current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "BUY",
                "confidence": confidence,
                "reason": f"ML(EMA{self.genome.ema_fast}/{self.genome.ema_slow}): PF={self.genome.profit_factor:.2f}, RSI={current_rsi:.1f}",
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
        elif bearish_ema and vol_ok:
            sl = current_price + current_atr * self.genome.atr_sl_multiplier
            tp = current_price - current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "SELL",
                "confidence": confidence,
                "reason": f"ML(EMA{self.genome.ema_fast}/{self.genome.ema_slow}): PF={self.genome.profit_factor:.2f}, RSI={current_rsi:.1f}",
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

    def _ichimoku(self, highs, lows, closes, tenkan=9, kijun=26, senkou_b=52):
        """Ichimoku Cloud"""
        n = len(highs)
        r_tenkan = [None] * n
        r_kijun = [None] * n
        r_senkou_a = [None] * n
        r_senkou_b = [None] * n
        r_chikou = [None] * n
        for i in range(n):
            if i >= tenkan - 1:
                r_tenkan[i] = (max(highs[i - tenkan + 1:i + 1]) + min(lows[i - tenkan + 1:i + 1])) / 2
            if i >= kijun - 1:
                r_kijun[i] = (max(highs[i - kijun + 1:i + 1]) + min(lows[i - kijun + 1:i + 1])) / 2
            if i >= kijun - 1 and r_tenkan[i] is not None and r_kijun[i] is not None:
                if i + 26 < n:
                    r_senkou_a[i + 26] = (r_tenkan[i] + r_kijun[i]) / 2
            if i >= senkou_b - 1:
                if i + 26 < n:
                    r_senkou_b[i + 26] = (max(highs[i - senkou_b + 1:i + 1]) + min(lows[i - senkou_b + 1:i + 1])) / 2
            if i - 26 >= 0:
                r_chikou[i - 26] = closes[i]
        return r_tenkan, r_kijun, r_senkou_a, r_senkou_b, r_chikou

    def _heikin_ashi(self, opens, highs, lows, closes):
        """Heikin Ashi candles"""
        n = len(closes)
        if n == 0:
            return [], [], [], []
        ha_o = [None] * n
        ha_h = [None] * n
        ha_l = [None] * n
        ha_c = [None] * n
        ha_c[0] = (opens[0] + highs[0] + lows[0] + closes[0]) / 4
        ha_o[0] = (opens[0] + closes[0]) / 2
        ha_h[0] = highs[0]
        ha_l[0] = lows[0]
        for i in range(1, n):
            ha_c[i] = (opens[i] + highs[i] + lows[i] + closes[i]) / 4
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
            ha_h[i] = max(highs[i], ha_o[i], ha_c[i])
            ha_l[i] = min(lows[i], ha_o[i], ha_c[i])
        return ha_o, ha_h, ha_l, ha_c

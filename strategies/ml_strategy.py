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

        i = len(closes) - 1
        if i < 1 or ema_fast[i] is None or ema_slow[i] is None or rsi[i] is None or atr[i] is None:
            return {"signal": "HOLD", "confidence": 0, "reason": "Indicators not ready"}

        current_price = closes[i]
        current_rsi = rsi[i]
        current_atr = atr[i]

        # Check trend: EMA fast above slow = bullish, below = bearish
        trend = "BULL" if ema_fast[i] > ema_slow[i] else "BEAR"

        # Check RSI not extreme
        if current_rsi >= self.genome.rsi_overbought:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI overbought ({current_rsi:.1f} >= {self.genome.rsi_overbought})"}
        if current_rsi <= self.genome.rsi_oversold:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": f"RSI oversold ({current_rsi:.1f} <= {self.genome.rsi_oversold})"}

        # Check for EMA crossover (entry signal)
        # Long: fast crosses above slow
        if trend == "BULL" and ema_fast[i-1] <= ema_slow[i-1]:
            # Just crossed up
            sl = current_price - current_atr * self.genome.atr_sl_multiplier
            tp = current_price + current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "BUY",
                "confidence": confidence,
                "reason": f"ML: EMA{self.genome.ema_fast} crossed above EMA{self.genome.ema_slow}, RSI={current_rsi:.1f}",
                "entry_zone": current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "ml_genome": self.genome.id
            }
        # Short: fast crosses below slow
        elif trend == "BEAR" and ema_fast[i-1] >= ema_slow[i-1]:
            sl = current_price + current_atr * self.genome.atr_sl_multiplier
            tp = current_price - current_atr * self.genome.atr_tp_multiplier
            confidence = min(95, 60 + int(self.genome.profit_factor * 10))
            return {
                "signal": "SELL",
                "confidence": confidence,
                "reason": f"ML: EMA{self.genome.ema_fast} crossed below EMA{self.genome.ema_slow}, RSI={current_rsi:.1f}",
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

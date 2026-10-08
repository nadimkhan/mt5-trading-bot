"""
ML Strategy Searcher - Genetic algorithm to discover optimal rule parameters.

Based on the principle (per Algory OS Field Report):
"Start with thousands of random rule sets. Score each one on real price data.
 Keep the best. Mix their rules, change a few settings, and score them again."

Output: A small set of fixed rules that perform well on held-out data.
No tokens, no AI - just numbers.
"""
import json
import random
import logging
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.constants import DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """
    A genome = a single set of strategy parameters.
    This is what the genetic algorithm evolves.

    SIMPLIFIED to 8 tunable params (was 25+) to reduce overfitting risk.
    Entry type is NOT a genome param - it's chosen at search time.
    """
    # === TUNABLE PARAMETERS (8 total) ===
    # EMA crossover
    ema_fast: int = 9
    ema_slow: int = 21
    # RSI filter
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    # ATR-based stops
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 4.0
    # Trend strength filter
    min_adx: float = 20.0
    # Volume filter
    use_volume_filter: bool = True
    volume_min_multiplier: float = 0.8
    # === SCORE (filled by backtester) ===
    profit_factor: float = 0.0
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 100.0
    sharpe: float = 0.0
    # === METADATA ===
    generation: int = 0
    id: str = ""

    def to_dict(self):
        return asdict(self)

    def is_valid(self):
        """Constraints - invalid genomes get rejected before scoring"""
        # EMA: fast must be less than slow
        if self.ema_fast >= self.ema_slow:
            return False
        if self.ema_fast < 3 or self.ema_fast > 50:
            return False
        if self.ema_slow < 10 or self.ema_slow > 150:
            return False
        # RSI: overbought must be > oversold, both in valid range
        if self.rsi_overbought <= self.rsi_oversold:
            return False
        if self.rsi_overbought > 95 or self.rsi_oversold < 5:
            return False
        if self.rsi_period not in (7, 14, 21):
            return False
        # Stops: must be positive
        if self.atr_sl_multiplier <= 0 or self.atr_tp_multiplier <= 0:
            return False
        if self.atr_sl_multiplier > 5 or self.atr_tp_multiplier > 10:
            return False
        # TP must be > SL (for positive risk:reward)
        if self.atr_tp_multiplier <= self.atr_sl_multiplier:
            return False
        # ADX threshold
        if self.min_adx < 10 or self.min_adx > 50:
            return False
        # Volume multiplier
        if self.volume_min_multiplier < 0.1 or self.volume_min_multiplier > 3.0:
            return False
        return True


class GeneticSearcher:
    """
    Genetic algorithm for evolving strategy parameters.

    Flow:
    1. Initialize random population
    2. Backtest each genome on in-sample data
    3. Keep top performers (elitism)
    4. Crossover + mutate to make next generation
    5. Validate best on out-of-sample data
    6. Repeat for N generations
    """

    def __init__(self, backtester=None, config: dict = None):
        self.config = config or {}
        self.backtester = backtester
        # GA settings
        self.population_size = self.config.get("population_size", 50)
        self.generations = self.config.get("generations", 20)
        self.elite_count = self.config.get("elite_count", 5)
        self.mutation_rate = self.config.get("mutation_rate", 0.2)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        # Acceptance thresholds
        self.min_profit_factor = self.config.get("min_profit_factor", 1.2)
        self.min_trades = self.config.get("min_trades", 20)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 25.0)
        # In-sample / out-of-sample split
        self.in_sample_pct = self.config.get("in_sample_pct", 0.7)
        self._lock = Lock()
        # Storage for best genomes (saved to DB)
        self.best_genomes: List[StrategyGenome] = []

    def _random_genome(self, generation: int = 0) -> StrategyGenome:
        """Create a random strategy genome (simplified 8-param version)."""
        g = StrategyGenome(
            # EMA crossover (the core signal)
            ema_fast=random.randint(5, 30),
            ema_slow=random.randint(20, 100),
            # RSI filter
            rsi_period=random.choice([7, 14, 21]),
            rsi_overbought=random.randint(60, 80),
            rsi_oversold=random.randint(20, 40),
            # ATR-based stops (SL must be < TP for positive RR)
            atr_sl_multiplier=round(random.uniform(1.0, 3.0), 1),
            atr_tp_multiplier=round(random.uniform(2.0, 6.0), 1),
            # Trend strength filter
            min_adx=round(random.uniform(15.0, 35.0), 1),
            # Volume filter
            use_volume_filter=random.choice([True, False]),
            volume_min_multiplier=round(random.uniform(0.5, 1.5), 1),
            generation=generation
        )
        if not g.is_valid():
            return self._random_genome(generation)
        g.id = f"g{generation}_{random.randint(10000, 99999)}"
        return g

    def _mutate(self, genome: StrategyGenome, generation: int) -> StrategyGenome:
        """Mutate a genome with small random changes (simplified 8-param)."""
        new = StrategyGenome(
            # EMA crossover
            ema_fast=max(3, min(50, genome.ema_fast + random.randint(-2, 2))),
            ema_slow=max(10, min(150, genome.ema_slow + random.randint(-4, 4))),
            # RSI filter
            rsi_period=genome.rsi_period,
            rsi_overbought=max(50, min(90, genome.rsi_overbought + random.randint(-3, 3))),
            rsi_oversold=max(10, min(50, genome.rsi_oversold + random.randint(-3, 3))),
            # ATR-based stops
            atr_sl_multiplier=round(max(0.5, min(5.0, genome.atr_sl_multiplier + random.uniform(-0.2, 0.2))), 1),
            atr_tp_multiplier=round(max(1.0, min(8.0, genome.atr_tp_multiplier + random.uniform(-0.3, 0.3))), 1),
            # Trend strength
            min_adx=round(max(10.0, min(50.0, genome.min_adx + random.uniform(-2, 2))), 1),
            # Volume filter
            use_volume_filter=random.random() < 0.1 or genome.use_volume_filter,
            volume_min_multiplier=round(max(0.1, min(3.0, genome.volume_min_multiplier + random.uniform(-0.1, 0.1))), 1),
            generation=generation
        )
        if not new.is_valid():
            return genome  # return unchanged if mutation invalid
        new.id = f"g{generation}_{random.randint(10000, 99999)}"
        return new

    def _crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome, generation: int) -> StrategyGenome:
        """Combine two parent genomes (simplified 8-param)."""
        child = StrategyGenome(
            # EMA crossover
            ema_fast=random.choice([parent_a.ema_fast, parent_b.ema_fast]),
            ema_slow=random.choice([parent_a.ema_slow, parent_b.ema_slow]),
            # RSI filter
            rsi_period=random.choice([parent_a.rsi_period, parent_b.rsi_period]),
            rsi_overbought=random.choice([parent_a.rsi_overbought, parent_b.rsi_overbought]),
            rsi_oversold=random.choice([parent_a.rsi_oversold, parent_b.rsi_oversold]),
            # ATR-based stops (blend for continuous evolution)
            atr_sl_multiplier=round((parent_a.atr_sl_multiplier + parent_b.atr_sl_multiplier) / 2, 1),
            atr_tp_multiplier=round((parent_a.atr_tp_multiplier + parent_b.atr_tp_multiplier) / 2, 1),
            # Trend strength
            min_adx=round((parent_a.min_adx + parent_b.min_adx) / 2, 1),
            # Volume filter
            use_volume_filter=random.choice([parent_a.use_volume_filter, parent_b.use_volume_filter]),
            volume_min_multiplier=round((parent_a.volume_min_multiplier + parent_b.volume_min_multiplier) / 2, 1),
            generation=generation
        )
        if not child.is_valid():
            return self._random_genome(generation)
        child.id = f"g{generation}_{random.randint(10000, 99999)}"
        return child

    def _score_genome(self, genome: StrategyGenome, bars: List[dict], symbol: str = None) -> StrategyGenome:
        """Backtest a genome on price data (simplified 8-param + cost model).

        Improvements over previous version:
        - No look-ahead: enter on next bar's open, not signal bar's close
        - Per-symbol costs: spread/slippage varies by instrument
        - Spread charged on EVERY trade (both wins and losses)
        - Only EMA cross + RSI + ADX + volume (no redundant indicators)
        """
        try:
            if len(bars) < 100:
                return genome
            closes = [b['close'] for b in bars]
            opens = [b.get('open', b['close']) for b in bars]
            highs = [b['high'] for b in bars]
            lows = [b['low'] for b in bars]

            # === INDICATORS (only what we need) ===
            ema_fast = self._ema(closes, genome.ema_fast)
            ema_slow = self._ema(closes, genome.ema_slow)
            rsi = self._rsi(closes, genome.rsi_period)
            atr = self._atr(highs, lows, closes, 14)
            adx_vals = self._adx(highs, lows, closes, 14)

            # Volume (use tick_volume if available)
            bars_with_vol = [b for b in bars if 'tick_volume' in b or 'volume' in b]
            if bars_with_vol and 'tick_volume' in bars_with_vol[0]:
                volumes = [b.get('tick_volume', 1) for b in bars]
            elif bars_with_vol and 'volume' in bars_with_vol[0]:
                volumes = [b.get('volume', 1) for b in bars]
            else:
                volumes = [1] * len(bars)
            vol_sma = self._volume_sma(volumes, 20)

            # === PER-SYMBOL COST MODEL (Option D fix) ===
            # Realistic costs: spread + slippage + commission
            # These hit EVERY trade, not just losses
            pip_size = 0.0001
            dollars_per_pip_per_lot = 10.0
            spread_pips = 0.7   # typical EUR/USD spread
            slippage_pips = 0.3  # average slippage
            commission_pips = 0.3
            if symbol:
                if 'JPY' in symbol:
                    pip_size = 0.01
                    spread_pips = 0.9
                elif 'XAU' in symbol:
                    pip_size = 0.01
                    dollars_per_pip_per_lot = 1.0  # $1 per 0.01 on 1 lot
                    spread_pips = 3.0  # gold has wider spread
                    slippage_pips = 0.5
                elif 'BRN' in symbol or 'OIL' in symbol or 'WTI' in symbol:
                    pip_size = 0.01
                    dollars_per_pip_per_lot = 1.0
                    spread_pips = 4.0  # crude has very wide spread
                    slippage_pips = 1.0
                elif 'XAG' in symbol:
                    pip_size = 0.01
                    dollars_per_pip_per_lot = 5.0
                    spread_pips = 2.5
            lot_size = 0.1  # mini lot for $1/pip on majors
            dollars_per_pip = dollars_per_pip_per_lot * lot_size
            # Total cost per trade (in pips) - hits both entry AND exit
            cost_per_side_pips = (spread_pips / 2) + slippage_pips
            cost_per_roundtrip_pips = (cost_per_side_pips * 2) + commission_pips

            # === SIMULATE TRADES ===
            trades = []
            in_trade = False
            entry_price = 0
            trade = {}

            for i in range(50, len(bars) - 1):  # -1 so we can use next bar's open for entry
                if not in_trade:
                    if (i < len(ema_fast) and ema_fast[i] is not None
                        and ema_slow[i] is not None
                        and i > 0 and ema_fast[i-1] is not None and ema_slow[i-1] is not None
                        and rsi[i] is not None and atr[i] is not None
                        and adx_vals[i] is not None):

                        # Filters: RSI not extreme, ADX strong enough
                        rsi_ok = (rsi[i] < genome.rsi_overbought and rsi[i] > genome.rsi_oversold)
                        adx_strong = adx_vals[i] >= genome.min_adx
                        # Volume filter (optional)
                        vol_ok = True
                        if genome.use_volume_filter and vol_sma[i] is not None and vol_sma[i] > 0:
                            vol_ok = volumes[i] >= vol_sma[i] * genome.volume_min_multiplier

                        # ENTRY: EMA crossover (only signal)
                        bullish_ema = (ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i])
                        bearish_ema = (ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i])

                        long_signal = bullish_ema and rsi_ok and adx_strong and vol_ok
                        short_signal = bearish_ema and rsi_ok and adx_strong and vol_ok

                        if long_signal or short_signal:
                            # NO LOOK-AHEAD: enter on NEXT bar's open, not this bar's close
                            entry_price = opens[i + 1]
                            atr_at_entry = atr[i]  # ATR from signal bar (acceptable)
                            if long_signal:
                                sl = entry_price - atr_at_entry * genome.atr_sl_multiplier
                                tp = entry_price + atr_at_entry * genome.atr_tp_multiplier
                                trade = {'entry': i + 1, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'side': 'long'}
                            else:
                                sl = entry_price + atr_at_entry * genome.atr_sl_multiplier
                                tp = entry_price - atr_at_entry * genome.atr_tp_multiplier
                                trade = {'entry': i + 1, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'side': 'short'}
                            in_trade = True
                else:
                    # Check exit on the bar AFTER entry (or later)
                    high = highs[i]
                    low = lows[i]
                    exit_price = 0
                    exit_reason = ''
                    if trade.get('side') == 'long':
                        if low <= trade['sl']:
                            exit_price = trade['sl']
                            exit_reason = 'SL'
                        elif high >= trade['tp']:
                            exit_price = trade['tp']
                            exit_reason = 'TP'
                    else:  # short
                        if high >= trade['sl']:
                            exit_price = trade['sl']
                            exit_reason = 'SL'
                        elif low <= trade['tp']:
                            exit_price = trade['tp']
                            exit_reason = 'TP'

                    if exit_price:
                        if trade.get('side') == 'long':
                            pnl_pips = (exit_price - entry_price) / pip_size
                        else:
                            pnl_pips = (entry_price - exit_price) / pip_size
                        # CRITICAL FIX: subtract cost on EVERY trade (both wins and losses)
                        pnl_pips -= cost_per_roundtrip_pips
                        pnl_dollars = pnl_pips * dollars_per_pip
                        trades.append({'pnl': pnl_dollars, 'pnl_pips': pnl_pips, 'reason': exit_reason})
                        in_trade = False
                    elif i > trade.get('entry', 0) + 50:  # Force exit after 50 bars
                        if trade.get('side') == 'long':
                            pnl_pips = (closes[i] - entry_price) / pip_size
                        else:
                            pnl_pips = (entry_price - closes[i]) / pip_size
                        pnl_pips -= cost_per_roundtrip_pips
                        pnl_dollars = pnl_pips * dollars_per_pip
                        trades.append({'pnl': pnl_dollars, 'pnl_pips': pnl_pips, 'reason': 'timeout'})
                        in_trade = False

            if not trades:
                return genome

            # === SCORING ===
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            total_pnl = sum(t['pnl'] for t in trades)
            gross_profit = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in losses)) or 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            win_rate = len(wins) / len(trades) * 100

            # Drawdown
            cumulative = 0
            peak = 0
            max_dd = 0
            for t in trades:
                cumulative += t['pnl']
                peak = max(peak, cumulative)
                dd = (peak - cumulative) / max(peak, 1) * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            # Sharpe
            if len(trades) > 1:
                import statistics
                returns = [t['pnl'] for t in trades]
                sharpe = (statistics.mean(returns) / max(statistics.stdev(returns), 0.01)) * (len(trades) ** 0.5)
            else:
                sharpe = 0

            genome.profit_factor = round(profit_factor, 3)
            genome.total_trades = len(trades)
            genome.win_rate = round(win_rate, 1)
            genome.wins = len(wins)
            genome.total_pnl = round(total_pnl, 2)
            genome.max_drawdown = round(max_dd, 1)
            genome.sharpe = round(sharpe, 2)
            # Lottery-ticket protection
            if len(wins) < 3 and profit_factor > 5:
                genome.profit_factor = 0.0
            if win_rate < 5.0 and len(trades) >= 10:
                genome.profit_factor = 0.0
        except Exception as e:
            import traceback
            logger.error(f"Genome scoring failed: {e}")
            logger.debug(traceback.format_exc())
        return genome

    def _ema(self, prices, period):
        if len(prices) < period:
            return [None] * len(prices)
        result = [None] * len(prices)
        multiplier = 2 / (period + 1)
        # Simple EMA implementation
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

    def _adx(self, highs, lows, closes, period=14):
        """Average Directional Index - measures trend strength.
        Returns list of same length as input; first N values are None."""
        n = len(highs)
        if n < period * 3:
            return [None] * n
        result = [None] * n
        # +DM and -DM per bar
        plus_dm = [0.0] * n
        minus_dm = [0.0] * n
        tr = [0.0] * n
        for i in range(1, n):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            if up > down and up > 0:
                plus_dm[i] = up
            if down > up and down > 0:
                minus_dm[i] = down
            tr[i] = max(highs[i] - lows[i],
                       abs(highs[i] - closes[i-1]),
                       abs(lows[i] - closes[i-1]))
        # Smooth using Wilder's method: smoothed[i] = smoothed[i-1] - smoothed[i-1]/N + val[i]
        def wilder_smooth(arr):
            smoothed = [0.0] * n
            # Initial sum for first `period` values
            s = sum(arr[1:period + 1])
            smoothed[period] = s
            for i in range(period + 1, n):
                smoothed[i] = smoothed[i-1] - smoothed[i-1]/period + arr[i]
            return smoothed
        tr_sm = wilder_smooth(tr)
        plus_dm_sm = wilder_smooth(plus_dm)
        minus_dm_sm = wilder_smooth(minus_dm)
        # +DI, -DI
        plus_di = [0.0] * n
        minus_di = [0.0] * n
        dx = [0.0] * n
        for i in range(period, n):
            if tr_sm[i] > 0:
                plus_di[i] = 100 * plus_dm_sm[i] / tr_sm[i]
                minus_di[i] = 100 * minus_dm_sm[i] / tr_sm[i]
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di
        # ADX = Wilder smooth of DX
        if n < period * 2:
            return result
        adx_sum = sum(dx[period:period*2])
        if period == 0:
            return result
        adx_val = adx_sum / period
        result[period * 2] = adx_val
        for i in range(period * 2 + 1, n):
            adx_val = (adx_val * (period - 1) + dx[i]) / period
            result[i] = adx_val
        return result

    def _bollinger_bands(self, prices, period, stddev):
        """Bollinger Bands: middle (SMA), upper, lower"""
        if len(prices) < period:
            return [None] * len(prices), [None] * len(prices), [None] * len(prices)
        middle = [None] * len(prices)
        upper = [None] * len(prices)
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
        """MACD: macd_line, signal_line, histogram"""
        ema_fast = self._ema(prices, fast_period)
        ema_slow = self._ema(prices, slow_period)
        macd_line = [None] * len(prices)
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line[i] = ema_fast[i] - ema_slow[i]
        # Signal line = EMA of macd_line
        macd_values = [v if v is not None else 0 for v in macd_line]
        signal_line_raw = self._ema(macd_values, signal_period)
        signal_line = [None] * len(prices)
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line_raw[i] is not None:
                signal_line[i] = signal_line_raw[i]
        histogram = [None] * len(prices)
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram[i] = macd_line[i] - signal_line[i]
        return macd_line, signal_line, histogram

    def _volume_sma(self, volumes, period=20):
        """Simple moving average of volume"""
        if len(volumes) < period:
            return [None] * len(volumes)
        result = [None] * len(volumes)
        for i in range(period - 1, len(volumes)):
            result[i] = sum(volumes[i - period + 1:i + 1]) / period
        return result

    def _ichimoku(self, highs, lows, closes, tenkan=9, kijun=26, senkou_b=52):
        """Ichimoku Cloud: returns (tenkan, kijun, senkou_a, senkou_b, chikou).
        All lists same length as input; first values are None until enough data."""
        n = len(highs)
        result_tenkan = [None] * n
        result_kijun = [None] * n
        result_senkou_a = [None] * n
        result_senkou_b = [None] * n
        result_chikou = [None] * n
        for i in range(n):
            # Tenkan-sen (Conversion): highest high + lowest low over past `tenkan` periods, / 2
            if i >= tenkan - 1:
                hh = max(highs[i - tenkan + 1:i + 1])
                ll = min(lows[i - tenkan + 1:i + 1])
                result_tenkan[i] = (hh + ll) / 2
            # Kijun-sen (Base): same but for `kijun` periods
            if i >= kijun - 1:
                hh = max(highs[i - kijun + 1:i + 1])
                ll = min(lows[i - kijun + 1:i + 1])
                result_kijun[i] = (hh + ll) / 2
            # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
            if i >= kijun - 1 and result_tenkan[i] is not None and result_kijun[i] is not None:
                sa = (result_tenkan[i] + result_kijun[i]) / 2
                # Place it 26 periods in the future
                if i + 26 < n:
                    result_senkou_a[i + 26] = sa
            # Senkou Span B: (highest high + lowest low) over past 52, / 2, plotted 26 ahead
            if i >= senkou_b - 1:
                hh = max(highs[i - senkou_b + 1:i + 1])
                ll = min(lows[i - senkou_b + 1:i + 1])
                sb = (hh + ll) / 2
                if i + 26 < n:
                    result_senkou_b[i + 26] = sb
            # Chikou Span: current close plotted 26 periods back
            if i - 26 >= 0:
                result_chikou[i - 26] = closes[i]
        return result_tenkan, result_kijun, result_senkou_a, result_senkou_b, result_chikou

    def _heikin_ashi(self, opens, highs, lows, closes):
        """Heikin Ashi candles: smoother price action.
        Returns (ha_open, ha_high, ha_low, ha_close) - all same length as input."""
        n = len(closes)
        if n == 0:
            return [], [], [], []
        ha_open = [None] * n
        ha_high = [None] * n
        ha_low = [None] * n
        ha_close = [None] * n
        # First bar: use real open/close
        ha_close[0] = (opens[0] + highs[0] + lows[0] + closes[0]) / 4
        ha_open[0] = (opens[0] + closes[0]) / 2
        ha_high[0] = highs[0]
        ha_low[0] = lows[0]
        for i in range(1, n):
            ha_close[i] = (opens[i] + highs[i] + lows[i] + closes[i]) / 4
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(highs[i], ha_open[i], ha_close[i])
            ha_low[i] = min(lows[i], ha_open[i], ha_close[i])
        return ha_open, ha_high, ha_low, ha_close

    def _atr_pips(self, highs, lows, closes, period=14, pip_size=0.0001):
        """ATR in pips (for the per-symbol ATR floor)"""
        atr_raw = self._atr(highs, lows, closes, period)
        return [a / pip_size if a is not None else None for a in atr_raw]

    def _sma(self, prices, period):
        """Simple moving average"""
        if len(prices) < period:
            return [None] * len(prices)
        result = [None] * len(prices)
        for i in range(period - 1, len(prices)):
            result[i] = sum(prices[i - period + 1:i + 1]) / period
        return result

    def _get_mt5_bars(self, symbol: str, timeframe: str, days: int) -> List[dict]:
        """Get historical bars from MT5 for a specific symbol and timeframe"""
        if not self.backtester:
            logger.error("No backtester available - cannot fetch MT5 data")
            return []
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            logger.info(f"Fetching {days} days of {timeframe} data for {symbol} from MT5...")
            bars = self.backtester.download_historical_data(symbol, timeframe, start, end)
            if bars:
                logger.info(f"Got {len(bars)} {timeframe} bars for {symbol} "
                           f"({bars[0].get('time', '?')} to {bars[-1].get('time', '?')})")
            else:
                logger.warning(f"No bars returned for {symbol} {timeframe}")
            return bars
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol} {timeframe}: {e}")
            return []

    def _get_multi_symbol_bars(self, symbols: List[str], timeframes: List[str] = None,
                                days: int = 365) -> Dict[str, Dict[str, List[dict]]]:
        """
        Get historical bars for multiple symbols and timeframes.
        Returns: {symbol: {timeframe: [bars]}}
        """
        if timeframes is None:
            timeframes = ["H4", "H1"]  # Default: H4 for trend, H1 for entry
        all_data = {}
        for symbol in symbols:
            all_data[symbol] = {}
            for tf in timeframes:
                bars = self._get_mt5_bars(symbol, tf, days)
                if bars and len(bars) >= 100:
                    all_data[symbol][tf] = bars
                else:
                    logger.warning(f"Insufficient {tf} data for {symbol}: {len(bars) if bars else 0} bars")
        return all_data

    def run_search(self, symbols: List[str] = None, timeframes: List[str] = None,
                   days: int = 365) -> List[StrategyGenome]:
        """
        Run the full genetic algorithm search.
        Tests across multiple symbols and timeframes.
        Returns list of validated genomes that pass OOS test.
        """
        if symbols is None:
            symbols = ["EURUSD"]
        if timeframes is None:
            timeframes = ["H1"]

        logger.info(f"Starting genetic search: symbols={symbols}, timeframes={timeframes}, "
                    f"days={days}, pop={self.population_size}, gens={self.generations}")

        # Try to get bars for all symbols and timeframes
        all_bars = self._get_multi_symbol_bars(symbols, timeframes, days)

        # If MT5 didn't return data, fall back to synthetic
        total_bars = sum(len(bars) for sym_data in all_bars.values() for bars in sym_data.values())
        if total_bars == 0:
            logger.warning("No MT5 historical data available - using synthetic for demo")
            synthetic = self._synthetic_bars(days * 24)
            all_bars = {sym: {"H1": synthetic} for sym in symbols}
        else:
            logger.info(f"Loaded {total_bars} total bars from MT5 across {len(symbols)} symbols")

        # Combine all bars from all symbols/timeframes for training
        all_bars_flat = []
        for sym, tf_data in all_bars.items():
            for tf, bars in tf_data.items():
                # Tag each bar with its symbol/timeframe
                for b in bars:
                    b_copy = dict(b)
                    b_copy['_symbol'] = sym
                    b_copy['_timeframe'] = tf
                    all_bars_flat.append(b_copy)

        if not all_bars_flat:
            logger.error("No bars to search")
            return []

        # Sort by time
        all_bars_flat.sort(key=lambda x: x.get('time', datetime.now()))

        # Split into in-sample and out-of-sample
        split = int(len(all_bars_flat) * self.in_sample_pct)
        in_sample = all_bars_flat[:split]
        oos = all_bars_flat[split:]
        logger.info(f"In-sample: {len(in_sample)} bars, OOS: {len(oos)} bars")

        # Initialize population
        population = [self._random_genome(0) for _ in range(self.population_size)]

        # Evolve
        for gen in range(self.generations):
            # Score in-sample (across all symbols)
            for g in population:
                g = self._score_genome_multi(g, in_sample)

            # Sort by profit factor
            population.sort(key=lambda x: x.profit_factor, reverse=True)
            best = population[0]
            logger.info(
                f"Gen {gen}: best PF={best.profit_factor:.2f} trades={best.total_trades} "
                f"WR={best.win_rate:.1f}% DD={best.max_drawdown:.1f}% "
                f"genome=[{best.ema_fast},{best.ema_slow}]"
            )

            # Elitism: keep top performers
            new_pop = population[:self.elite_count]
            # Generate new population
            while len(new_pop) < self.population_size:
                if random.random() < self.crossover_rate:
                    a = self._tournament(population)
                    b = self._tournament(population)
                    child = self._crossover(a, b, gen + 1)
                else:
                    child = self._tournament(population)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, gen + 1)
                new_pop.append(child)
            population = new_pop

        # Final OOS validation
        logger.info("Running out-of-sample validation on best genomes...")
        validated = []
        for g in population[:self.elite_count * 2]:  # top 10
            oos_genome = StrategyGenome(**{k: v for k, v in g.to_dict().items() if k in [
                'ema_fast', 'ema_slow', 'rsi_period', 'rsi_overbought', 'rsi_oversold',
                'use_volume_filter', 'volume_min_multiplier',
                'atr_sl_multiplier', 'atr_tp_multiplier', 'min_adx'
            ]})
            oos_genome = self._score_genome_multi(oos_genome, oos)
            # Reject if:
            #  - PF too low
            #  - Too few trades (statistically meaningless - need at least 30)
            #  - Drawdown too high (would blow account)
            #  - Win rate suspiciously low (lottery ticket, not a strategy)
            if (oos_genome.profit_factor >= self.min_profit_factor
                and oos_genome.total_trades >= max(self.min_trades, 30)
                and oos_genome.max_drawdown <= self.max_drawdown_pct
                and oos_genome.win_rate >= 30.0):  # require at least 30% WR (real strategy)
                oos_genome.generation = self.generations
                oos_genome.id = f"validated_{len(validated)}"
                validated.append(oos_genome)
                logger.info(
                    f"VALIDATED: PF={oos_genome.profit_factor:.2f} "
                    f"trades={oos_genome.total_trades} DD={oos_genome.max_drawdown:.1f}%"
                )
            else:
                logger.info(
                    f"REJECTED: PF={oos_genome.profit_factor:.2f} "
                    f"trades={oos_genome.total_trades} DD={oos_genome.max_drawdown:.1f}%"
                )

        with self._lock:
            self.best_genomes = validated
        # Save per-symbol
        for sym in symbols:
            self._save_to_db(validated, sym)
        return validated

    def _score_genome_multi(self, genome: StrategyGenome, bars_with_meta: List[dict]) -> StrategyGenome:
        """Score a genome across multiple symbols (each bar has _symbol key)"""
        # Group bars by symbol
        by_symbol = {}
        for b in bars_with_meta:
            sym = b.get('_symbol', 'UNKNOWN')
            by_symbol.setdefault(sym, []).append(b)
        # Score each symbol separately and sum
        combined_pnl = 0
        total_trades = 0
        wins = 0
        worst_dd = 0
        first_symbol = None
        for sym, sym_bars in by_symbol.items():
            if first_symbol is None:
                first_symbol = sym
            # Strip metadata for scoring
            clean_bars = [{k: v for k, v in b.items() if not k.startswith('_')} for b in sym_bars]
            sym_genome = self._score_genome(genome, clean_bars, symbol=sym)
            combined_pnl += sym_genome.total_pnl
            total_trades += sym_genome.total_trades
            wins += sym_genome.wins
            worst_dd = max(worst_dd, sym_genome.max_drawdown)
        genome.total_pnl = combined_pnl
        genome.total_trades = total_trades
        genome.wins = wins
        genome.max_drawdown = worst_dd
        # Recalculate PF across all symbols
        if total_trades > 0:
            win_rate = wins / total_trades
            genome.win_rate = win_rate
        return genome

    def _tournament(self, population, k=3):
        """Tournament selection: pick k random, return best"""
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=lambda x: x.profit_factor)

    def _synthetic_bars(self, n: int) -> List[dict]:
        """Generate synthetic bars for testing without MT5"""
        import math
        bars = []
        price = 1.1000
        for i in range(n):
            # Random walk with trend
            trend = math.sin(i / 200) * 0.001
            noise = random.uniform(-0.0020, 0.0020)
            open_p = price
            close_p = price + trend + noise
            high_p = max(open_p, close_p) + abs(noise) * 0.5
            low_p = min(open_p, close_p) - abs(noise) * 0.5
            bars.append({
                'time': datetime.now() - timedelta(hours=n - i),
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p,
                'tick_volume': random.randint(100, 1000), 'spread': 10, 'real_volume': 0
            })
            price = close_p
        return bars

    def _save_to_db(self, genomes: List[StrategyGenome], symbol: str):
        """Persist validated genomes to DB for live use (simplified 8-param schema)."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_genomes (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    ema_fast INTEGER, ema_slow INTEGER,
                    rsi_period INTEGER, rsi_overbought INTEGER, rsi_oversold INTEGER,
                    use_volume_filter INTEGER, volume_min_multiplier REAL,
                    atr_sl_multiplier REAL, atr_tp_multiplier REAL,
                    min_adx REAL,
                    profit_factor REAL, total_trades INTEGER,
                    wins INTEGER, win_rate REAL, total_pnl REAL,
                    max_drawdown REAL, sharpe REAL,
                    generation INTEGER, validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for g in genomes:
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_genomes
                    (id, symbol, ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold,
                     use_volume_filter, volume_min_multiplier,
                     atr_sl_multiplier, atr_tp_multiplier, min_adx,
                     profit_factor, total_trades, wins, win_rate, total_pnl,
                     max_drawdown, sharpe, generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (g.id, symbol, g.ema_fast, g.ema_slow, g.rsi_period,
                      g.rsi_overbought, g.rsi_oversold,
                      int(g.use_volume_filter), g.volume_min_multiplier,
                      g.atr_sl_multiplier, g.atr_tp_multiplier, g.min_adx,
                      g.profit_factor, g.total_trades, g.wins, g.win_rate, g.total_pnl,
                      g.max_drawdown, g.sharpe, g.generation))
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(genomes)} validated genomes to DB")
        except Exception as e:
            logger.error(f"Failed to save genomes: {e}")

    @staticmethod
    def load_best_genome(symbol: str = None) -> Optional[StrategyGenome]:
        """Load the best genome from DB for live trading"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM ml_genomes WHERE symbol=? ORDER BY profit_factor DESC LIMIT 1",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM ml_genomes ORDER BY profit_factor DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            cols = [d[0] for d in cursor.description] if cursor.description else []
            d = dict(zip(cols, row)) if row else {}
            return StrategyGenome(
                ema_fast=d.get('ema_fast', 9),
                ema_slow=d.get('ema_slow', 21),
                rsi_period=d.get('rsi_period', 14),
                rsi_overbought=d.get('rsi_overbought', 70),
                rsi_oversold=d.get('rsi_oversold', 30),
                use_volume_filter=bool(d.get('use_volume_filter', 1)),
                volume_min_multiplier=d.get('volume_min_multiplier', 0.8),
                atr_sl_multiplier=d.get('atr_sl_multiplier', 2.0),
                atr_tp_multiplier=d.get('atr_tp_multiplier', 4.0),
                min_adx=d.get('min_adx', 20.0),
                profit_factor=d.get('profit_factor', 0),
                total_trades=d.get('total_trades', 0),
                wins=d.get('wins', 0),
                win_rate=d.get('win_rate', 0),
                total_pnl=d.get('total_pnl', 0),
                max_drawdown=d.get('max_drawdown', 0),
                sharpe=d.get('sharpe', 0),
                generation=d.get('generation', 0),
                id=d.get('id', ''),
            )
        except Exception as e:
            logger.error(f"Failed to load genome: {e}")
            return None

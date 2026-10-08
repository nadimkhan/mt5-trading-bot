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
    """
    # EMA periods (for crossover)
    ema_fast: int = 9
    ema_slow: int = 21
    # RSI thresholds
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    # Bollinger Bands
    bb_period: int = 20
    bb_stddev: float = 2.0
    use_bb_filter: bool = True  # Trade only at BB extremes
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_macd_filter: bool = True
    # Volume filter
    use_volume_filter: bool = True
    volume_min_multiplier: float = 0.8  # Volume must be >= 80% of 20-bar avg
    # ATR-based stops
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 4.0
    # Trend strength
    min_adx: float = 20.0
    # Risk filters
    min_atr_pips: float = 5.0  # Don't trade dead markets
    max_spread_pips: float = 5.0
    # Signal logic toggle
    use_ema_cross: bool = True
    use_macd_cross: bool = True
    use_bb_bounce: bool = False
    # Score (filled by backtester)
    profit_factor: float = 0.0
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 100.0
    sharpe: float = 0.0
    # Metadata
    generation: int = 0
    id: str = ""

    def to_dict(self):
        return asdict(self)

    def is_valid(self):
        """Constraints - invalid genomes get rejected"""
        if self.ema_fast >= self.ema_slow:
            return False
        if self.ema_fast < 2 or self.ema_fast > 100:
            return False
        if self.ema_slow < 5 or self.ema_slow > 200:
            return False
        if self.atr_sl_multiplier <= 0 or self.atr_tp_multiplier <= 0:
            return False
        if self.rsi_overbought <= self.rsi_oversold:
            return False
        if self.rsi_overbought > 95 or self.rsi_oversold < 5:
            return False
        if self.min_adx < 0 or self.min_adx > 60:
            return False
        if self.bb_period < 5 or self.bb_period > 60:
            return False
        if self.bb_stddev < 0.5 or self.bb_stddev > 4.0:
            return False
        if self.macd_fast >= self.macd_slow:
            return False
        if self.macd_slow < 5 or self.macd_slow > 60:
            return False
        if self.volume_min_multiplier < 0.1 or self.volume_min_multiplier > 3.0:
            return False
        if self.min_atr_pips < 0 or self.min_atr_pips > 50:
            return False
        # Must enable at least one entry signal
        if not (self.use_ema_cross or self.use_macd_cross or self.use_bb_bounce):
            return False
        if self.max_spread_pips < 0.1 or self.max_spread_pips > 50:
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
        """Create a random strategy genome"""
        g = StrategyGenome(
            ema_fast=random.randint(3, 30),
            ema_slow=random.randint(15, 80),
            rsi_period=random.choice([7, 14, 21]),
            rsi_overbought=random.randint(60, 85),
            rsi_oversold=random.randint(15, 40),
            # Bollinger
            bb_period=random.choice([15, 20, 25, 30]),
            bb_stddev=round(random.uniform(1.5, 2.8), 1),
            use_bb_filter=random.choice([True, False]),
            # MACD
            macd_fast=random.choice([8, 10, 12, 15]),
            macd_slow=random.choice([20, 24, 26, 30]),
            macd_signal=random.choice([7, 9, 12]),
            use_macd_filter=random.choice([True, False]),
            # Volume
            use_volume_filter=random.choice([True, False]),
            volume_min_multiplier=round(random.uniform(0.5, 1.2), 1),
            # Stops
            atr_sl_multiplier=round(random.uniform(1.0, 4.0), 1),
            atr_tp_multiplier=round(random.uniform(2.0, 8.0), 1),
            # Trend
            min_adx=round(random.uniform(15, 35), 1),
            min_atr_pips=round(random.uniform(3.0, 12.0), 1),
            max_spread_pips=round(random.uniform(1.0, 10.0), 1),
            # Strategy selection
            use_ema_cross=random.choice([True, False]),
            use_macd_cross=random.choice([True, False]),
            use_bb_bounce=random.choice([True, False]),
            generation=generation
        )
        if not g.is_valid():
            return self._random_genome(generation)
        g.id = f"g{generation}_{random.randint(10000, 99999)}"
        return g

    def _mutate(self, genome: StrategyGenome, generation: int) -> StrategyGenome:
        """Mutate a genome with small random changes"""
        new = StrategyGenome(
            ema_fast=genome.ema_fast + random.randint(-3, 3),
            ema_slow=genome.ema_slow + random.randint(-5, 5),
            rsi_period=genome.rsi_period,
            rsi_overbought=genome.rsi_overbought + random.randint(-5, 5),
            rsi_oversold=genome.rsi_oversold + random.randint(-3, 3),
            # Bollinger
            bb_period=genome.bb_period,
            bb_stddev=round(genome.bb_stddev + random.uniform(-0.2, 0.2), 1),
            use_bb_filter=random.random() < 0.1 or genome.use_bb_filter,  # 10% chance to flip
            # MACD
            macd_fast=genome.macd_fast,
            macd_slow=genome.macd_slow,
            macd_signal=genome.macd_signal,
            use_macd_filter=random.random() < 0.1 or genome.use_macd_filter,
            # Volume
            use_volume_filter=random.random() < 0.1 or genome.use_volume_filter,
            volume_min_multiplier=round(genome.volume_min_multiplier + random.uniform(-0.1, 0.1), 1),
            # Stops
            atr_sl_multiplier=round(genome.atr_sl_multiplier + random.uniform(-0.3, 0.3), 1),
            atr_tp_multiplier=round(genome.atr_tp_multiplier + random.uniform(-0.5, 0.5), 1),
            # Trend
            min_adx=round(genome.min_adx + random.uniform(-3, 3), 1),
            min_atr_pips=round(genome.min_atr_pips + random.uniform(-1, 1), 1),
            max_spread_pips=round(genome.max_spread_pips + random.uniform(-1, 1), 1),
            # Strategy
            use_ema_cross=random.random() < 0.1 or genome.use_ema_cross,
            use_macd_cross=random.random() < 0.1 or genome.use_macd_cross,
            use_bb_bounce=random.random() < 0.1 or genome.use_bb_bounce,
            generation=generation
        )
        if not new.is_valid():
            return genome  # return unchanged if mutation invalid
        new.id = f"g{generation}_{random.randint(10000, 99999)}"
        return new

    def _crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome, generation: int) -> StrategyGenome:
        """Combine two parent genomes"""
        child = StrategyGenome(
            ema_fast=random.choice([parent_a.ema_fast, parent_b.ema_fast]),
            ema_slow=random.choice([parent_a.ema_slow, parent_b.ema_slow]),
            rsi_period=random.choice([parent_a.rsi_period, parent_b.rsi_period]),
            rsi_overbought=random.choice([parent_a.rsi_overbought, parent_b.rsi_overbought]),
            rsi_oversold=random.choice([parent_a.rsi_oversold, parent_b.rsi_oversold]),
            # Bollinger
            bb_period=random.choice([parent_a.bb_period, parent_b.bb_period]),
            bb_stddev=round((parent_a.bb_stddev + parent_b.bb_stddev) / 2, 1),
            use_bb_filter=random.choice([parent_a.use_bb_filter, parent_b.use_bb_filter]),
            # MACD
            macd_fast=random.choice([parent_a.macd_fast, parent_b.macd_fast]),
            macd_slow=random.choice([parent_a.macd_slow, parent_b.macd_slow]),
            macd_signal=random.choice([parent_a.macd_signal, parent_b.macd_signal]),
            use_macd_filter=random.choice([parent_a.use_macd_filter, parent_b.use_macd_filter]),
            # Volume
            use_volume_filter=random.choice([parent_a.use_volume_filter, parent_b.use_volume_filter]),
            volume_min_multiplier=round((parent_a.volume_min_multiplier + parent_b.volume_min_multiplier) / 2, 1),
            # Stops
            atr_sl_multiplier=round((parent_a.atr_sl_multiplier + parent_b.atr_sl_multiplier) / 2, 1),
            atr_tp_multiplier=round((parent_a.atr_tp_multiplier + parent_b.atr_tp_multiplier) / 2, 1),
            # Trend
            min_adx=round((parent_a.min_adx + parent_b.min_adx) / 2, 1),
            min_atr_pips=round((parent_a.min_atr_pips + parent_b.min_atr_pips) / 2, 1),
            max_spread_pips=round((parent_a.max_spread_pips + parent_b.max_spread_pips) / 2, 1),
            # Strategy
            use_ema_cross=random.choice([parent_a.use_ema_cross, parent_b.use_ema_cross]),
            use_macd_cross=random.choice([parent_a.use_macd_cross, parent_b.use_macd_cross]),
            use_bb_bounce=random.choice([parent_a.use_bb_bounce, parent_b.use_bb_bounce]),
            generation=generation
        )
        if not child.is_valid():
            return self._random_genome(generation)
        child.id = f"g{generation}_{random.randint(10000, 99999)}"
        return child

    def _score_genome(self, genome: StrategyGenome, bars: List[dict], symbol: str = None) -> StrategyGenome:
        """Backtest a genome on price data and set its score fields"""
        try:
            if len(bars) < 100:
                return genome
            closes = [b['close'] for b in bars]
            highs = [b['high'] for b in bars]
            lows = [b['low'] for b in bars]

            # Compute indicators
            ema_fast = self._ema(closes, genome.ema_fast)
            ema_slow = self._ema(closes, genome.ema_slow)
            rsi = self._rsi(closes, genome.rsi_period)
            atr = self._atr(highs, lows, closes, 14)
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, genome.bb_period, genome.bb_stddev)
            # MACD
            macd_line, macd_signal, macd_hist = self._macd(closes, genome.macd_fast, genome.macd_slow, genome.macd_signal)
            # Volume (use tick_volume if available, fallback to 1s)
            bars_with_vol = [b for b in bars if 'tick_volume' in b or 'volume' in b]
            if bars_with_vol and 'tick_volume' in bars_with_vol[0]:
                volumes = [b.get('tick_volume', 1) for b in bars]
            elif bars_with_vol and 'volume' in bars_with_vol[0]:
                volumes = [b.get('volume', 1) for b in bars]
            else:
                volumes = [1] * len(bars)  # no volume data → pass filter trivially
            vol_sma = self._volume_sma(volumes, 20)

            # Determine pip size and dollar value per pip based on symbol
            pip_size = 0.0001
            dollars_per_pip_per_lot = 10.0  # 1 standard lot of major forex = $10/pip
            if symbol:
                if 'JPY' in symbol:
                    pip_size = 0.01
                elif 'XAU' in symbol or 'BRN' in symbol or 'OIL' in symbol:
                    pip_size = 0.01
                    dollars_per_pip_per_lot = 1.0  # 1 lot XAUUSD = $1 per 0.01 move
                elif any(c in symbol for c in ['BTC', 'ETH']):
                    pip_size = 1.0
                    dollars_per_pip_per_lot = 1.0

            # Use 0.1 lot (mini) for $1/pip on majors, $0.10/pip on XAUUSD
            lot_multiplier = 0.1
            dollars_per_pip = dollars_per_pip_per_lot * lot_multiplier
            # Spread cost in pips (~0.5-1.5 for majors, ~3-5 for XAUUSD)
            spread_pips = 1.5
            if symbol and ('XAU' in symbol or 'BRN' in symbol or 'OIL' in symbol):
                spread_pips = 3.0
            commission_pips = 0.5  # round-trip commission

            # Walk through bars, simulate trades
            trades = []
            in_trade = False
            entry_price = 0
            atr_at_entry = 0
            trade = {}

            # Simple ADX calc
            adx_period = 14
            adx_vals = self._adx(highs, lows, closes, adx_period)

            for i in range(50, len(bars)):
                if not in_trade:
                    # Check entry conditions
                    if (i < len(ema_fast) and ema_fast[i] is not None
                        and ema_slow[i] is not None
                        and i > 0 and ema_fast[i-1] is not None and ema_slow[i-1] is not None
                        and rsi[i] is not None and atr[i] is not None
                        and bb_lower[i] is not None and bb_upper[i] is not None
                        and macd_line[i] is not None and macd_signal[i] is not None
                        and macd_line[i-1] is not None and macd_signal[i-1] is not None
                        and adx_vals[i] is not None):

                        # ATR floor: don't trade dead markets
                        atr_pips = atr[i] / pip_size
                        if atr_pips < genome.min_atr_pips:
                            continue

                        # 1. EMA crossover detection
                        bullish_ema = (ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i])
                        bearish_ema = (ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i])

                        # 2. MACD confirmation
                        macd_bull = (macd_line[i-1] <= macd_signal[i-1] and macd_line[i] > macd_signal[i])
                        macd_bear = (macd_line[i-1] >= macd_signal[i-1] and macd_line[i] < macd_signal[i])

                        # 3. Bollinger band extremes
                        bb_bull = closes[i] <= bb_lower[i]   # touched/under lower band
                        bb_bear = closes[i] >= bb_upper[i]   # touched/over upper band

                        # 4. Volume filter
                        vol_ok = True
                        if genome.use_volume_filter and vol_sma[i] is not None and vol_sma[i] > 0:
                            vol_ok = volumes[i] >= vol_sma[i] * genome.volume_min_multiplier

                        # 5. RSI filter
                        rsi_ok = (rsi[i] < genome.rsi_overbought and rsi[i] > genome.rsi_oversold)

                        # 6. ADX trend strength filter
                        adx_strong = adx_vals[i] >= 20

                        # Decide entry signal based on which filters the genome uses
                        long_signal = False
                        short_signal = False

                        # EMA cross + MACD cross + ADX = strongest setup
                        if genome.use_ema_cross and genome.use_macd_filter and adx_strong and rsi_ok and vol_ok:
                            if bullish_ema and macd_bull:
                                long_signal = True
                            if bearish_ema and macd_bear:
                                short_signal = True
                        # EMA cross only (less strict)
                        elif genome.use_ema_cross and adx_strong and rsi_ok and vol_ok:
                            if bullish_ema:
                                long_signal = True
                            if bearish_ema:
                                short_signal = True
                        # MACD cross only
                        elif genome.use_macd_filter and adx_strong and rsi_ok and vol_ok:
                            if macd_bull:
                                long_signal = True
                            if macd_bear:
                                short_signal = True
                        # Bollinger band bounce (mean reversion)
                        elif genome.use_bb_bounce and vol_ok:
                            # Long at lower band, short at upper band
                            if bb_bull and macd_line[i] > macd_signal[i] and rsi[i] < 35:
                                long_signal = True
                            if bb_bear and macd_line[i] < macd_signal[i] and rsi[i] > 65:
                                short_signal = True

                        if long_signal:
                            in_trade = True
                            entry_price = closes[i]
                            atr_at_entry = atr[i]
                            sl = entry_price - atr_at_entry * genome.atr_sl_multiplier
                            tp = entry_price + atr_at_entry * genome.atr_tp_multiplier
                            trade = {'entry': i, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'side': 'long'}
                        elif short_signal:
                            in_trade = True
                            entry_price = closes[i]
                            atr_at_entry = atr[i]
                            sl = entry_price + atr_at_entry * genome.atr_sl_multiplier
                            tp = entry_price - atr_at_entry * genome.atr_tp_multiplier
                            trade = {'entry': i, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'side': 'short'}
                else:
                    # Check exit (SL/TP) - check if high/low hit levels this bar
                    high = highs[i]
                    low = lows[i]
                    pnl = 0
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
                        # Convert pips to dollars: simple, realistic formula
                        pnl_dollars = pnl_pips * dollars_per_pip
                        # Subtract transaction costs (spread + commission)
                        pnl_dollars -= (spread_pips + commission_pips) * dollars_per_pip
                        trades.append({'pnl': pnl_dollars, 'reason': exit_reason})
                        in_trade = False
                    elif i > trade.get('entry', 0) + 50:  # Force exit after 50 bars
                        if trade.get('side') == 'long':
                            pnl_pips = (closes[i] - entry_price) / pip_size
                        else:
                            pnl_pips = (entry_price - closes[i]) / pip_size
                        pnl_dollars = pnl_pips * dollars_per_pip
                        pnl_dollars -= (spread_pips + commission_pips) * dollars_per_pip
                        trades.append({'pnl': pnl_dollars, 'reason': 'timeout'})
                        in_trade = False

            if not trades:
                return genome

            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            total_pnl = sum(t['pnl'] for t in trades)
            gross_profit = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in losses)) or 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            win_rate = len(wins) / len(trades) * 100
            # Drawdown (rough)
            cumulative = 0
            peak = 0
            max_dd = 0
            for t in trades:
                cumulative += t['pnl']
                peak = max(peak, cumulative)
                dd = (peak - cumulative) / max(peak, 1) * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            # Sharpe (rough annualized)
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
            # Sanity check: if PF is huge but win rate is tiny, it's a "lottery ticket" strategy
            # Cap the profit factor to discourage overfitting to a single lucky trade
            # Real strategies have WR >= 30% with reasonable trade counts
            if len(wins) < 3 and profit_factor > 5:
                # Reject this score by setting PF very low
                genome.profit_factor = 0.0
                logger.debug(f"Capping lottery-ticket PF: {profit_factor:.1f} with only {len(wins)} wins out of {len(trades)} trades")
            # Also reject if WR < 5% even with many trades (random strategy)
            if win_rate < 5.0 and len(trades) >= 10:
                genome.profit_factor = 0.0
                logger.debug(f"Capping low-WR strategy: {win_rate:.1f}% win rate over {len(trades)} trades")
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
                'bb_period', 'bb_stddev', 'use_bb_filter',
                'macd_fast', 'macd_slow', 'macd_signal', 'use_macd_filter',
                'use_volume_filter', 'volume_min_multiplier',
                'atr_sl_multiplier', 'atr_tp_multiplier', 'min_adx', 'min_atr_pips',
                'max_spread_pips', 'use_ema_cross', 'use_macd_cross', 'use_bb_bounce'
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
        """Persist validated genomes to DB for live use"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_genomes (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    ema_fast INTEGER, ema_slow INTEGER,
                    rsi_period INTEGER, rsi_overbought INTEGER, rsi_oversold INTEGER,
                    bb_period INTEGER, bb_stddev REAL, use_bb_filter INTEGER,
                    macd_fast INTEGER, macd_slow INTEGER, macd_signal INTEGER, use_macd_filter INTEGER,
                    use_volume_filter INTEGER, volume_min_multiplier REAL,
                    atr_sl_multiplier REAL, atr_tp_multiplier REAL,
                    min_adx REAL, min_atr_pips REAL, max_spread_pips REAL,
                    use_ema_cross INTEGER, use_macd_cross INTEGER, use_bb_bounce INTEGER,
                    profit_factor REAL, total_trades INTEGER,
                    win_rate REAL, max_drawdown REAL, sharpe REAL,
                    generation INTEGER, validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for g in genomes:
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_genomes
                    (id, symbol, ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold,
                     bb_period, bb_stddev, use_bb_filter,
                     macd_fast, macd_slow, macd_signal, use_macd_filter,
                     use_volume_filter, volume_min_multiplier,
                     atr_sl_multiplier, atr_tp_multiplier, min_adx, min_atr_pips, max_spread_pips,
                     use_ema_cross, use_macd_cross, use_bb_bounce,
                     profit_factor, total_trades, win_rate, max_drawdown, sharpe, generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (g.id, symbol, g.ema_fast, g.ema_slow, g.rsi_period,
                      g.rsi_overbought, g.rsi_oversold,
                      g.bb_period, g.bb_stddev, int(g.use_bb_filter),
                      g.macd_fast, g.macd_slow, g.macd_signal, int(g.use_macd_filter),
                      int(g.use_volume_filter), g.volume_min_multiplier,
                      g.atr_sl_multiplier, g.atr_tp_multiplier, g.min_adx, g.min_atr_pips, g.max_spread_pips,
                      int(g.use_ema_cross), int(g.use_macd_cross), int(g.use_bb_bounce),
                      g.profit_factor, g.total_trades, g.win_rate,
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
                bb_period=d.get('bb_period', 20),
                bb_stddev=d.get('bb_stddev', 2.0),
                use_bb_filter=bool(d.get('use_bb_filter', 1)),
                macd_fast=d.get('macd_fast', 12),
                macd_slow=d.get('macd_slow', 26),
                macd_signal=d.get('macd_signal', 9),
                use_macd_filter=bool(d.get('use_macd_filter', 1)),
                use_volume_filter=bool(d.get('use_volume_filter', 1)),
                volume_min_multiplier=d.get('volume_min_multiplier', 0.8),
                atr_sl_multiplier=d.get('atr_sl_multiplier', 2.0),
                atr_tp_multiplier=d.get('atr_tp_multiplier', 4.0),
                min_adx=d.get('min_adx', 20.0),
                min_atr_pips=d.get('min_atr_pips', 5.0),
                max_spread_pips=d.get('max_spread_pips', 5.0),
                use_ema_cross=bool(d.get('use_ema_cross', 1)),
                use_macd_cross=bool(d.get('use_macd_cross', 1)),
                use_bb_bounce=bool(d.get('use_bb_bounce', 0)),
                profit_factor=d.get('profit_factor', 0),
                total_trades=d.get('total_trades', 0),
                win_rate=d.get('win_rate', 0),
                max_drawdown=d.get('max_drawdown', 0),
                sharpe=d.get('sharpe', 0),
                generation=d.get('generation', 0),
                id=d.get('id', ''),
            )
        except Exception as e:
            logger.error(f"Failed to load genome: {e}")
            return None

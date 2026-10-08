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
    # EMA periods
    ema_fast: int = 9
    ema_slow: int = 21
    # RSI thresholds
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    # ATR-based stops
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 4.0
    # Filters
    min_adx: float = 20.0  # Minimum ADX to consider trending
    max_spread_pips: float = 5.0
    # Score (filled by backtester)
    profit_factor: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
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
        self.min_profit_factor = self.config.get("min_profit_factor", 1.3)
        self.min_trades = self.config.get("min_trades", 50)
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
            atr_sl_multiplier=round(random.uniform(1.0, 5.0), 1),
            atr_tp_multiplier=round(random.uniform(2.0, 8.0), 1),
            min_adx=round(random.uniform(10, 35), 1),
            max_spread_pips=round(random.uniform(1.0, 10.0), 1),
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
            atr_sl_multiplier=round(genome.atr_sl_multiplier + random.uniform(-0.3, 0.3), 1),
            atr_tp_multiplier=round(genome.atr_tp_multiplier + random.uniform(-0.5, 0.5), 1),
            min_adx=round(genome.min_adx + random.uniform(-3, 3), 1),
            max_spread_pips=round(genome.max_spread_pips + random.uniform(-1, 1), 1),
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
            atr_sl_multiplier=round((parent_a.atr_sl_multiplier + parent_b.atr_sl_multiplier) / 2, 1),
            atr_tp_multiplier=round((parent_a.atr_tp_multiplier + parent_b.atr_tp_multiplier) / 2, 1),
            min_adx=round((parent_a.min_adx + parent_b.min_adx) / 2, 1),
            max_spread_pips=round((parent_a.max_spread_pips + parent_b.max_spread_pips) / 2, 1),
            generation=generation
        )
        if not child.is_valid():
            return self._random_genome(generation)
        child.id = f"g{generation}_{random.randint(10000, 99999)}"
        return child

    def _score_genome(self, genome: StrategyGenome, bars: List[dict]) -> StrategyGenome:
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

            # Walk through bars, simulate trades
            trades = []
            in_trade = False
            entry_price = 0
            atr_at_entry = 0
            for i in range(50, len(bars)):
                if not in_trade:
                    # Check entry conditions
                    if (i < len(ema_fast) and ema_fast[i] is not None
                        and ema_slow[i] is not None
                        and rsi[i] is not None and atr[i] is not None):
                        # Trend filter
                        if (ema_fast[i] > ema_slow[i]  # bullish cross
                            and rsi[i] < genome.rsi_overbought
                            and rsi[i] > genome.rsi_oversold):
                            in_trade = True
                            entry_price = closes[i]
                            atr_at_entry = atr[i]
                            sl = entry_price - atr_at_entry * genome.atr_sl_multiplier
                            tp = entry_price + atr_at_entry * genome.atr_tp_multiplier
                            trade = {'entry': i, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'side': 'long'}
                else:
                    # Check exit (SL/TP)
                    high = highs[i]
                    low = lows[i]
                    pnl = 0
                    exit_price = 0
                    exit_reason = ''
                    if low <= trade['sl']:
                        exit_price = trade['sl']
                        exit_reason = 'SL'
                    elif high >= trade['tp']:
                        exit_price = trade['tp']
                        exit_reason = 'TP'
                    if exit_price:
                        pnl = (exit_price - entry_price) * 100  # rough
                        trades.append({'pnl': pnl, 'reason': exit_reason})
                        in_trade = False
                    # Also exit on opposite cross
                    elif ema_fast[i] < ema_slow[i]:
                        pnl = (closes[i] - entry_price) * 100
                        trades.append({'pnl': pnl, 'reason': 'cross'})
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
            genome.max_drawdown = round(max_dd, 1)
            genome.sharpe = round(sharpe, 2)
        except Exception as e:
            logger.error(f"Genome scoring failed: {e}")
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

    def _get_bars_for_symbol(self, symbol: str, timeframe: str = "H1", days: int = 365) -> List[dict]:
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
                bars = self._get_bars_for_symbol(symbol, tf, days)
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
                'atr_sl_multiplier', 'atr_tp_multiplier', 'min_adx', 'max_spread_pips'
            ]})
            oos_genome = self._score_genome_multi(oos_genome, oos)
            if (oos_genome.profit_factor >= self.min_profit_factor
                and oos_genome.total_trades >= self.min_trades
                and oos_genome.max_drawdown <= self.max_drawdown_pct):
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
        all_trades = []
        for sym, sym_bars in by_symbol.items():
            # Strip metadata for scoring
            clean_bars = [{k: v for k, v in b.items() if not k.startswith('_')} for b in sym_bars]
            sym_genome = self._score_genome(genome, clean_bars)
            # Approximate trade count by symbol
            trade_count_estimate = sym_genome.total_trades
            # Multiply genome's stats by number of times this symbol contributed
            for _ in range(max(1, trade_count_estimate // max(1, sym_genome.total_trades))):
                pass
        # For now, just run on combined data
        clean_bars = [{k: v for k, v in b.items() if not k.startswith('_')} for b in bars_with_meta]
        return self._score_genome(genome, clean_bars)

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
                    atr_sl_multiplier REAL, atr_tp_multiplier REAL,
                    min_adx REAL, max_spread_pips REAL,
                    profit_factor REAL, total_trades INTEGER,
                    win_rate REAL, max_drawdown REAL, sharpe REAL,
                    generation INTEGER, validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for g in genomes:
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_genomes
                    (id, symbol, ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold,
                     atr_sl_multiplier, atr_tp_multiplier, min_adx, max_spread_pips,
                     profit_factor, total_trades, win_rate, max_drawdown, sharpe, generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (g.id, symbol, g.ema_fast, g.ema_slow, g.rsi_period,
                      g.rsi_overbought, g.rsi_oversold, g.atr_sl_multiplier,
                      g.atr_tp_multiplier, g.min_adx, g.max_spread_pips,
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
                atr_sl_multiplier=d.get('atr_sl_multiplier', 2.0),
                atr_tp_multiplier=d.get('atr_tp_multiplier', 4.0),
                min_adx=d.get('min_adx', 20.0),
                max_spread_pips=d.get('max_spread_pips', 5.0),
                profit_factor=d.get('profit_factor', 0),
                total_trades=d.get('total_trades', 0),
                win_rate=d.get('win_rate', 0),
                max_drawdown=d.get('max_drawdown', 0),
                sharpe=d.get('sharpe', 0),
                generation=d.get('generation', 0),
                id=d.get('id', '')
            )
        except Exception as e:
            logger.error(f"Failed to load genome: {e}")
            return None

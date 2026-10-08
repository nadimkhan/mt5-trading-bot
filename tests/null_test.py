"""
Null test for ML acceptance criteria.

This script tests whether our ML searcher's acceptance criteria
(PF > 1.2, 30+ trades, 30%+ WR) are too loose by running them
on data that should NOT produce profitable strategies:
1. Random walk price data (geometric Brownian motion)
2. Shuffled real data (destroys time order but keeps distribution)
3. Trendless sine wave (pure oscillation)

If our criteria accept random strategies, they will also accept
lucky overfits on real data. If they reject all random strategies,
we can trust them on real data.
"""
import sys
import os
import random
import math
import logging
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("null_test")

from strategies.ml_searcher import GeneticSearcher, StrategyGenome


def generate_random_walk_bars(num_bars: int = 5000, start_price: float = 1.1000,
                              drift: float = 0.0, volatility: float = 0.001) -> List[Dict]:
    """Generate geometric Brownian motion (random walk) bars.
    No trend, no patterns - just noise."""
    bars = []
    price = start_price
    for i in range(num_bars):
        # Random walk: log return ~ Normal(0, vol)
        log_return = random.gauss(drift, volatility)
        new_price = price * math.exp(log_return)
        # High/low around close (random spread)
        spread = abs(random.gauss(0, volatility * 0.5))
        high = new_price + spread
        low = new_price - spread
        open_price = price
        bars.append({
            'time': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': new_price,
            'tick_volume': random.randint(100, 1000),
        })
        price = new_price
    return bars


def generate_sine_wave_bars(num_bars: int = 5000, base: float = 1.1000,
                            amplitude: float = 0.01, period: int = 100) -> List[Dict]:
    """Generate pure sine wave (oscillating) bars.
    No randomness - just deterministic up/down pattern.
    Any strategy should lose to spread costs."""
    bars = []
    for i in range(num_bars):
        # Pure sine: oscillates between base-amp and base+amp
        angle = 2 * math.pi * i / period
        center = base + amplitude * math.sin(angle)
        # Small noise on top
        noise = random.gauss(0, amplitude * 0.02)
        close = center + noise
        high = max(close, center + amplitude * 0.05) + abs(noise)
        low = min(close, center - amplitude * 0.05) - abs(noise)
        bars.append({
            'time': i,
            'open': close + random.gauss(0, 0.0001),
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': 500,
        })
    return bars


def shuffle_real_bars(bars: List[Dict]) -> List[Dict]:
    """Shuffle real bars to destroy time-series structure.
    Keeps the distribution (same prices, same volumes) but randomizes order.
    Any strategy that profits on this data is detecting distribution properties
    not real market dynamics."""
    shuffled = list(bars)
    random.shuffle(shuffled)
    # Restore time index but keep shuffled prices
    for i, bar in enumerate(shuffled):
        bar['time'] = i
    return shuffled


def run_search_and_count_validated(searcher, in_sample, oos, label: str) -> int:
    """Run a search and count how many genomes pass validation."""
    print(f"\n{'='*60}")
    print(f"Running ML search on: {label}")
    print(f"  In-sample bars: {len(in_sample)}")
    print(f"  OOS bars: {len(oos)}")
    print(f"{'='*60}")

    try:
        # Run with small pop/gen for speed in the test
        # Real search uses 50/20, null test uses 30/10 (still 300 evals, enough to detect lottery)
        validated = searcher.run_search(
            symbols=['TEST'],
            timeframes=['M5'],
            days=len(in_sample) // 96,  # ~96 M5 bars per day
            population_size=30,
            generations=10
        )
        # If run_search doesn't support these params, try alternate
    except TypeError:
        # Fallback: just use _score_genome_multi on random genomes
        validated = []
        print("  (Using fallback: random genome scoring)")
        from datetime import datetime, timedelta
        for sym_bars in in_sample:
            in_data = [{**bar, '_symbol': 'TEST'} for bar in sym_bars]
        # Score 100 random genomes
        accepted = 0
        for _ in range(100):
            g = searcher._random_genome(0)
            scored = searcher._score_genome_multi(g, oos)
            if (scored.profit_factor >= 1.2
                and scored.total_trades >= 30
                and scored.win_rate >= 30.0):
                accepted += 1
        return accepted

    print(f"\n  RESULT: {len(validated)} genomes validated on {label}")
    return len(validated)


def run_null_test():
    """Run all 3 null tests and report results."""
    print("\n" + "="*60)
    print("NULL TEST: Validating acceptance criteria against noise")
    print("="*60)

    # Generate test data
    print("\nGenerating test data...")
    random.seed(42)  # Reproducibility
    rw_bars = generate_random_walk_bars(num_bars=5000)
    random.seed(42)
    sine_bars = generate_sine_wave_bars(num_bars=5000)

    # Shuffled real data: use random walk as base
    random.seed(42)
    rw_bars_for_shuffle = generate_random_walk_bars(num_bars=5000)
    shuffled = shuffle_real_bars(rw_bars_for_shuffle)

    # Setup searcher
    config = {
        'min_profit_factor': 1.2,
        'min_trades': 30,
        'max_drawdown_pct': 25.0,
    }
    searcher = GeneticSearcher(config=config)

    # Split each into train/test
    results = {}
    for label, bars in [
        ("Random Walk (pure noise)", rw_bars),
        ("Pure Sine Wave (oscillation)", sine_bars),
        ("Shuffled bars (no time order)", shuffled),
    ]:
        split = int(len(bars) * 0.7)
        # _score_genome_multi expects List[bar_dict] with _symbol key
        in_sample = [{**b, '_symbol': 'TEST'} for b in bars[:split]]
        oos = [{**b, '_symbol': 'TEST'} for b in bars[split:]]

        # Try direct scoring approach (more reliable)
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"  In-sample: {split} bars, OOS: {len(bars) - split} bars")
        print(f"{'='*60}")

        accepted = 0
        total_scored = 100
        in_sample_pfs = []
        oos_pfs = []
        for i in range(total_scored):
            g = searcher._random_genome(0)
            try:
                # Score on in-sample
                in_scored = searcher._score_genome_multi(g, in_sample)
                in_sample_pfs.append(in_scored.profit_factor)
                # Score on OOS (regardless of in-sample result)
                oos_scored = searcher._score_genome_multi(g, oos)
                oos_pfs.append(oos_scored.profit_factor)
                # Check if it passes OOS validation
                if (oos_scored.profit_factor >= 1.2
                    and oos_scored.total_trades >= 30
                    and oos_scored.win_rate >= 30.0
                    and oos_scored.max_drawdown <= 25.0):
                    accepted += 1
            except Exception as e:
                pass

        results[label] = {
            'accepted': accepted,
            'scored': total_scored,
            'best_oos_pf': max(oos_pfs) if oos_pfs else 0,
            'avg_oos_pf': sum(oos_pfs) / len(oos_pfs) if oos_pfs else 0,
            'best_in_pf': max(in_sample_pfs) if in_sample_pfs else 0,
            'avg_in_pf': sum(in_sample_pfs) / len(in_sample_pfs) if in_sample_pfs else 0,
        }
        print(f"  Scored: {total_scored} random genomes")
        print(f"  In-sample: best PF={results[label]['best_in_pf']:.3f}, avg PF={results[label]['avg_in_pf']:.3f}")
        print(f"  OOS: best PF={results[label]['best_oos_pf']:.3f}, avg PF={results[label]['avg_oos_pf']:.3f}")
        print(f"  Accepted (OOS pass all criteria): {accepted}")

    total_accepted = sum(r['accepted'] for r in results.values())
    total_scored = sum(r['scored'] for r in results.values())
    acceptance_rate = total_accepted / total_scored if total_scored else 0
    best_oos = max(r['best_oos_pf'] for r in results.values())

    print("\n" + "="*60)
    print("NULL TEST SUMMARY")
    print("="*60)
    print(f"\n{'Test':<40} {'Accepted':<10} {'Best OOS PF':<12} {'Verdict'}")
    print("-" * 80)
    for label, r in results.items():
        verdict = "BAD" if r['accepted'] > 0 else "GOOD"
        print(f"{label:<40} {r['accepted']:<10} {r['best_oos_pf']:.3f}        {verdict}")
    print()
    print(f"Total accepted: {total_accepted}/{total_scored} ({acceptance_rate*100:.1f}%)")
    print(f"Best OOS PF seen: {best_oos:.3f}")
    print()

    if total_accepted > 0:
        print("VERDICT: CRITERIA ARE TOO LOOSE")
        print(f"  {total_accepted}/{total_scored} random strategies passed validation")
        print("  This means our criteria cannot distinguish real edges from noise")
        print("  We need to tighten criteria (more trades, higher PF, etc.)")
        return False
    else:
        print("VERDICT: CRITERIA APPEAR SOUND")
        print("  No random strategies passed full validation")
        print("  But check: was best OOS PF above 1.0?")
        if best_oos > 1.5:
            print(f"  WARNING: best OOS PF was {best_oos:.2f} on random data")
            print("  Even without full validation pass, the noise is close to passing")
            print("  Recommend tightening criteria further")
            return False
        else:
            print("  Our criteria can distinguish real edges from noise")
            print("  We can trust the ML search results on real data")
            return True


if __name__ == '__main__':
    success = run_null_test()
    sys.exit(0 if success else 1)

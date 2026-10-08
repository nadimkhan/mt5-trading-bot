"""Test regime detection on synthetic + real-looking data"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import random
import math
from strategies.regime_aware import RegimeAwareStrategy


def make_bull_data(n=200, start=1.1, drift=0.001, volatility=0.005):
    """Bull trending data"""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * math.exp(random.gauss(drift, volatility)))
    return prices


def make_bear_data(n=200, start=1.1, drift=-0.001, volatility=0.005):
    """Bear trending data"""
    return make_bull_data(n, start, drift, volatility)


def make_sideways_data(n=200, start=1.1, amplitude=0.02, period=30):
    """Sideways oscillating data"""
    prices = []
    for i in range(n):
        prices.append(start + amplitude * math.sin(2 * math.pi * i / period))
    return prices


def make_volatile_data(n=200, start=1.1, volatility=0.02):
    """Volatile data (big swings)"""
    return make_bull_data(n, start, 0, volatility)


def prices_to_ohlc(prices):
    """Convert price series to OHLC bars (simplified)"""
    highs = [p * (1 + abs(random.gauss(0, 0.001))) for p in prices]
    lows = [p * (1 - abs(random.gauss(0, 0.001))) for p in prices]
    opens = [prices[0]] + prices[:-1]
    volumes = [random.randint(100, 1000) for _ in prices]
    return {'closes': prices, 'highs': highs, 'lows': lows, 'opens': opens, 'volumes': volumes}


# Test 1: Bull market detection
print('=' * 70)
print('TEST 1: BULL TRENDING DATA')
print('=' * 70)
random.seed(42)
bull_prices = make_bull_data(n=300, drift=0.0008, volatility=0.003)
bull_data = prices_to_ohlc(bull_prices)
strategy = RegimeAwareStrategy()
result = strategy.check_entry(bull_data)
print(f"Detected regime: {result['regime']}")
print(f"  Reason: {result.get('regime_reason', 'N/A')}")
print(f"  ADX: {result.get('adx', 0):.1f}")
print(f"  Signal: {result.get('signal', 'N/A')}")
print(f"  Reason: {result.get('reason', 'N/A')[:80]}")

# Test 2: Bear market
print('\n' + '=' * 70)
print('TEST 2: BEAR TRENDING DATA')
print('=' * 70)
random.seed(42)
bear_prices = make_bear_data(n=300, drift=-0.0008, volatility=0.003)
bear_data = prices_to_ohlc(bear_prices)
result = strategy.check_entry(bear_data)
print(f"Detected regime: {result['regime']}")
print(f"  Reason: {result.get('regime_reason', 'N/A')}")
print(f"  ADX: {result.get('adx', 0):.1f}")
print(f"  Signal: {result.get('signal', 'N/A')}")
print(f"  Reason: {result.get('reason', 'N/A')[:80]}")

# Test 3: Sideways
print('\n' + '=' * 70)
print('TEST 3: SIDEWAYS OSCILLATING DATA')
print('=' * 70)
random.seed(42)
sideways_prices = make_sideways_data(n=300, amplitude=0.015, period=25)
sideways_data = prices_to_ohlc(sideways_prices)
result = strategy.check_entry(sideways_data)
print(f"Detected regime: {result['regime']}")
print(f"  Reason: {result.get('regime_reason', 'N/A')}")
print(f"  ADX: {result.get('adx', 0):.1f}")
print(f"  Signal: {result.get('signal', 'N/A')}")
print(f"  Reason: {result.get('reason', 'N/A')[:80]}")

# Test 4: Volatile
print('\n' + '=' * 70)
print('TEST 4: VOLATILE DATA')
print('=' * 70)
random.seed(42)
vol_prices = make_volatile_data(n=300, volatility=0.025)
vol_data = prices_to_ohlc(vol_prices)
result = strategy.check_entry(vol_data)
print(f"Detected regime: {result['regime']}")
print(f"  Reason: {result.get('regime_reason', 'N/A')}")
print(f"  ADX: {result.get('adx', 0):.1f}")
print(f"  Signal: {result.get('signal', 'N/A')}")
print(f"  Reason: {result.get('reason', 'N/A')[:80]}")

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print("If regime detection works correctly:")
print("  - Bull data → BULL regime")
print("  - Bear data → BEAR regime")
print("  - Sideways data → SIDEWAYS regime")
print("  - Volatile data → VOLATILE regime (skip)")

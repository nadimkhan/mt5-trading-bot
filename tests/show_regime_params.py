"""Show the regime parameter sets for review"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
from strategies.regime_aware import REGIME_PARAMS

print('=' * 70)
print('PRE-RESEARCHED PARAMETER SETS (per regime)')
print('=' * 70)
for regime, params in REGIME_PARAMS.items():
    if params.get('skip'):
        print(f'\n{regime}: SKIP (no trading)')
    else:
        print(f'\n{regime}: {params["name"]}')
        print(f'  Description: {params["description"]}')
        print(f'  Expected win rate: {params["expected_win_rate"]}%')
        print(f'  Expected R:R: {params["expected_rr"]}')
        for k, v in params.items():
            if k not in ('name', 'description', 'expected_win_rate', 'expected_rr'):
                print(f'  {k}: {v}')

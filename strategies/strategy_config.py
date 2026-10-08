"""
Strategy Configurations - User-configurable parameters
Each strategy has its own config that can be edited via the dashboard.
"""
import json
import os
import logging
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.constants import STRATEGY_CONFIGS_PATH
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

CONFIG_FILE = STRATEGY_CONFIGS_PATH
_lock = Lock()

# Default configurations for each strategy
# Valid timeframe options for MT5
TIMEFRAME_OPTIONS = ["M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
                     "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1"]

DEFAULT_CONFIGS = {
    "scalp": {
        "name": "Scalp Strategy",
        "description": "Multi-timeframe trend + pullback + confirmation",
        "enabled": True,
        "timeframes": {
            "trend": {"value": "H4", "type": "select", "options": TIMEFRAME_OPTIONS,
                      "label": "Trend Timeframe", "description": "Higher timeframe for trend direction (H4/H1/D1)"},
            "entry": {"value": "M15", "type": "select", "options": TIMEFRAME_OPTIONS,
                      "label": "Entry Timeframe", "description": "Timeframe for pullback entry setup (M15/M5)"},
            "confirm": {"value": "M5", "type": "select", "options": TIMEFRAME_OPTIONS,
                        "label": "Confirm Timeframe", "description": "Timeframe for entry confirmation (M5/M1)"},
            "scalp_interval": {"value": 15, "type": "int", "min": 5, "max": 300, "step": 5,
                              "label": "Scalp Check Interval (sec)", "description": "How often to check for new entries"},
            "trend_interval": {"value": 60, "type": "int", "min": 30, "max": 3600, "step": 30,
                               "label": "Trend Update Interval (sec)", "description": "How often to update trend analysis"}
        },
        "parameters": {
            # EMA periods
            "ema_fast": {"value": 9, "type": "int", "min": 3, "max": 50, "label": "EMA Fast Period", "description": "Fast EMA for short-term momentum"},
            "ema_medium": {"value": 20, "type": "int", "min": 5, "max": 100, "label": "EMA Medium Period", "label2": "EMA 20", "description": "Medium EMA for pullback detection"},
            "ema_slow": {"value": 50, "type": "int", "min": 10, "max": 200, "label": "EMA Slow Period", "description": "Slow EMA for trend confirmation"},
            "ema_trend": {"value": 200, "type": "int", "min": 50, "max": 500, "label": "EMA Trend Period", "description": "Trend-defining EMA (H4)"},
            # Entry criteria
            "pullback_max_pips": {"value": 15, "type": "int", "min": 1, "max": 100, "label": "Max Pullback (pips)", "description": "Max distance from EMA20 for entry"},
            "min_confidence": {"value": 60, "type": "int", "min": 0, "max": 100, "label": "Min Confidence %", "description": "Minimum confidence to take a trade"},
            # Risk
            "atr_sl_multiplier": {"value": 3.0, "type": "float", "min": 0.5, "max": 10.0, "step": 0.1, "label": "ATR SL Multiplier", "description": "Stop Loss = ATR x this value"},
            "atr_tp_multiplier": {"value": 5.0, "type": "float", "min": 0.5, "max": 20.0, "step": 0.1, "label": "ATR TP Multiplier", "description": "Take Profit = ATR x this value"},
            "min_risk_reward": {"value": 1.5, "type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "label": "Min Risk:Reward", "description": "Reject trades below this RR"},
            # Filters
            "rsi_overbought": {"value": 70, "type": "int", "min": 50, "max": 95, "label": "RSI Overbought", "description": "Skip BUY if RSI above this"},
            "rsi_oversold": {"value": 30, "type": "int", "min": 5, "max": 50, "label": "RSI Oversold", "description": "Skip SELL if RSI below this"},
            "max_spread_pips": {"value": 3.0, "type": "float", "min": 0.1, "max": 20.0, "step": 0.1, "label": "Max Spread (pips)", "description": "Skip trade if spread exceeds this"},
            "max_open_trades": {"value": 2, "type": "int", "min": 1, "max": 10, "label": "Max Open Trades", "description": "Maximum concurrent positions"},
            "max_trades_per_day": {"value": 10, "type": "int", "min": 1, "max": 50, "label": "Max Trades Per Day", "description": "Stop trading after N trades in a day"},
            "daily_loss_limit_pct": {"value": 3.0, "type": "float", "min": 0.5, "max": 10.0, "step": 0.5, "label": "Daily Loss Limit %", "description": "Stop trading if daily loss exceeds this"},
            "loss_streak_reduction_pct": {"value": 50, "type": "int", "min": 0, "max": 100, "label": "Loss Streak Reduction %", "description": "Reduce lot size by this % after consecutive losses (0=disabled)"},
            "loss_streak_threshold": {"value": 2, "type": "int", "min": 1, "max": 10, "label": "Loss Streak Threshold", "description": "Number of consecutive losses before reducing size"},
            "win_streak_increase_pct": {"value": 0, "type": "int", "min": 0, "max": 100, "label": "Win Streak Bonus %", "description": "Increase lot size by this % after consecutive wins (0=disabled)"},
            "win_streak_threshold": {"value": 3, "type": "int", "min": 1, "max": 10, "label": "Win Streak Threshold", "description": "Number of consecutive wins before bonus size"},
        }
    },
    "trend": {
        "name": "Trend Following",
        "description": "EMA crossover trend following",
        "enabled": True,
        "timeframes": {
            "trend": {"value": "H4", "type": "select", "options": TIMEFRAME_OPTIONS,
                      "label": "Trend Timeframe", "description": "Higher timeframe for trend bias"},
            "entry": {"value": "M15", "type": "select", "options": TIMEFRAME_OPTIONS,
                      "label": "Entry Timeframe", "description": "Timeframe for crossover signal"},
            "confirm": {"value": "M5", "type": "select", "options": TIMEFRAME_OPTIONS,
                        "label": "Confirm Timeframe", "description": "Timeframe for confirmation"},
            "scalp_interval": {"value": 60, "type": "int", "min": 5, "max": 300, "step": 5,
                              "label": "Check Interval (sec)", "description": "How often to check for new entries"},
            "trend_interval": {"value": 300, "type": "int", "min": 30, "max": 3600, "step": 30,
                               "label": "Trend Update Interval (sec)", "description": "How often to update trend analysis"}
        },
        "parameters": {
            "ema_fast": {"value": 9, "type": "int", "min": 3, "max": 50, "label": "EMA Fast Period"},
            "ema_slow": {"value": 21, "type": "int", "min": 5, "max": 100, "label": "EMA Slow Period"},
            "min_confidence": {"value": 50, "type": "int", "min": 0, "max": 100, "label": "Min Confidence %"},
            "atr_sl_multiplier": {"value": 2.5, "type": "float", "min": 0.5, "max": 10.0, "step": 0.1, "label": "ATR SL Multiplier"},
            "atr_tp_multiplier": {"value": 6.0, "type": "float", "min": 0.5, "max": 20.0, "step": 0.1, "label": "ATR TP Multiplier"},
            "min_risk_reward": {"value": 2.0, "type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "label": "Min Risk:Reward"},
            "rsi_overbought": {"value": 75, "type": "int", "min": 50, "max": 95, "label": "RSI Overbought"},
            "rsi_oversold": {"value": 25, "type": "int", "min": 5, "max": 50, "label": "RSI Oversold"},
            "max_spread_pips": {"value": 5.0, "type": "float", "min": 0.1, "max": 20.0, "step": 0.1, "label": "Max Spread (pips)"},
            "max_open_trades": {"value": 3, "type": "int", "min": 1, "max": 10, "label": "Max Open Trades"},
            "max_trades_per_day": {"value": 15, "type": "int", "min": 1, "max": 50, "label": "Max Trades Per Day"},
            "daily_loss_limit_pct": {"value": 3.0, "type": "float", "min": 0.5, "max": 10.0, "step": 0.5, "label": "Daily Loss Limit %"},
            "loss_streak_reduction_pct": {"value": 40, "type": "int", "min": 0, "max": 100, "label": "Loss Streak Reduction %"},
            "loss_streak_threshold": {"value": 2, "type": "int", "min": 1, "max": 10, "label": "Loss Streak Threshold"},
            "win_streak_increase_pct": {"value": 20, "type": "int", "min": 0, "max": 100, "label": "Win Streak Bonus %"},
            "win_streak_threshold": {"value": 3, "type": "int", "min": 1, "max": 10, "label": "Win Streak Threshold"},
        }
    }
}


def load_configs():
    """Load strategy configs from disk, or return defaults"""
    with _lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults (so new params appear)
                merged = {}
                for key, default in DEFAULT_CONFIGS.items():
                    merged[key] = saved.get(key, default)
                    # Merge timeframes
                    if 'timeframes' in default:
                        if 'timeframes' not in merged[key]:
                            merged[key]['timeframes'] = default['timeframes']
                        else:
                            for tfkey, tfval in default['timeframes'].items():
                                if tfkey not in merged[key]['timeframes']:
                                    merged[key]['timeframes'][tfkey] = tfval
                                elif 'options' in tfval:
                                    # Keep saved value but ensure options are present
                                    if 'options' not in merged[key]['timeframes'][tfkey]:
                                        merged[key]['timeframes'][tfkey]['options'] = tfval['options']
                    # Merge parameters
                    if 'parameters' in merged[key]:
                        for pkey, pval in default['parameters'].items():
                            if pkey not in merged[key]['parameters']:
                                merged[key]['parameters'][pkey] = pval
                            elif 'value' in pval and pkey in merged[key]['parameters']:
                                # Keep saved value but keep default metadata
                                if 'min' in pval:
                                    merged[key]['parameters'][pkey]['min'] = pval['min']
                                if 'max' in pval:
                                    merged[key]['parameters'][pkey]['max'] = pval['max']
                return merged
            except Exception as e:
                logger.error(f"Failed to load configs: {e}")
        return DEFAULT_CONFIGS.copy()


def save_configs(configs):
    """Save strategy configs to disk"""
    with _lock:
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(configs, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save configs: {e}")
            return False


def reset_configs():
    """Reset configs to defaults"""
    with _lock:
        try:
            if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)
            return DEFAULT_CONFIGS.copy()
        except Exception as e:
            logger.error(f"Failed to reset configs: {e}")
            return None


def get_strategy_param(configs, strategy_name, param_name):
    """Get a specific parameter value"""
    try:
        return configs[strategy_name]['parameters'][param_name]['value']
    except (KeyError, TypeError):
        # Return default
        return DEFAULT_CONFIGS[strategy_name]['parameters'][param_name]['value']


def update_strategy_param(configs, strategy_name, param_name, new_value):
    """Update a single parameter or timeframe and validate"""
    if strategy_name not in configs:
        return False, f"Unknown strategy: {strategy_name}"

    # Check parameters first
    if param_name in configs[strategy_name].get('parameters', {}):
        param = configs[strategy_name]['parameters'][param_name]
        # Validate value
        try:
            val = type(param['value'])(new_value)
            if 'min' in param and val < param['min']:
                return False, f"Value below minimum ({param['min']})"
            if 'max' in param and val > param['max']:
                return False, f"Value above maximum ({param['max']})"
            param['value'] = val
            return True, "OK"
        except (ValueError, TypeError) as e:
            return False, f"Invalid value type: {e}"

    # Check timeframes
    if param_name in configs[strategy_name].get('timeframes', {}):
        param = configs[strategy_name]['timeframes'][param_name]
        # Validate select type (must be in options) or int
        try:
            if param.get('type') == 'select':
                if new_value not in param.get('options', []):
                    return False, f"Invalid option. Must be one of: {', '.join(param.get('options', []))}"
                param['value'] = str(new_value)
            else:
                val = type(param['value'])(new_value)
                if 'min' in param and val < param['min']:
                    return False, f"Value below minimum ({param['min']})"
                if 'max' in param and val > param['max']:
                    return False, f"Value above maximum ({param['max']})"
                param['value'] = val
            return True, "OK"
        except (ValueError, TypeError) as e:
            return False, f"Invalid value type: {e}"

    return False, f"Unknown parameter: {param_name}"

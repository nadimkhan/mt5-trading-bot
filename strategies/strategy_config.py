"""
Strategy Configurations - User-configurable parameters
Each strategy has its own config that can be edited via the dashboard.
"""
import json
import os
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

CONFIG_FILE = "E:/projects/mt5-trading-bot/strategy_configs.json"
_lock = Lock()

# Default configurations for each strategy
DEFAULT_CONFIGS = {
    "scalp": {
        "name": "Scalp Strategy",
        "description": "H4 trend + M15 pullback + M1 confirmation",
        "enabled": True,
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
            "daily_loss_limit_pct": {"value": 3.0, "type": "float", "min": 0.5, "max": 10.0, "step": 0.5, "label": "Daily Loss Limit %", "description": "Stop trading if daily loss exceeds this"},
        }
    },
    "trend": {
        "name": "Trend Following",
        "description": "EMA crossover trend following",
        "enabled": True,
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
            "daily_loss_limit_pct": {"value": 3.0, "type": "float", "min": 0.5, "max": 10.0, "step": 0.5, "label": "Daily Loss Limit %"},
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
    """Update a single parameter and validate"""
    if strategy_name not in configs:
        return False, f"Unknown strategy: {strategy_name}"
    if param_name not in configs[strategy_name]['parameters']:
        return False, f"Unknown parameter: {param_name}"
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

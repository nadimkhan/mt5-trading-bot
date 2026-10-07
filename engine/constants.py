"""
Shared constants and utilities used across modules.
"""
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database (single source of truth)
DB_PATH = os.path.join(PROJECT_ROOT, "bot.db")
DB_PATH = DB_PATH.replace("\\", "/")  # Use forward slashes for consistency

# Config file
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CONFIG_PATH = CONFIG_PATH.replace("\\", "/")

# Strategy configs
STRATEGY_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "strategy_configs.json")
STRATEGY_CONFIGS_PATH = STRATEGY_CONFIGS_PATH.replace("\\", "/")

# Logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

"""
Trade Counter - Tracks daily trades and win/loss streaks
Adjusts lot size based on streak history
"""
import sqlite3
import logging
from datetime import datetime, date
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

DB_PATH = "E:/projects/mt5-trading-bot/bot.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_trades_today() -> int:
    """Count trades opened today"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(opened_at) = ?", (today,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_recent_results(count: int = 10) -> list:
    """Get last N closed trade results (True for win, False for loss)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pnl FROM trades WHERE status='CLOSED' AND pnl IS NOT NULL ORDER BY closed_at DESC LIMIT ?", (count,))
    rows = cursor.fetchall()
    conn.close()
    return [r['pnl'] > 0 for r in rows]


def get_streak() -> Tuple[int, str]:
    """
    Get current streak info.
    Returns (streak_count, 'win' or 'loss')
    """
    results = get_recent_results(20)
    if not results:
        return 0, "none"
    last = results[0]
    streak = 0
    for r in results:
        if r == last:
            streak += 1
        else:
            break
    return streak, ("win" if last else "loss")


def calculate_lot_adjustment(loss_reduction_pct: int = 50, loss_threshold: int = 2,
                               win_increase_pct: int = 0, win_threshold: int = 3) -> float:
    """
    Calculate lot size multiplier based on current streak.
    Returns multiplier (e.g., 0.5 = 50% of normal, 1.2 = 120% of normal, 1.0 = no change)
    """
    streak, streak_type = get_streak()
    if streak_type == "loss" and streak >= loss_threshold and loss_reduction_pct > 0:
        # Reduce lot size after consecutive losses
        reduction = (100 - loss_reduction_pct) / 100.0
        logger.info(f"Loss streak {streak}: reducing lot to {reduction*100:.0f}% of normal")
        return reduction
    elif streak_type == "win" and streak >= win_threshold and win_increase_pct > 0:
        # Increase lot size after consecutive wins
        increase = (100 + win_increase_pct) / 100.0
        logger.info(f"Win streak {streak}: increasing lot to {increase*100:.0f}% of normal")
        return increase
    return 1.0


def get_daily_pnl() -> float:
    """Get total P&L for today"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()
    cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='CLOSED' AND DATE(closed_at) = ?", (today,))
    pnl = cursor.fetchone()[0] or 0
    conn.close()
    return pnl


def can_trade_today(max_trades: int, max_daily_loss_pct: float, account_balance: float) -> Tuple[bool, str]:
    """
    Check if we can take another trade today.
    Returns (can_trade, reason)
    """
    trades_today = get_trades_today()
    if trades_today >= max_trades:
        return False, f"Max trades per day reached ({trades_today}/{max_trades})"
    daily_pnl = get_daily_pnl()
    if account_balance > 0:
        loss_pct = abs(daily_pnl) / account_balance * 100
        if daily_pnl < 0 and loss_pct >= max_daily_loss_pct:
            return False, f"Daily loss limit reached ({loss_pct:.1f}% >= {max_daily_loss_pct}%)"
    return True, "OK"


def get_trade_stats() -> dict:
    """Get trade statistics for display"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(opened_at) = ?", (today,))
    trades_today = cursor.fetchone()[0]
    cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='CLOSED' AND DATE(closed_at) = ?", (today,))
    daily_pnl = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl > 0")
    total_wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl <= 0")
    total_losses = cursor.fetchone()[0]
    conn.close()
    streak, streak_type = get_streak()
    return {
        'trades_today': trades_today,
        'daily_pnl': daily_pnl,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'streak_count': streak,
        'streak_type': streak_type,
        'win_rate': (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    }

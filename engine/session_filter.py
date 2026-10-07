"""
Session Filter - Trade only during optimal market hours
- London/NY overlap for majors
- High-impact news avoidance
"""
import logging
from datetime import datetime, time
import pytz

logger = logging.getLogger(__name__)

# Major forex sessions in UTC
SESSIONS = {
    "LONDON": {"start": 7, "end": 16},      # 7am-4pm UTC
    "NEW_YORK": {"start": 12, "end": 21},   # 12pm-9pm UTC
    "TOKYO": {"start": 0, "end": 9},        # 12am-9am UTC
    "SYDNEY": {"start": 22, "end": 7},      # 10pm-7am UTC
}

# High impact news events to avoid (approximate times in UTC)
# Format: (month, day) ranges or specific events
HIGH_IMPACT_NEWS = {
    # Non-Farm Payrolls - First Friday of each month ~13:30 UTC
    "NFP": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months
    # FOMC Meetings - 8x per year, dates TBD but meetings are always on days 1-2 of some months
    "FOMC": [3, 6, 9, 12],  # March, June, September, December
    # CPI releases - typically mid-month
    "CPI_US": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


class SessionFilter:
    """Filter for trading session and news events"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Session settings
        session_config = self.config.get("sessions", {})
        self.trade_london_ny_overlap = session_config.get("london_ny_overlap", True)
        self.trade_majors_only = session_config.get("majors_only", True)
        
        # News filter settings
        news_config = self.config.get("news_filter", {})
        self.avoid_high_impact = news_config.get("avoid_high_impact", True)
        self.news_window_minutes = news_config.get("window_minutes", 30)  # Minutes before/after to avoid
        
        # Timezone
        self.tz = pytz.UTC
        
        # High impact news dates (would be fetched from API in production)
        self._news_dates = self._load_news_calendar()
    
    def _load_news_calendar(self) -> list:
        """Load high-impact news dates for current month"""
        # In production, this would fetch from an economic calendar API
        # For now, return empty - user must maintain calendar
        return []
    
    def is_trading_allowed(self, symbol: str = None) -> dict:
        """
        Check if trading is allowed right now.
        Returns dict with 'allowed' bool and 'reason' string.
        """
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        
        # Check session filter
        session_check = self._check_session(current_hour, symbol)
        if not session_check["allowed"]:
            return session_check
        
        # Check news filter
        news_check = self._check_news(now_utc)
        if not news_check["allowed"]:
            return news_check
        
        return {"allowed": True, "reason": "Session and news check passed"}
    
    def _check_session(self, hour: int, symbol: str = None) -> dict:
        """Check if current hour is in a tradeable session"""
        
        # Define major pairs that benefit from London/NY overlap
        majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
        
        if self.trade_london_ny_overlap:
            # London: 7-16 UTC, NY: 12-21 UTC
            # Overlap: 12-16 UTC (4 hours)
            london_open = 7
            london_close = 16
            ny_open = 12
            ny_close = 21
            
            in_london = london_open <= hour < london_close
            in_ny = ny_open <= hour < ny_close
            
            # For majors, require London/NY overlap
            if symbol and symbol in majors:
                if in_london and in_ny:
                    return {"allowed": True, "reason": f"London/NY overlap ({hour}:00 UTC)"}
                else:
                    return {
                        "allowed": False,
                        "reason": f"Outside London/NY overlap. London: {london_open}-{london_close}, NY: {ny_open}-{ny_close} UTC"
                    }
            else:
                # For other pairs, allow if either session is open
                if in_london or in_ny:
                    session = "London" if in_london else "NY"
                    return {"allowed": True, "reason": f"{session} session open ({hour}:00 UTC)"}
                else:
                    return {"allowed": False, "reason": f"Both sessions closed ({hour}:00 UTC)"}
        
        # If not requiring overlap, just check if any major session is open
        for session_name, times in SESSIONS.items():
            if times["start"] <= hour < times["end"]:
                return {"allowed": True, "reason": f"{session_name} session open"}
        
        return {"allowed": False, "reason": f"No trading session open ({hour}:00 UTC)"}
    
    def _check_news(self, now: datetime) -> dict:
        """Check if we're in a high-impact news window"""
        if not self.avoid_high_impact:
            return {"allowed": True, "reason": "News filter disabled"}
        
        # Check if today is a known high-impact news day
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        
        # NFP - First Friday of month at 13:30 UTC
        if month in HIGH_IMPACT_NEWS["NFP"]:
            if self._is_first_friday(now) and hour == 13 and minute >= 0:
                return {"allowed": False, "reason": "NFP release - avoiding news window"}
        
        # Check 30-minute window around news
        # This is simplified - in production would use actual news times
        news_windows = self._get_news_windows(now)
        for news_time in news_windows:
            diff_minutes = abs((now - news_time).total_seconds() / 60)
            if diff_minutes <= self.news_window_minutes:
                return {"allowed": False, "reason": f"High-impact news - {self.news_window_minutes}min window"}
        
        return {"allowed": True, "reason": "No news events nearby"}
    
    def _is_first_friday(self, dt: datetime) -> bool:
        """Check if given date is the first Friday of the month"""
        return dt.weekday() == 4 and dt.day <= 7
    
    def _get_news_windows(self, now: datetime):
        """Get high-impact news times for today"""
        # In production, this would query an economic calendar API
        # Returns list of datetime objects for today's news events
        return []
    
    def get_next_trade_window(self) -> dict:
        """Get the next available trading window"""
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        
        if self.trade_london_ny_overlap:
            # Next overlap window
            if current_hour < 12:
                next_start = 12  # NY opens
            elif current_hour < 16:
                return {"allowed": True, "reason": "Currently in London/NY overlap", "minutes_remaining": (16 - current_hour) * 60 - now_utc.minute}
            else:
                # Next day at 12 UTC
                return {"allowed": False, "reason": "Next London/NY overlap: Tomorrow 12:00 UTC", "next_start_utc": "12:00"}
        
        return {"allowed": True, "reason": "Check individual sessions"}
    
    def is_major_pair(self, symbol: str) -> bool:
        """Check if symbol is a major pair requiring overlap"""
        majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
        return symbol in majors

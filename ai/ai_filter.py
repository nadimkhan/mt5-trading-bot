"""
AI Filter - AI only vetoes or approves setups
AI doesn't make decisions - it filters them based on news and regime
"""
import logging
import json
from typing import Dict, List

logger = logging.getLogger(__name__)


class AIFilter:
    """
    AI acts as a filter on rule-based setups.
    - Does NOT make trading decisions
    - Only vetoes based on: news events, regime, unexpected conditions
    - Approves if conditions are favorable
    """
    
    def __init__(self, provider: str, api_key: str, model: str = None,
                 max_tokens: int = 500, temperature: float = 0.3):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enabled = bool(api_key)
        
    def evaluate_setup(self, symbol: str, setup: Dict, market_data: Dict, 
                      news_events: List = None, regime: Dict = None) -> Dict:
        """
        AI evaluates a setup and returns approval/veto.
        
        Args:
            symbol: Trading symbol
            setup: Rule-based setup from strategy (signal, confidence, entry, SL, TP)
            market_data: Current market conditions
            news_events: Any upcoming news events
            regime: Market regime (TRENDING/RANGING)
            
        Returns:
            Dict with 'approved' bool, 'reason', 'adjustments'
        """
        if not self.enabled:
            # No AI - auto-approve
            return {
                "approved": True,
                "reason": "AI disabled - auto-approved",
                "adjustments": {}
            }
        
        try:
            # Build context for AI
            context = self._build_context(symbol, setup, market_data, news_events, regime)
            
            # Query AI
            response = self._query_ai(context)
            
            # Parse response
            result = self._parse_response(response, setup)
            
            logger.info(f"AI Filter: {symbol} {setup.get('signal')} -> {'APPROVED' if result['approved'] else 'VETOED'}: {result['reason']}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI Filter error: {e}")
            # On error, default to approve but log
            return {
                "approved": True,
                "reason": f"AI error ({str(e)[:50]}) - defaulting to approve",
                "adjustments": {}
            }
    
    def _build_context(self, symbol: str, setup: Dict, market_data: Dict,
                      news_events: List, regime: Dict) -> str:
        """Build prompt context for AI"""
        
        signal = setup.get("signal", "UNKNOWN")
        confidence = setup.get("confidence", 0)
        entry = setup.get("entry_zone", setup.get("current_price"))
        sl = setup.get("stop_loss")
        tp = setup.get("take_profit")
        
        # Market regime info
        regime_text = "UNKNOWN"
        if regime:
            regime_text = f"{regime.get('regime', 'UNKNOWN')} (ADX: {regime.get('adx', 'N/A')})"
        
        # News info
        news_text = "None scheduled"
        if news_events:
            news_text = ", ".join([f"{n.get('time')} - {n.get('impact', 'UNKNOWN')}" for n in news_events[:3]])
        
        # Current market conditions
        m15_data = market_data.get('M15', {})
        rsi = m15_data.get('rsi', 'N/A')
        spread = m15_data.get('spread', 'N/A')
        
        context = f"""You are a RISK MANAGER evaluating a trading setup. Your job is to VETO dangerous setups, not approve good ones.

SETUP DETAILS:
- Symbol: {symbol}
- Signal: {signal}
- Confidence: {confidence}%
- Entry: {entry}
- Stop Loss: {sl}
- Take Profit: {tp}

MARKET CONDITIONS:
- Regime: {regime_text}
- RSI(14): {rsi}
- Spread: {spread} pips
- News: {news_text}

ANALYSIS CHECKLIST:
1. Is the regime favorable for this direction? (Trending vs Ranging)
2. Is there high-impact news in the next 2 hours?
3. Is RSI extreme (overbought >70, oversold <30)?
4. Is spread unusually wide?
5. Are there any red flags that should veto this setup?

Respond ONLY in JSON format:
{{"approved": true/false, "reason": "specific reason", "adjustments": {{"sl": new_sl, "tp": new_tp, "lot_size": new_size}}}}

If you see no major issues, approve. If anything concerns you, veto with specific reason.
JSON response:"""
        
        return context
    
    def _query_ai(self, context: str) -> str:
        """Query the AI provider"""
        if self.provider == "groq":
            return self._query_groq(context)
        elif self.provider == "claude":
            return self._query_claude(context)
        elif self.provider == "openai":
            return self._query_openai(context)
        else:
            return '{"approved": true, "reason": "Unknown provider", "adjustments": {}}'
    
    def _query_groq(self, context: str) -> str:
        """Query Groq API"""
        import requests
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model or "qwen/qwen3.8-27b",
            "messages": [{"role": "user", "content": context}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} {response.text}")
            return '{"approved": true, "reason": "API error", "adjustments": {}}'
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    
    def _query_claude(self, context: str) -> str:
        """Query Claude API"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self.model or "claude-sonnet-4-20250514",
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": context}]
        )
        
        return response.content[0].text
    
    def _query_openai(self, context: str) -> str:
        """Query OpenAI API"""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[{"role": "user", "content": context}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _parse_response(self, response: str, setup: Dict) -> Dict:
        """Parse AI JSON response"""
        try:
            # Try to extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            # Find JSON object
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = json_str[start:end]
            
            result = json.loads(json_str)
            
            # Validate
            if not isinstance(result.get("approved"), bool):
                result["approved"] = True
                result["reason"] = result.get("reason", "Parse error - approved")
            
            if "adjustments" not in result:
                result["adjustments"] = {}
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {
                "approved": True,
                "reason": f"Parse error - approving by default",
                "adjustments": {}
            }


class NewsChecker:
    """Check for upcoming high-impact news events"""
    
    def __init__(self):
        # High impact news - in production would use economic calendar API
        self.high_impact_events = {
            "USD": ["NFP", "FOMC", "CPI", "GDP", "PMI"],
            "EUR": ["ECB", "CPI_EU", "GDP_EU"],
            "GBP": ["BOE", "CPI_UK", "GDP_UK"],
            "JPY": ["BOJ", "CPI_JP"],
        }
    
    def check_upcoming_news(self, symbol: str, hours_ahead: int = 2) -> List[Dict]:
        """
        Check for upcoming high-impact news.
        In production, would query an economic calendar API.
        Returns list of news events.
        """
        # Placeholder - returns empty for now
        # In production: integrate with forexfactory, investing.com, or similar API
        return []
    
    def is_news_window(self, symbol: str, minutes: int = 30) -> bool:
        """Check if currently in a news window to avoid"""
        events = self.check_upcoming_news(symbol, hours_ahead=1)
        return len(events) > 0

"""
AI Analyzer - Connects to cloud AI providers (Claude, OpenAI, MiniMax, Groq)
"""
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, provider="claude", api_key=None, model=None, max_tokens=1000, temperature=0.7):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Provider-specific settings
        self.endpoints = {
            "claude": "https://api.anthropic.com/v1/messages",
            "openai": "https://api.openai.com/v1/chat/completions",
            "minimax": "https://api.minimax.io/v1/chat/completions",  # Correct endpoint
            "groq": "https://api.groq.com/openai/v1/chat/completions"
        }
        
        self.default_models = {
            "claude": "claude-3-5-haiku-20241022",
            "openai": "gpt-4o-mini",
            "minimax": "MiniMax-Text-01",
            "groq": "llama-3.2-3b-preview"
        }
        
    def _get_model(self):
        """Get the model to use"""
        if self.model:
            return self.model
        return self.default_models.get(self.provider, "gpt-4o-mini")
        
    def analyze_market(self, market_data, positions=None, trade_history=None):
        """
        Send market data to AI and get trading decision
        
        Args:
            market_data: dict with symbol analysis from indicators
            positions: list of current open positions
            trade_history: list of recent trades
            
        Returns:
            dict with AI's decision and reasoning
        """
        if not self.api_key:
            return {
                "error": "No API key configured",
                "decision": "SKIP",
                "reasoning": "AI provider not configured"
            }
            
        # Build the prompt
        prompt = self._build_analysis_prompt(market_data, positions, trade_history)
        
        try:
            if self.provider == "claude":
                return self._call_claude(prompt, market_data)
            elif self.provider == "openai":
                return self._call_openai(prompt, market_data)
            elif self.provider == "minimax":
                return self._call_minimax(prompt, market_data)
            elif self.provider == "groq":
                return self._call_groq(prompt, market_data)
            else:
                return {
                    "error": f"Unknown provider: {self.provider}",
                    "decision": "SKIP",
                    "reasoning": "Unknown AI provider"
                }
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "error": str(e),
                "decision": "SKIP",
                "reasoning": f"AI call failed: {str(e)}"
            }
            
    def _build_analysis_prompt(self, market_data, positions, trade_history):
        """Build the analysis prompt for the AI - Scalping strategy M15/M5/M1"""

        # Format multi-timeframe market data for prompt
        symbols_analysis = ""
        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                continue

            # Extract trend direction
            trend_dir = data.get('trend_direction', 'SIDEWAYS')

            # Extract data per timeframe (new M15/M5/M1)
            m15_data = data.get('M15', {})
            m5_data = data.get('M5', {})
            m1_data = data.get('M1', {})

            # Get EMA crossover info
            m5_ema = m5_data.get('ema_cross', {})
            m1_ema = m1_data.get('ema_cross', {})

            symbols_analysis += f"""
{symbol} - SCALPING ANALYSIS (M15/M5/M1):

=== M15 TREND (Intra-session direction) ===
- Overall: {m15_data.get('overall', 'N/A')}
- Trend: {m15_data.get('trend', 'N/A')}
- RSI: {m15_data.get('rsi', 'N/A')} ({m15_data.get('rsi_signal', 'N/A')})
- MACD: {m15_data.get('macd', {}).get('trend', 'N/A')}
- EMA Cross: {m15_data.get('ema_cross', {}).get('signal', 'N/A')}

=== M5 ENTRY (Setup zone) ===
- Price: {m5_data.get('bid', 'N/A')}
- Spread: {m5_data.get('spread', 'N/A')} pips
- RSI: {m5_data.get('rsi', 'N/A')} ({m5_data.get('rsi_signal', 'N/A')})
- RSI 7: {m5_data.get('rsi_7', 'N/A')} (faster)
- EMA 9/21: {m5_ema.get('ema_9', 'N/A')} / {m5_ema.get('ema_21', 'N/A')} ({m5_ema.get('signal', 'N/A')})
- EMA Distance: {m5_ema.get('distance_pips', 'N/A')} pips
- Bollinger: Upper {m5_data.get('bollinger', {}).get('upper', 'N/A')}, Lower {m5_data.get('bollinger', {}).get('lower', 'N/A')}
- Support: {m5_data.get('support', 'N/A')}
- Resistance: {m5_data.get('resistance', 'N/A')}
- Overall: {m5_data.get('overall', 'N/A')}

=== M1 PRECISION (Exact entry) ===
- Price: {m1_data.get('bid', 'N/A')}
- RSI: {m1_data.get('rsi', 'N/A')}
- EMA 9/21: {m1_ema.get('ema_9', 'N/A')} / {m1_ema.get('ema_21', 'N/A')} ({m1_ema.get('signal', 'N/A')})
- ATR: {m5_data.get('atr', 'N/A')}

- Has Position: {data.get('has_position', False)}

"""

        # Format positions
        positions_text = "No open positions"
        if positions and len(positions) > 0:
            positions_text = "\n".join([
                f"- {p['symbol']} {p['type']} {p['volume']}lots @ {p['price_open']} (P/L: ${p['profit']:.2f})"
                for p in positions
            ])

        # Format recent trades
        trades_text = "No recent trades"
        if trade_history and len(trade_history) > 0:
            recent = trade_history[-5:]
            trades_text = "\n".join([
                f"- {t['symbol']} {t['type']} {t.get('exit_time', 'N/A')}: {'+' if t.get('pnl', 0) > 0 else ''}{t.get('pnl', 0):.2f} ({t.get('strategy', 'N/A')})"
                for t in recent
            ])

        prompt = f"""You are a professional forex scalping AI. Use M15/M5/M1 multi-timeframe analysis.

SCALPING STRATEGY:
- M15 sets intra-session trend direction
- M5 identifies trade setups with EMA 9/21 crossover + RSI
- M1 confirms exact entry timing
- Only trade in M15 trend direction (BUY in uptrend, SELL in downtrend)
- Quick scalp trades: 10-30 pips targets, stops 10-15 pips

CURRENT MARKET DATA:
{symbols_analysis}

OPEN POSITIONS:
{positions_text}

RECENT TRADES:
{trades_text}

Your task: Decide what action to take for each symbol.

For each symbol, output a JSON decision:
{{
  "symbol": "EURUSD",
  "action": "BUY" or "SELL" or "HOLD",
  "lot_size": 0.01 to 0.10,
  "entry_reason": "Why you're entering",
  "stop_loss_pips": 10 to 15,
  "take_profit_pips": 10 to 30,
  "confidence": 50 to 95,
  "risk_level": "LOW" or "MEDIUM" or "HIGH"
}}

Rules:
- BUY only if: M15 trend is BULL and M5 EMA cross is BULLISH and RSI(7) < 60
- SELL only if: M15 trend is BEAR and M5 EMA cross is BEARISH and RSI(7) > 40
- HOLD if: M15 trend is SIDEWAYS or M5/M1 disagree with M15
- Avoid: RSI(7) > 70 (overbought) or < 30 (oversold) for entries
- Scalp targets: 10-30 pips, stop loss 10-15 pips max
- Max risk per trade: 2% of account

Output JSON only, no other text. If no clear setup, output HOLD for that symbol."""

        return prompt
        
    def _call_claude(self, prompt, market_data):
        """Call Claude API"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self._get_model(),
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_ai_response(response.content[0].text, market_data)
        
    def _call_openai(self, prompt, market_data):
        """Call OpenAI API"""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self._get_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return self._parse_ai_response(response.choices[0].message.content, market_data)
        
    def _call_minimax(self, prompt, market_data):
        """Call MiniMax API using Anthropic SDK format (for M Plan subscription)"""
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url="https://api.minimax.io/anthropic"
        )

        response = client.messages.create(
            model="MiniMax-M3.1-Flash-Preview",
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from response content blocks
        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                content_text += block.thinking

        return self._parse_ai_response(content_text, market_data)
        
    def _call_groq(self, prompt, market_data):
        """Call Groq API"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self._get_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(
            self.endpoints["groq"],
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return {"error": f"Groq error: {response.text}", "decision": "SKIP"}
            
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return self._parse_ai_response(content, market_data)
        
    def _parse_ai_response(self, content, market_data):
        """Parse AI response into structured decision"""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle AI thinking tags (MiniMax models)
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()

            # Handle JSON wrapped in markdown
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end]
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end]

            # Try to find JSON object or array
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]

                # Handle multiple JSON objects - find all complete objects with symbol and action
                import re
                # Find all potential JSON objects
                decisions = []

                # Simple approach: find all {"symbol":...} patterns
                pattern = r'\{"symbol":\s*"([^"]+)"[^}]*\}'
                matches = re.findall(pattern, json_str)

                # Actually parse properly - find all { ... "symbol": "..." ... "action": "..." ... }
                # Split by "symbol" to find each object
                parts = json_str.split('"symbol":')
                for i, part in enumerate(parts[1:], 1):  # Skip first empty part
                    try:
                        # Find the full object - from "symbol" to the closing }
                        # Find where this object ends
                        bracket_count = 1
                        pos = 0
                        obj_str = '"symbol":' + part
                        for j, c in enumerate(part):
                            if c == '{':
                                bracket_count += 1
                            elif c == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    pos = j + 1
                                    break
                        full_obj_str = '{"symbol":' + part[:pos]
                        candidate = json.loads(full_obj_str)
                        if 'symbol' in candidate and 'action' in candidate:
                            decisions.append(candidate)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

                if decisions:
                    # Return the first decision with all decisions attached
                    decision = decisions[0]
                    decision['_all_decisions'] = decisions
                else:
                    # Fall back to full parsing
                    try:
                        decision = json.loads(json_str)
                    except json.JSONDecodeError:
                        raise json.JSONDecodeError("No valid JSON with symbol/action found", json_str, 0)

            decision["raw_response"] = content
            decision["timestamp"] = datetime.now()

            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"Content: {content[:500]}")
            return {
                "error": "Failed to parse AI response",
                "raw_response": content[:500],
                "decision": "SKIP",
                "reasoning": "Could not understand AI response"
            }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with mock data
    analyzer = AIAnalyzer(provider="claude", api_key="test")
    
    market_data = {
        "EURUSD": {
            "current_price": 1.12150,
            "trend": "BULL_TREND",
            "rsi": 38,
            "rsi_signal": "OVERSOLD",
            "macd": {"histogram": 0.0005, "trend": "BULLISH"},
            "bollinger": {"upper": 1.1250, "lower": 1.1180},
            "atr": 15,
            "support": 1.1200,
            "resistance": 1.1240,
            "overall": "BULLISH",
            "confidence": 72,
            "change_1h": {"percent": 0.15}
        }
    }
    
    print("Market data prepared")
    print("AI Analyzer initialized")

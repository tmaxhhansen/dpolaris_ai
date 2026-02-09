"""
dPolaris AI - Main AI Engine

The core intelligence combining:
- Claude API for reasoning and analysis
- Claude CLI for deep research with web access
- Local ML models for predictions
- Persistent memory for learning
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncIterator
import asyncio
import logging

import anthropic

from .config import Config, get_config
from .database import Database
from .memory import DPolarisMemory
from .claude_cli import ClaudeCLI, ClaudeCLIPool

logger = logging.getLogger("dpolaris.ai")


class DPolarisAI:
    """
    Main AI engine for dPolaris.

    Orchestrates:
    - Direct Claude API calls for analysis
    - Claude CLI for web research
    - Local ML model predictions
    - Memory-based learning and personalization
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Database] = None,
        memory: Optional[DPolarisMemory] = None,
    ):
        self.config = config or get_config()
        self.db = db or Database()
        self.memory = memory or DPolarisMemory(self.db)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.config.api_keys.anthropic,
        )

        # Claude CLI for research
        self.claude_cli = ClaudeCLI() if self.config.ai.use_claude_cli else None
        self.cli_pool = ClaudeCLIPool(max_concurrent=2)

        # Session management
        self.session_id = str(uuid.uuid4())[:8]
        self.conversation_history: list[dict] = []

        # Load system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with memory context"""
        base_prompt = self._get_base_prompt()

        # Add portfolio context
        portfolio = self.db.get_latest_portfolio()
        if portfolio:
            portfolio_context = f"""
## CURRENT PORTFOLIO STATE
- Total Value: ${portfolio.get('total_value', 0):,.2f}
- Cash: ${portfolio.get('cash', 0):,.2f}
- Daily P/L: ${portfolio.get('daily_pnl', 0):+,.2f}
- Goal Progress: {portfolio.get('goal_progress', 0):.1f}%
"""
        else:
            portfolio_context = "\n## PORTFOLIO: Not yet initialized\n"

        # Add memory context
        memory_context = self.memory.build_context(max_tokens=1500)

        # Add recent trade stats
        trade_stats = self.db.get_trade_stats()
        if trade_stats.get("total_trades"):
            trade_context = f"""
## TRADING STATISTICS
- Total Trades: {trade_stats['total_trades']}
- Win Rate: {trade_stats.get('win_rate', 0):.1f}%
- Average Win: ${trade_stats.get('avg_win', 0):,.2f}
- Average Loss: ${trade_stats.get('avg_loss', 0):,.2f}
- Profit Factor: {trade_stats.get('profit_factor', 0):.2f}
"""
        else:
            trade_context = "\n## TRADING STATISTICS: No closed trades yet\n"

        # Combine all context
        full_prompt = f"{base_prompt}\n{portfolio_context}\n{trade_context}\n{memory_context}"
        return full_prompt

    def _get_base_prompt(self) -> str:
        """Get base system prompt"""
        return f"""# dPolaris AI - Trading Intelligence System

You are dPolaris_ai, an elite trading intelligence system designed to help reach a portfolio goal of ${self.config.goal.target:,.0f}.

## PRIME DIRECTIVES

1. **CAPITAL PRESERVATION FIRST** - Protecting capital always supersedes making profits.
2. **PROBABILISTIC THINKING** - Express confidence levels, never certainties.
3. **GOAL AWARENESS** - Every analysis connects back to the goal.
4. **INTELLECTUAL HONESTY** - Acknowledge uncertainty clearly.
5. **ANTI-FRAGILITY** - Prefer positions that benefit from volatility.

## CAPABILITIES

You have access to:
1. **Direct Analysis** - Instant analysis using your knowledge
2. **Deep Research** - Claude CLI for web research (when @research is used)
3. **ML Predictions** - Local trained models (when @predict is used)
4. **Trade Journal** - Historical trades and patterns
5. **Memory System** - Learnings and user preferences

## COMMANDS

- `@scout` - Scan for trading opportunities
- `@analyze SYMBOL` - Deep analysis of a symbol
- `@research SYMBOL` - Web research using Claude CLI
- `@predict SYMBOL` - ML model prediction
- `@risk` - Portfolio risk assessment
- `@journal` - Trade journal operations
- `@learn` - Record a learning/insight
- `@performance` - Performance metrics
- `@regime` - Market regime assessment

## RISK PARAMETERS

- Max position size: {self.config.risk.max_position_size_percent}%
- Max portfolio risk per trade: {self.config.risk.max_portfolio_risk_percent}%
- Max correlated exposure: {self.config.risk.max_correlated_exposure}%
- Minimum cash reserve: {self.config.risk.min_cash_reserve_percent}%
- Max acceptable drawdown: {self.config.risk.max_drawdown_percent}%

## COMMUNICATION STYLE

- Be direct and lead with conclusions
- Use numbers and quantify everything
- Show your reasoning
- Flag uncertainty with confidence levels
- End with specific actionable recommendations
- No hype words ("moon", "guaranteed", "easy")

## CONFIDENCE CALIBRATION

- HIGH (>70%): "I expect..." / "This will likely..."
- MEDIUM (50-70%): "This may..." / "Reasonable chance..."
- LOW (<50%): "It's possible but uncertain..."

Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

    async def chat(self, user_message: str) -> str:
        """
        Process a chat message and return response.

        Handles special commands and routes to appropriate handlers.
        """
        # Save user message
        self.db.save_conversation(self.session_id, "user", user_message)
        self.conversation_history.append({"role": "user", "content": user_message})

        # Check for special commands
        if user_message.strip().startswith("@"):
            response = await self._handle_command(user_message)
        else:
            response = await self._chat_with_claude(user_message)

        # Save assistant response
        self.db.save_conversation(self.session_id, "assistant", response)
        self.conversation_history.append({"role": "assistant", "content": response})

        # Learn from interaction if applicable
        self._extract_learnings(user_message, response)

        return response

    async def _handle_command(self, message: str) -> str:
        """Handle special @ commands"""
        parts = message.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "@scout": self._cmd_scout,
            "@analyze": self._cmd_analyze,
            "@research": self._cmd_research,
            "@predict": self._cmd_predict,
            "@risk": self._cmd_risk,
            "@journal": self._cmd_journal,
            "@learn": self._cmd_learn,
            "@performance": self._cmd_performance,
            "@regime": self._cmd_regime,
            "@train": self._cmd_train,
        }

        handler = handlers.get(command)
        if handler:
            return await handler(args)
        else:
            return await self._chat_with_claude(message)

    async def _chat_with_claude(self, message: str) -> str:
        """Direct chat with Claude API"""
        try:
            # Build messages with recent history
            messages = self.conversation_history[-self.config.ai.conversation_memory_limit:]

            response = self.client.messages.create(
                model=self.config.ai.model,
                max_tokens=self.config.ai.max_tokens,
                system=self.system_prompt,
                messages=messages,
                temperature=self.config.ai.temperature,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error communicating with Claude: {str(e)}"

    # ==================== Command Handlers ====================

    async def _cmd_scout(self, args: str) -> str:
        """Scout for opportunities"""
        prompt = f"""Scan for trading opportunities based on current market conditions.

Focus on:
1. High IV rank stocks for premium selling (IV rank > 50%)
2. Strong momentum stocks with pullbacks to support
3. Upcoming earnings plays with favorable risk/reward
4. Sector rotation opportunities

Watchlist to prioritize: {', '.join(self.config.watchlist)}

{args if args else ''}

Provide a structured scout report with:
- Top 3 high-conviction opportunities
- Key levels and suggested strategies
- Risk factors for each
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_analyze(self, symbol: str) -> str:
        """Deep analysis of a symbol"""
        if not symbol:
            return "Please specify a symbol to analyze. Usage: @analyze AAPL"

        symbol = symbol.upper().strip()

        # Get any memories about this symbol
        symbol_context = self.memory.build_symbol_context(symbol)

        prompt = f"""Perform comprehensive analysis on {symbol}.

{symbol_context}

Analyze:
1. **Fundamental**: Business quality, financials, valuation
2. **Technical**: Trend, key levels, momentum
3. **Options**: IV rank, options flow, strategy opportunities
4. **Catalysts**: Upcoming events, earnings, news

Provide:
- Executive summary with bull/bear stance
- Conviction score (1-10)
- Key levels to watch
- Recommended strategy (if any)
- Risk factors
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_research(self, symbol: str) -> str:
        """Deep research using Claude CLI"""
        if not symbol:
            return "Please specify a symbol to research. Usage: @research AAPL"

        if not self.claude_cli:
            return "Claude CLI not available. Using standard analysis instead."

        symbol = symbol.upper().strip()

        # Use Claude CLI for web research
        result = await self.claude_cli.analyze_stock(symbol, "comprehensive")

        if result.success:
            # Also do a quick Claude analysis
            synthesis_prompt = f"""Based on this research about {symbol}:

{result.content}

Synthesize the key findings and provide:
1. Investment thesis (bull or bear)
2. Key risks
3. Suggested action
"""
            synthesis = await self._chat_with_claude(synthesis_prompt)
            return f"## Research Results for {symbol}\n\n{result.content}\n\n---\n\n## Synthesis\n\n{synthesis}"
        else:
            return f"Research failed: {result.error}. Falling back to standard analysis."

    async def _cmd_predict(self, symbol: str) -> str:
        """ML model prediction"""
        if not symbol:
            return "Please specify a symbol. Usage: @predict AAPL"

        symbol = symbol.upper().strip()

        # Check for available models
        try:
            from ml import Predictor
            from tools.market_data import fetch_historical_data

            predictor = Predictor()
            models = predictor.list_available_models()

            if not models:
                return "No ML models trained yet. Use @train to train a model first."

            # Get historical data
            df = await fetch_historical_data(symbol, days=300)
            if df is None or df.empty:
                return f"Could not fetch data for {symbol}"

            # Make prediction with available model
            model_name = models[0]["name"]
            signal = predictor.generate_trading_signal(model_name, df, symbol)

            return f"""## ML Prediction for {symbol}

**Signal**: {signal['signal']}
**Direction**: {signal['direction']}
**Confidence**: {signal['confidence']:.1%}
**Probability Up**: {signal['probability_up']:.1%}

### Context
- RSI: {signal['context']['rsi']:.1f}
- Trend: {signal['context']['trend']}
- Volatility: {signal['context']['volatility']}
- Momentum: {signal['context']['momentum']}

*Model: {signal['model']}*
*Note: This is a model prediction, not financial advice.*
"""

        except ImportError as e:
            return f"ML module not available: {e}"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Prediction failed: {str(e)}"

    async def _cmd_risk(self, args: str) -> str:
        """Portfolio risk assessment"""
        positions = self.db.get_open_positions()
        portfolio = self.db.get_latest_portfolio()

        positions_text = "\n".join([
            f"- {p['symbol']}: {p['quantity']} @ ${p['entry_price']:.2f} "
            f"(P/L: ${p.get('unrealized_pnl', 0):.2f})"
            for p in positions
        ]) if positions else "No open positions"

        prompt = f"""Assess portfolio risk:

**Portfolio**:
- Total Value: ${portfolio.get('total_value', 0) if portfolio else 0:,.2f}
- Cash: ${portfolio.get('cash', 0) if portfolio else 0:,.2f}

**Open Positions**:
{positions_text}

**Risk Parameters**:
- Max position size: {self.config.risk.max_position_size_percent}%
- Max drawdown: {self.config.risk.max_drawdown_percent}%
- Min cash reserve: {self.config.risk.min_cash_reserve_percent}%

Analyze:
1. Position sizing compliance
2. Concentration risk
3. Correlation exposure
4. Greeks exposure (if options)
5. Drawdown status

Provide risk score and recommendations.
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_journal(self, args: str) -> str:
        """Trade journal operations"""
        trades = self.db.get_trades(limit=10)
        stats = self.db.get_trade_stats()

        trades_text = "\n".join([
            f"- {t['entry_date'][:10]}: {t['symbol']} {t['strategy']} "
            f"P/L: ${t.get('pnl', 0):.2f} ({t.get('pnl_percent', 0):.1f}%)"
            for t in trades
        ]) if trades else "No trades recorded"

        prompt = f"""Trade journal summary:

**Recent Trades**:
{trades_text}

**Statistics**:
- Total: {stats.get('total_trades', 0)}
- Win Rate: {stats.get('win_rate', 0):.1f}%
- Avg Win: ${stats.get('avg_win', 0):.2f}
- Avg Loss: ${stats.get('avg_loss', 0):.2f}
- Profit Factor: {stats.get('profit_factor', 0):.2f}

Analyze patterns and provide insights on:
1. What's working
2. What to improve
3. Strategy recommendations
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_learn(self, args: str) -> str:
        """Record a learning"""
        if not args:
            return "Please provide a learning to record. Usage: @learn [category] your learning"

        # Parse category if provided
        parts = args.split(maxsplit=1)
        if parts[0].lower() in DPolarisMemory.CATEGORIES:
            category = parts[0].lower()
            content = parts[1] if len(parts) > 1 else ""
        else:
            category = "market_insight"
            content = args

        if not content:
            return "Please provide content for the learning."

        self.memory.learn(category, content, importance=0.7)
        return f"Learned: [{category}] {content}"

    async def _cmd_performance(self, args: str) -> str:
        """Performance metrics"""
        stats = self.db.get_trade_stats()
        history = self.db.get_performance_history(days=90)
        strategy_rankings = self.memory.get_strategy_rankings()

        rankings_text = "\n".join([
            f"- {r['strategy']}: {r['win_rate']:.1f}% ({r['total_trades']} trades)"
            for r in strategy_rankings[:5]
        ]) if strategy_rankings else "Not enough data"

        portfolio = self.db.get_latest_portfolio()
        goal_progress = portfolio.get('goal_progress', 0) if portfolio else 0

        prompt = f"""Performance analysis:

**Goal Progress**: {goal_progress:.1f}% toward ${self.config.goal.target:,.0f}

**Trading Stats** (All Time):
- Trades: {stats.get('total_trades', 0)}
- Win Rate: {stats.get('win_rate', 0):.1f}%
- Total P/L: ${stats.get('total_pnl', 0):,.2f}
- Best Trade: ${stats.get('best_trade', 0):,.2f}
- Worst Trade: ${stats.get('worst_trade', 0):,.2f}

**Strategy Rankings**:
{rankings_text}

Analyze and provide:
1. Are we on track for the goal?
2. What adjustments are needed?
3. Strategy recommendations
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_regime(self, args: str) -> str:
        """Market regime assessment"""
        snapshot = self.db.get_latest_market_snapshot()

        if self.claude_cli:
            # Get fresh regime assessment
            result = await self.claude_cli.research_market_regime()
            if result.success:
                return f"## Market Regime Assessment\n\n{result.content}"

        prompt = """Assess the current market regime based on your knowledge:

1. **Trend**: Bull, Bear, or Transitional?
2. **Volatility**: Low (<15 VIX), Normal (15-20), Elevated (20-30), Crisis (>30)
3. **Breadth**: Healthy or Narrow?
4. **Fed Stance**: Hawkish, Neutral, or Dovish?

Provide:
- Current regime classification
- Key indicators to watch
- Recommended positioning adjustments
- Strategies favored in this regime
"""
        return await self._chat_with_claude(prompt)

    async def _cmd_train(self, args: str) -> str:
        """Train ML model"""
        parts = args.split() if args else []
        symbol = parts[0].upper() if parts else "SPY"

        try:
            from ml import ModelTrainer
            from tools.market_data import fetch_historical_data

            # Fetch data
            df = await fetch_historical_data(symbol, days=500)
            if df is None or df.empty:
                return f"Could not fetch data for {symbol}"

            # Train model
            trainer = ModelTrainer()
            result = trainer.train_full_pipeline(
                df,
                model_name=f"{symbol}_direction",
                target="target_direction",
                target_horizon=5,
            )

            return f"""## Model Training Complete

**Model**: {result['model_name']}
**Type**: {result['model_type']}
**Target**: {result['target']}

**Metrics**:
- Accuracy: {result['metrics']['accuracy']:.1%}
- Precision: {result['metrics']['precision']:.1%}
- Recall: {result['metrics']['recall']:.1%}
- F1 Score: {result['metrics']['f1']:.1%}

**Top Features**:
{chr(10).join([f"- {f[0]}: {f[1]:.3f}" for f in result['feature_importance'][:5]])}

Model saved to: {result['model_path']}
"""

        except Exception as e:
            logger.error(f"Training error: {e}")
            return f"Training failed: {str(e)}"

    # ==================== Learning ====================

    def _extract_learnings(self, user_message: str, response: str):
        """Extract potential learnings from interaction"""
        # Learn preferences from user messages
        lower_msg = user_message.lower()

        if "prefer" in lower_msg or "like" in lower_msg or "want" in lower_msg:
            self.memory.learn_preference("stated_preference", user_message)

        if "risk" in lower_msg and ("too much" in lower_msg or "concerned" in lower_msg):
            self.memory.learn_risk_observation(f"User expressed risk concern: {user_message}")

    # ==================== Streaming ====================

    async def stream_chat(self, user_message: str) -> AsyncIterator[str]:
        """Stream chat response"""
        self.db.save_conversation(self.session_id, "user", user_message)
        self.conversation_history.append({"role": "user", "content": user_message})

        full_response = ""

        try:
            with self.client.messages.stream(
                model=self.config.ai.model,
                max_tokens=self.config.ai.max_tokens,
                system=self.system_prompt,
                messages=self.conversation_history[-self.config.ai.conversation_memory_limit:],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            full_response = error_msg
            yield error_msg

        self.db.save_conversation(self.session_id, "assistant", full_response)
        self.conversation_history.append({"role": "assistant", "content": full_response})

    # ==================== Utilities ====================

    def refresh_system_prompt(self):
        """Refresh the system prompt with latest data"""
        self.system_prompt = self._build_system_prompt()

    def get_session_id(self) -> str:
        """Get current session ID"""
        return self.session_id

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())[:8]

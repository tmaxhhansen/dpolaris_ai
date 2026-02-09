"""
Claude CLI Integration for dPolaris AI

This module allows dPolaris to leverage the local Claude Code CLI
for enhanced analysis capabilities like:
- Deep web research on stocks/options
- Complex multi-step analysis
- File processing and document analysis
- Real-time market research
"""

import subprocess
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, AsyncIterator
from dataclasses import dataclass
import logging

logger = logging.getLogger("dpolaris.claude_cli")


@dataclass
class ClaudeResponse:
    """Response from Claude CLI"""
    content: str
    success: bool
    error: Optional[str] = None
    session_id: Optional[str] = None


class ClaudeCLI:
    """
    Interface to Claude Code CLI for enhanced analysis.

    Enables dPolaris to spawn Claude CLI sessions for tasks that benefit
    from Claude's tool use capabilities (web search, file analysis, etc.)
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        model: str = "sonnet",  # Use sonnet for most tasks, opus for complex
        max_turns: int = 10,
    ):
        self.working_dir = working_dir or Path.cwd()
        self.model = model
        self.max_turns = max_turns
        self._verify_cli_installed()

    def _verify_cli_installed(self) -> bool:
        """Check if Claude CLI is available"""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Claude CLI found: {result.stdout.strip()}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        logger.warning("Claude CLI not found. Some features will be limited.")
        return False

    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> ClaudeResponse:
        """
        Use Claude CLI to perform deep stock analysis with web research.

        Args:
            symbol: Stock ticker symbol
            analysis_type: 'comprehensive', 'technical', 'fundamental', 'news', 'options'
        """
        prompts = {
            "comprehensive": f"""Analyze {symbol} comprehensively:

1. Search for recent news and developments about {symbol}
2. Look up the company's recent earnings and financial metrics
3. Check analyst ratings and price targets
4. Identify any upcoming catalysts (earnings, product launches, etc.)
5. Assess the current technical setup if possible

Provide a structured analysis with:
- Executive Summary (bull/bear/neutral stance)
- Key Findings from your research
- Risk Factors
- Potential Catalysts
- Suggested Action (if any)

Be specific with numbers and dates. Cite your sources.""",

            "news": f"""Research the latest news and developments for {symbol}:

1. Search for news from the past week
2. Look for any SEC filings or insider transactions
3. Check for analyst upgrades/downgrades
4. Find any social media sentiment shifts

Summarize the key developments and their potential impact on the stock.""",

            "options": f"""Analyze options activity and setup for {symbol}:

1. Research current implied volatility levels
2. Look up upcoming events that could affect options pricing
3. Search for any unusual options activity reports
4. Check earnings date and expected move

Provide recommendations for options strategies based on your findings.""",

            "fundamental": f"""Deep dive into {symbol}'s fundamentals:

1. Research recent earnings reports and guidance
2. Look up revenue growth trends
3. Find profit margin information
4. Check valuation metrics vs peers
5. Research competitive position and moat

Provide a fundamental assessment with specific numbers.""",

            "technical": f"""Analyze {symbol}'s technical setup:

1. Research current price action and trend
2. Look for key support/resistance levels mentioned by analysts
3. Check relative strength vs market
4. Find any notable chart patterns being discussed

Provide technical levels and potential trade setups."""
        }

        prompt = prompts.get(analysis_type, prompts["comprehensive"])
        return await self.run_prompt(prompt, task_description=f"Analyzing {symbol}")

    async def research_market_regime(self) -> ClaudeResponse:
        """Use Claude CLI to assess current market regime"""
        prompt = """Assess the current market regime:

1. Search for the current VIX level and recent trend
2. Look up S&P 500 and Nasdaq positioning vs moving averages
3. Research recent Fed commentary and rate expectations
4. Check market breadth indicators if available
5. Find any notable sector rotations

Provide a market regime assessment:
- Current regime (Bull/Bear/Transition/Range-bound)
- Volatility environment (Low/Normal/Elevated/Crisis)
- Key risks to monitor
- Recommended positioning adjustments"""

        return await self.run_prompt(prompt, task_description="Market regime assessment")

    async def find_opportunities(self, criteria: dict) -> ClaudeResponse:
        """Use Claude CLI to scan for trading opportunities"""
        criteria_str = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])

        prompt = f"""Find trading opportunities matching these criteria:
{criteria_str}

1. Search for stocks meeting these criteria
2. Look for recent unusual options activity
3. Check for earnings plays in the next 2 weeks
4. Find high IV rank stocks for premium selling

For each opportunity found:
- Symbol and current price
- Why it matches the criteria
- Key levels to watch
- Suggested strategy
- Risk factors"""

        return await self.run_prompt(prompt, task_description="Opportunity scan")

    async def analyze_earnings(self, symbol: str) -> ClaudeResponse:
        """Deep earnings analysis using Claude CLI"""
        prompt = f"""Perform earnings analysis for {symbol}:

1. Find the next earnings date
2. Research analyst estimates (EPS and revenue)
3. Look up historical earnings surprises
4. Check options expected move vs historical
5. Research management guidance trends
6. Find any pre-earnings analyst notes

Provide:
- Earnings date and time
- Consensus estimates
- Historical beat/miss rate
- Expected move from options
- Suggested earnings strategy (if any)
- Key metrics to watch"""

        return await self.run_prompt(prompt, task_description=f"Earnings analysis: {symbol}")

    async def run_prompt(
        self,
        prompt: str,
        task_description: str = "Claude analysis",
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> ClaudeResponse:
        """
        Run a prompt through Claude CLI.

        Args:
            prompt: The prompt to send to Claude
            task_description: Description for logging
            model: Override default model
            max_turns: Override default max turns
        """
        model = model or self.model
        max_turns = max_turns or self.max_turns

        try:
            # Create temp file for prompt (handles complex prompts better)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            # Build command
            cmd = [
                "claude",
                "--print",  # Print output directly
                "--model", model,
                "--max-turns", str(max_turns),
                "--dangerously-skip-permissions",  # For automated use
            ]

            logger.info(f"Running Claude CLI: {task_description}")

            # Run Claude CLI
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=300  # 5 minute timeout
            )

            # Clean up temp file
            os.unlink(prompt_file)

            if process.returncode == 0:
                return ClaudeResponse(
                    content=stdout.decode().strip(),
                    success=True,
                )
            else:
                return ClaudeResponse(
                    content="",
                    success=False,
                    error=stderr.decode().strip() or "Unknown error",
                )

        except asyncio.TimeoutError:
            return ClaudeResponse(
                content="",
                success=False,
                error="Claude CLI timed out after 5 minutes",
            )
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return ClaudeResponse(
                content="",
                success=False,
                error=str(e),
            )

    async def stream_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response from Claude CLI.

        Yields chunks of text as they're generated.
        """
        model = model or self.model

        cmd = [
            "claude",
            "--print",
            "--model", model,
            "--dangerously-skip-permissions",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
        )

        # Send prompt
        process.stdin.write(prompt.encode())
        await process.stdin.drain()
        process.stdin.close()

        # Stream output
        while True:
            chunk = await process.stdout.read(100)
            if not chunk:
                break
            yield chunk.decode()

        await process.wait()


class ClaudeCLIPool:
    """
    Pool of Claude CLI instances for concurrent operations.

    Manages multiple CLI sessions for parallel analysis.
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._instances: list[ClaudeCLI] = []

    async def run_analysis(
        self,
        prompt: str,
        model: str = "sonnet",
        priority: int = 5,
    ) -> ClaudeResponse:
        """Run analysis with concurrency control"""
        async with self.semaphore:
            cli = ClaudeCLI(model=model)
            return await cli.run_prompt(prompt)

    async def batch_analyze(
        self,
        symbols: list[str],
        analysis_type: str = "comprehensive",
    ) -> dict[str, ClaudeResponse]:
        """Analyze multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            cli = ClaudeCLI()
            task = cli.analyze_stock(symbol, analysis_type)
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            async with self.semaphore:
                results[symbol] = await task

        return results

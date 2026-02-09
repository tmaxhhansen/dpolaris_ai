"""
AI Memory System for dPolaris

Handles persistent learning, pattern recognition, and context management
across sessions and machine migrations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import logging

from .database import Database

logger = logging.getLogger("dpolaris.memory")


@dataclass
class MemoryEntry:
    """A single memory entry"""
    category: str
    content: str
    importance: float = 0.5
    created_at: Optional[str] = None
    access_count: int = 0


class DPolarisMemory:
    """
    Persistent memory system that maintains AI continuity.

    Categories:
    - trading_style: User's trading preferences and style
    - risk_tolerance: Observed risk comfort levels
    - mistake_pattern: Patterns that led to losses (to avoid)
    - success_pattern: Patterns that led to wins (to repeat)
    - market_insight: General market learnings
    - symbol_knowledge: Learned information about specific symbols
    - strategy_performance: How different strategies perform for this user
    """

    CATEGORIES = [
        "trading_style",
        "risk_tolerance",
        "mistake_pattern",
        "success_pattern",
        "market_insight",
        "symbol_knowledge",
        "strategy_performance",
    ]

    def __init__(self, db: Database, data_dir: Optional[Path] = None):
        self.db = db
        self.data_dir = data_dir or Path("~/dpolaris_data/memory").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Learning Operations ====================

    def learn(
        self,
        category: str,
        content: str,
        importance: float = 0.5,
        embedding: Optional[list[float]] = None,
    ) -> int:
        """
        Store a learning/insight.

        Args:
            category: Type of memory (see CATEGORIES)
            content: The actual learning/insight
            importance: 0.0 to 1.0, how important this memory is
            embedding: Optional vector embedding for semantic search
        """
        if category not in self.CATEGORIES:
            logger.warning(f"Unknown memory category: {category}")

        # Check for duplicate/similar content
        existing = self.db.get_memories(category=category, limit=100)
        for mem in existing:
            if self._is_similar(content, mem["content"]):
                # Update importance if new learning reinforces existing
                new_importance = min(1.0, mem["importance"] + 0.1)
                logger.debug(f"Reinforcing existing memory: {mem['id']}")
                return mem["id"]

        memory_id = self.db.save_memory(
            category=category,
            content=content,
            importance=importance,
            embedding=embedding,
        )
        logger.info(f"New memory saved: [{category}] {content[:50]}...")
        return memory_id

    def _is_similar(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """Check if two memories are similar (simple word overlap for now)"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold

    def learn_from_trade(self, trade: dict):
        """
        Extract learnings from a completed trade.

        Automatically analyzes trade outcome and stores relevant insights.
        """
        if trade.get("pnl") is None:
            return  # Trade not closed yet

        symbol = trade["symbol"]
        strategy = trade.get("strategy", "unknown")
        pnl = trade["pnl"]
        pnl_percent = trade.get("pnl_percent", 0)
        thesis = trade.get("thesis", "")
        lessons = trade.get("lessons", "")

        # Learn from winners
        if pnl > 0:
            self.learn(
                category="success_pattern",
                content=f"{strategy} on {symbol} was profitable ({pnl_percent:.1f}%). "
                        f"Thesis: {thesis}. Lessons: {lessons}",
                importance=min(0.8, 0.5 + abs(pnl_percent) / 100),
            )

            self.learn(
                category="strategy_performance",
                content=f"{strategy} strategy: WIN on {symbol} ({pnl_percent:.1f}%)",
                importance=0.6,
            )

        # Learn from losers
        else:
            self.learn(
                category="mistake_pattern",
                content=f"{strategy} on {symbol} lost ({pnl_percent:.1f}%). "
                        f"Thesis: {thesis}. What went wrong: {lessons}",
                importance=min(0.9, 0.5 + abs(pnl_percent) / 50),  # Losses weighted higher
            )

            self.learn(
                category="strategy_performance",
                content=f"{strategy} strategy: LOSS on {symbol} ({pnl_percent:.1f}%)",
                importance=0.7,
            )

    def learn_preference(self, preference_type: str, value: str):
        """Learn a user preference"""
        self.learn(
            category="trading_style",
            content=f"User preference - {preference_type}: {value}",
            importance=0.7,
        )

    def learn_risk_observation(self, observation: str):
        """Learn about user's risk tolerance"""
        self.learn(
            category="risk_tolerance",
            content=observation,
            importance=0.8,
        )

    # ==================== Recall Operations ====================

    def recall(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        min_importance: float = 0.3,
    ) -> list[dict]:
        """
        Retrieve memories.

        Args:
            category: Optional category filter
            limit: Max memories to return
            min_importance: Minimum importance threshold
        """
        return self.db.get_memories(
            category=category,
            limit=limit,
            min_importance=min_importance,
        )

    def recall_about_symbol(self, symbol: str, limit: int = 10) -> list[dict]:
        """Get all memories related to a specific symbol"""
        all_memories = self.db.get_memories(limit=500, min_importance=0)
        symbol_memories = [
            m for m in all_memories
            if symbol.upper() in m["content"].upper()
        ]
        return sorted(symbol_memories, key=lambda x: x["importance"], reverse=True)[:limit]

    def recall_strategy_performance(self, strategy: str) -> dict:
        """Get performance summary for a strategy"""
        memories = self.recall(category="strategy_performance", limit=100, min_importance=0)

        wins = 0
        losses = 0

        for mem in memories:
            if strategy.lower() in mem["content"].lower():
                if "WIN" in mem["content"]:
                    wins += 1
                elif "LOSS" in mem["content"]:
                    losses += 1

        total = wins + losses
        return {
            "strategy": strategy,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total * 100) if total > 0 else 0,
        }

    # ==================== Context Building ====================

    def build_context(self, max_tokens: int = 2000) -> str:
        """
        Build a context string from memories to inject into AI prompts.

        This gives Claude continuity and personalization.
        """
        context_parts = []

        # High-importance memories first
        high_priority = self.recall(min_importance=0.7, limit=10)
        if high_priority:
            context_parts.append("## Critical Learnings")
            for mem in high_priority:
                context_parts.append(f"- [{mem['category']}] {mem['content']}")

        # Trading style
        style_memories = self.recall(category="trading_style", limit=5)
        if style_memories:
            context_parts.append("\n## User's Trading Style")
            for mem in style_memories:
                context_parts.append(f"- {mem['content']}")

        # Risk tolerance
        risk_memories = self.recall(category="risk_tolerance", limit=3)
        if risk_memories:
            context_parts.append("\n## Risk Tolerance Observations")
            for mem in risk_memories:
                context_parts.append(f"- {mem['content']}")

        # Recent mistake patterns (to avoid)
        mistakes = self.recall(category="mistake_pattern", limit=5)
        if mistakes:
            context_parts.append("\n## Patterns to Avoid (Recent Mistakes)")
            for mem in mistakes:
                context_parts.append(f"- {mem['content']}")

        # Success patterns (to repeat)
        successes = self.recall(category="success_pattern", limit=5)
        if successes:
            context_parts.append("\n## Patterns That Work (Recent Successes)")
            for mem in successes:
                context_parts.append(f"- {mem['content']}")

        context = "\n".join(context_parts)

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[truncated]"

        return context

    def build_symbol_context(self, symbol: str) -> str:
        """Build context specific to a symbol"""
        memories = self.recall_about_symbol(symbol, limit=10)

        if not memories:
            return f"No previous experience with {symbol}."

        context_parts = [f"## Previous Experience with {symbol}"]
        for mem in memories:
            context_parts.append(f"- [{mem['category']}] {mem['content']}")

        return "\n".join(context_parts)

    # ==================== Analysis ====================

    def analyze_patterns(self) -> dict:
        """Analyze trading patterns from memory"""
        all_memories = self.recall(limit=500, min_importance=0)

        analysis = {
            "total_memories": len(all_memories),
            "by_category": {},
            "top_insights": [],
            "risk_profile": "",
            "style_summary": "",
        }

        # Count by category
        for category in self.CATEGORIES:
            cat_memories = [m for m in all_memories if m["category"] == category]
            analysis["by_category"][category] = len(cat_memories)

        # Top insights (highest importance)
        top = sorted(all_memories, key=lambda x: x["importance"], reverse=True)[:5]
        analysis["top_insights"] = [m["content"] for m in top]

        # Summarize risk profile
        risk_memories = [m for m in all_memories if m["category"] == "risk_tolerance"]
        if risk_memories:
            analysis["risk_profile"] = "; ".join([m["content"] for m in risk_memories[:3]])

        # Summarize style
        style_memories = [m for m in all_memories if m["category"] == "trading_style"]
        if style_memories:
            analysis["style_summary"] = "; ".join([m["content"] for m in style_memories[:3]])

        return analysis

    def get_strategy_rankings(self) -> list[dict]:
        """Rank strategies by historical performance"""
        strategies = {}
        memories = self.recall(category="strategy_performance", limit=200, min_importance=0)

        for mem in memories:
            content = mem["content"]
            # Parse strategy name from content
            if "strategy:" in content.lower():
                parts = content.split("strategy:")
                if len(parts) > 1:
                    strategy_part = parts[0].strip()
                    if "WIN" in content:
                        strategies.setdefault(strategy_part, {"wins": 0, "losses": 0})
                        strategies[strategy_part]["wins"] += 1
                    elif "LOSS" in content:
                        strategies.setdefault(strategy_part, {"wins": 0, "losses": 0})
                        strategies[strategy_part]["losses"] += 1

        # Calculate win rates and rank
        rankings = []
        for strategy, stats in strategies.items():
            total = stats["wins"] + stats["losses"]
            if total >= 3:  # Minimum sample size
                win_rate = stats["wins"] / total * 100
                rankings.append({
                    "strategy": strategy,
                    "total_trades": total,
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "win_rate": win_rate,
                })

        return sorted(rankings, key=lambda x: x["win_rate"], reverse=True)

    # ==================== Export/Import ====================

    def export(self, export_path: Optional[Path] = None) -> Path:
        """Export all memories for migration"""
        if export_path is None:
            export_path = self.data_dir / f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        all_memories = self.recall(limit=10000, min_importance=0)

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
            "memories": all_memories,
            "analysis": self.analyze_patterns(),
        }

        export_path.write_text(json.dumps(export_data, indent=2, default=str))
        logger.info(f"Memories exported to {export_path}")
        return export_path

    def import_memories(self, import_path: Path) -> int:
        """Import memories from export file"""
        data = json.loads(import_path.read_text())

        imported = 0
        for mem in data.get("memories", []):
            try:
                self.db.save_memory(
                    category=mem["category"],
                    content=mem["content"],
                    importance=mem.get("importance", 0.5),
                )
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")

        logger.info(f"Imported {imported} memories from {import_path}")
        return imported

    # ==================== Maintenance ====================

    def decay_importance(self, decay_factor: float = 0.95):
        """
        Decay importance of old memories.

        Memories that aren't accessed become less important over time.
        """
        # This would need direct SQL access - implement if needed
        pass

    def prune_low_importance(self, threshold: float = 0.1):
        """Remove memories below importance threshold"""
        # This would need direct SQL access - implement if needed
        pass

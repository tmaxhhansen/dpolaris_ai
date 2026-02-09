"""
dPolaris AI - Trading Intelligence System

A local AI trading assistant that combines:
- Claude API for reasoning and analysis
- Claude CLI for deep research with web access
- Local ML models for predictions
- Persistent memory for learning
- Broker integrations for real-time data
"""

__version__ = "0.1.0"
__author__ = "dPolaris"

from core.config import Config, get_config
from core.database import Database
from core.memory import DPolarisMemory
from core.ai import DPolarisAI

__all__ = [
    "Config",
    "get_config",
    "Database",
    "DPolarisMemory",
    "DPolarisAI",
]

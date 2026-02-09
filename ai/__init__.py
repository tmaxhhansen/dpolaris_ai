"""
AI Module for dPolaris

Includes:
- Scheduler for automated tasks (training, news scanning, predictions)
"""

from .scheduler import (
    DPolarisScheduler,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)

__all__ = [
    "DPolarisScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
]

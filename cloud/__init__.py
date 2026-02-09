"""
Cloud Sync Module for dPolaris

Provides Supabase integration for syncing predictions and data between devices.
"""

from .supabase_sync import (
    SupabaseSync,
    SyncConfig,
)

__all__ = [
    "SupabaseSync",
    "SyncConfig",
]

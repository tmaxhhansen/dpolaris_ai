"""dPolaris daemon package exports."""

from .orchestrator import OrchestratorConfig, OrchestratorDaemon, get_orchestrator_singleton

try:  # Optional dependency path; scheduler requires apscheduler.
    from .scheduler import DPolarisDaemon, run_daemon
except Exception:  # pragma: no cover
    DPolarisDaemon = None
    run_daemon = None

__all__ = [
    "DPolarisDaemon",
    "run_daemon",
    "OrchestratorDaemon",
    "OrchestratorConfig",
    "get_orchestrator_singleton",
]

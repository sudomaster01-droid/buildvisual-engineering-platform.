"""Execution context handed to every pipeline step.

Steps receive immutable-ish references (ids), a database session, storage and a
heartbeat callback. They never reach into global state for anything they could
be given explicitly, which keeps them testable and restart-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Job
from ..storage import Storage


@dataclass
class StepContext:
    job: Job
    db: Session
    settings: Settings
    storage: Storage
    worker_id: str
    heartbeat: Callable[[], None]
    log: logging.LoggerAdapter
    step_name: str = ""
    #: scratch space shared across steps within one job run (not persisted)
    cache: dict[str, Any] | None = None

    def note(self, message: str, *, level: str = "info", data: dict | None = None) -> None:
        """Record a user-visible event on the job."""
        from ..services.jobs import record_event

        record_event(
            self.db, self.job, stage=self.job.stage, message=message,
            level=level, data=data or {},
        )
        # Commit immediately: the engineer's page polls this table, and a long
        # step must not hide its progress behind an open transaction.
        self.db.commit()

    @property
    def overrides(self) -> dict[str, Any]:
        """Rectification/attempt overrides persisted on the job."""
        return (self.job.diagnostics or {}).get("overrides", {})

    @property
    def rectification(self) -> dict[str, Any]:
        return (self.job.diagnostics or {}).get("rectification", {})

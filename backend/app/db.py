"""Database engine, session factory and schema migrations.

Portable SQLAlchemy 2.0 metadata so the same models run on SQLite (zero-setup
local default) and PostgreSQL (compose / production). Migrations are ordered,
idempotent and recorded in ``schema_migrations``; this keeps deployment
reproducible without pulling Alembic's full machinery into a single-engineer
product. Adding Alembic later is a drop-in because the metadata is canonical.

All timestamps are stored as naive UTC. ``utcnow()`` is the only clock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    """Naive UTC 'now' — the single clock used by the whole application."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
#: Re-entrant: get_session_factory() holds it while calling get_engine().
_lock = threading.RLock()


def _make_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    url = settings.database_url
    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # Make the path relative to the data dir when it is the default.
        if url in ("sqlite:///./data/buildvisual.db",):
            url = f"sqlite:///{settings.data_dir / 'buildvisual.db'}"
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _enable_sqlite_pragmas(engine)
    return engine


def _enable_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        with _lock:
            if _SessionLocal is None:
                _SessionLocal = sessionmaker(
                    bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
                )
    return _SessionLocal


def reset_engine() -> None:
    """Test helper: drop cached engine/session factory."""
    global _engine, _SessionLocal
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _migration_0001_initial(session: Session) -> None:
    from . import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(bind=session.get_bind())


def _migration_0002_indexes(session: Session) -> None:
    from sqlalchemy import text

    bind = session.get_bind()
    stmts = [
        "CREATE INDEX IF NOT EXISTS ix_jobs_project_status ON jobs (project_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs (created_at)",
        "CREATE INDEX IF NOT EXISTS ix_steps_state_lease ON pipeline_steps (state, lease_expires_at)",
        "CREATE INDEX IF NOT EXISTS ix_clips_job_shot ON generated_clips (job_id, shot_id)",
        "CREATE INDEX IF NOT EXISTS ix_events_job_created ON job_events (job_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_usage_created ON usage_ledger (created_at)",
    ]
    for stmt in stmts:
        try:
            bind.execute(text(stmt))
        except Exception:  # pragma: no cover - backend-specific index syntax
            log.debug("index migration skipped: %s", stmt)


def _migration_0003_login_attempts(session: Session) -> None:
    """Login throttling table (added with the single-engineer login rework)."""
    from . import models  # noqa: F401

    bind = session.get_bind()
    models.LoginAttempt.__table__.create(bind=bind, checkfirst=True)
    try:
        bind.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_login_attempts_lookup "
            "ON login_attempts (email, ip_hint, created_at)"
        ))
    except Exception:  # pragma: no cover - index syntax varies by backend
        log.debug("login_attempts index skipped")


def _migration_0004_password_reset_tokens(session: Session) -> None:
    from . import models  # noqa: F401

    models.PasswordResetToken.__table__.create(
        bind=session.get_bind(), checkfirst=True
    )


#: Ordered, append-only. Never edit a shipped migration.
MIGRATIONS: list[tuple[str, callable]] = [
    ("0001_initial", _migration_0001_initial),
    ("0002_indexes", _migration_0002_indexes),
    ("0003_login_attempts", _migration_0003_login_attempts),
    ("0004_password_reset_tokens", _migration_0004_password_reset_tokens),
]


def migrate() -> list[str]:
    """Apply pending migrations. Safe to call concurrently and repeatedly."""
    from sqlalchemy import inspect, text

    engine = get_engine()
    applied: list[str] = []
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version VARCHAR(64) PRIMARY KEY,"
                " applied_at TIMESTAMP NOT NULL)"
            )
        )
    existing = {
        row[0]
        for row in engine.connect().execute(text("SELECT version FROM schema_migrations"))
    }
    for version, fn in MIGRATIONS:
        if version in existing:
            continue
        log.info("applying migration %s", version)
        with session_scope() as session:
            fn(session)
            session.execute(
                text(
                    "INSERT INTO schema_migrations (version, applied_at) "
                    "VALUES (:v, :t)"
                ),
                {"v": version, "t": utcnow()},
            )
        applied.append(version)
    # Sanity check: every mapped table exists.
    insp = inspect(engine)
    missing = {
        t for t in Base.metadata.tables if t not in insp.get_table_names()
    }
    if missing:  # pragma: no cover - indicates a broken migration chain
        raise RuntimeError(f"schema incomplete, missing tables: {sorted(missing)}")
    return applied

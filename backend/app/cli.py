"""Operator CLI: ``python -m app.cli <command>``.

Deliberately small — the product is one page plus one worker, so the operational
surface is: migrate, create the engineer account, run a worker, inspect a job,
sweep stalled work, and print capability/cost facts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import describe_settings, get_settings
from .db import migrate, session_scope
from .logging_setup import configure_logging

log = logging.getLogger("buildvisual.cli")


def cmd_migrate(_args) -> int:
    applied = migrate()
    print(json.dumps({"applied": applied}, indent=2))
    return 0


def cmd_create_user(args) -> int:
    from .models import User
    from .security import ensure_engineer_account, hash_password

    migrate()
    with session_scope() as db:
        if args.email and args.password:
            existing = db.query(User).filter(User.email == args.email.lower()).first()
            if existing:
                existing.password_hash = hash_password(args.password)
                print(f"updated {existing.email}")
            else:
                db.add(User(
                    email=args.email.lower(),
                    password_hash=hash_password(args.password),
                    display_name=args.name or "Engineer",
                ))
                print(f"created {args.email.lower()}")
        else:
            user = ensure_engineer_account(db)
            print(f"engineer account: {user.email if user else 'not configured'}")
    return 0


def cmd_set_password(args) -> int:
    """Reset the single engineer password.

    The password is read interactively unless ``--password`` is given, so it does
    not end up in shell history or process listings.
    """
    import getpass

    from .models import User
    from .security import hash_password, normalise_email

    migrate()
    email = normalise_email(args.email or get_settings().engineer_email)
    password = args.password or getpass.getpass("New password (min 10 characters): ")
    if len(password) < 10:
        print("password must be at least 10 characters", file=sys.stderr)
        return 2
    if not args.password:
        confirm = getpass.getpass("Repeat new password: ")
        if confirm != password:
            print("passwords did not match", file=sys.stderr)
            return 2
    with session_scope() as db:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            user = User(email=email, password_hash=hash_password(password),
                        display_name=get_settings().engineer_name or "Engineer")
            db.add(user)
            print(f"created engineer account {email}")
        else:
            user.password_hash = hash_password(password)
            user.is_active = True
            print(f"updated password for {email}")
        db.commit()
    print("Also keep ENGINEER_PASSWORD in the deployment configuration in step with "
          "this password, or the next startup will overwrite it (ENGINEER_PASSWORD_SYNC).")
    return 0


def cmd_worker(_args) -> int:
    from .pipeline.runner import main as worker_main

    worker_main()
    return 0


def cmd_sweep(_args) -> int:
    from .pipeline.runner import sweep_stalled

    migrate()
    with session_scope() as db:
        print(json.dumps(sweep_stalled(db), indent=2))
    return 0


def cmd_job(args) -> int:
    from .models import Job

    with session_scope() as db:
        job = db.get(Job, args.id)
        if job is None:
            print("no such job", file=sys.stderr)
            return 1
        print(json.dumps({
            "id": job.id,
            "kind": job.kind,
            "stage": job.stage,
            "status": job.status,
            "error": job.error_code,
            "message": job.user_message,
            "spent_cents": job.spent_cents,
            "video_version_id": job.video_version_id,
            "diagnostics": {
                key: value for key, value in (job.diagnostics or {}).items()
                if key in ("interpretation", "generation", "assembly", "quality", "rectification")
            },
        }, indent=2, default=str))
    return 0


def cmd_capabilities(_args) -> int:
    from .services.video.factory import capability_report

    print(json.dumps({
        "settings": describe_settings(),
        "providers": capability_report(),
    }, indent=2, default=str))
    return 0


def cmd_run(args) -> int:
    """Create a job from text and run it inline (used for verification)."""
    from .models import User
    from .pipeline.runner import run_job_to_completion
    from .services import orchestrator

    migrate()
    with session_scope() as db:
        user = db.query(User).first()
        if user is None:
            print("create the engineer account first: python -m app.cli create-user", file=sys.stderr)
            return 1
        job = orchestrator.create_generation_job(
            db, user_id=user.id, project_id=None, project_name=args.name,
            text=args.text, drawing=None, audio=None,
        )
        job_id = job.id
        print(f"job {job_id}")
    job = run_job_to_completion(job_id, worker_id="cli")
    print(json.dumps({"stage": job.stage, "status": job.status, "error": job.error_code,
                      "video_version_id": job.video_version_id}, indent=2))
    return 0 if job.status == "ready" else 1


def cmd_cost(args) -> int:
    from datetime import timedelta

    from sqlalchemy import select

    from .db import utcnow
    from .models import UsageLedger

    since = utcnow() - timedelta(days=args.days)
    with session_scope() as db:
        rows = db.execute(
            select(UsageLedger).where(UsageLedger.created_at >= since)
        ).scalars().all()
    by_operation: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = by_operation.setdefault(row.provider, {"cost_cents": 0, "rows": 0})
        bucket["cost_cents"] += row.cost_cents
        bucket["rows"] += 1
    print(json.dumps({
        "days": args.days,
        "total_cents": sum(b["cost_cents"] for b in by_operation.values()),
        "by_provider": by_operation,
        "note": "Costs are configured-rate estimates, not provider billing exports.",
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="buildvisual", description="BuildVisual operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=cmd_migrate)

    user = sub.add_parser("create-user", help="create or update the engineer account")
    user.add_argument("--email")
    user.add_argument("--password", help="omit to be prompted (keeps it out of shell history)")
    user.add_argument("--name")
    user.set_defaults(func=cmd_create_user)

    setpw = sub.add_parser("set-password", help="set a new password for the engineer account")
    setpw.add_argument("--email", help="defaults to ENGINEER_EMAIL")
    setpw.add_argument("--password", help="omit to be prompted")
    setpw.set_defaults(func=cmd_set_password)

    sub.add_parser("worker", help="run the background pipeline worker").set_defaults(func=cmd_worker)
    sub.add_parser("sweep", help="requeue stalled steps and fail dead jobs").set_defaults(func=cmd_sweep)
    sub.add_parser("capabilities", help="print provider capability report").set_defaults(func=cmd_capabilities)

    job = sub.add_parser("job", help="print one job's diagnostics")
    job.add_argument("id")
    job.set_defaults(func=cmd_job)

    run = sub.add_parser("run", help="create and run a job from text, inline")
    run.add_argument("--text", required=True)
    run.add_argument("--name", default=None)
    run.set_defaults(func=cmd_run)

    cost = sub.add_parser("cost", help="summarise the usage ledger")
    cost.add_argument("--days", type=int, default=30)
    cost.set_defaults(func=cmd_cost)

    args = parser.parse_args(argv)
    settings = get_settings()
    log.debug("settings: %s", describe_settings().get("video_provider"))
    del settings
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

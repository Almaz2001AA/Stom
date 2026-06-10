"""Admin CLI: fail jobs stuck in `running` past the timeout (run from cron).

A worker that is OOM/SIGKILLed mid-inference never runs its in-process failure
path, leaving its Job row `running` forever. This sweeps those rows to `failed`.
"""

from __future__ import annotations

import argparse

from stomserver.config import load_config
from stomserver.db.session import make_engine, make_session_factory
from stomserver.segmentation.worker import reap_stale_jobs


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Fail stale `running` jobs.")
    parser.add_argument(
        "--timeout-seconds", type=float, default=cfg.job_timeout_seconds,
        help=f"staleness threshold (default {cfg.job_timeout_seconds})",
    )
    args = parser.parse_args(argv)

    engine = make_engine(cfg.db_url)
    factory = make_session_factory(engine)
    reaped = reap_stale_jobs(factory, timeout_seconds=args.timeout_seconds)
    print(f"reaped {reaped} stale running job(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

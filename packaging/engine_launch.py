"""PyInstaller entry point for the engine-pack (the on-device inference CLI)."""

import multiprocessing

from stomengine.cli import main

if __name__ == "__main__":
    # MUST be the first thing the frozen binary does. nnU-Net's predictor spawns
    # multiprocessing.Pool workers; in a PyInstaller one-folder build each spawned
    # worker re-execs this very exe. Without freeze_support() the worker re-runs
    # main() -> predict -> spawns more workers, an unbounded fork bomb that hangs
    # inference forever (observed: a 32^3 smoke volume ran the full 6h CI timeout).
    multiprocessing.freeze_support()
    raise SystemExit(main())

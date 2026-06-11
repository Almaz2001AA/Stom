"""PyInstaller entry point for the engine-pack (the on-device inference CLI)."""

from stomengine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

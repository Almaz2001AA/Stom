"""PyInstaller entry script: launches the Stom desktop client."""

import sys

from stomclient.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

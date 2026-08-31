"""Format all Python source code using Ruff."""

import os
import subprocess
import sys


def main():
    """Run Ruff formatter on all Python files in the repository."""
    # Get the directory of this script (fuellib/cli)
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the fuellib package directory (one level up from cli)
    fuellib_dir = os.path.dirname(cli_dir)

    # Get the project root (one level up from fuellib package)
    project_root = os.path.dirname(fuellib_dir)

    try:
        # Call Ruff directly with the project root
        # Ruff will recursively find and format all .py files
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "format", project_root],
            check=True,
        )
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except OSError as e:
        print(f"Error running ruff formatter: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

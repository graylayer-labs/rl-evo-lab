from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_readme_generated_sections_are_in_sync() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/sync_readme.py", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

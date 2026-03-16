from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def launch_python_module(module: str, cwd: Path, args: list[str] | None = None) -> int:
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    process = subprocess.run(cmd, cwd=str(cwd), env=os.environ.copy(), check=False)
    return process.returncode

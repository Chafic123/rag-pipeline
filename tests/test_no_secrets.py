"""Fail if any file git would commit contains an API key.

A key that reaches git history has to be treated as leaked, so this check runs
in `make test` locally and in CI on every push. Real keys belong in .env only.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KEY_PATTERN = re.compile(r"sk-(proj-)?[A-Za-z0-9_-]{20,}")


def _files_git_would_commit() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.splitlines() if line]


def test_no_api_keys_in_files_git_would_commit():
    leaks = []
    for path in _files_git_would_commit():
        if not path.is_file():
            continue                      # tracked but deleted in the working tree
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue                      # binary file (images)
        if KEY_PATTERN.search(text):
            leaks.append(str(path.relative_to(ROOT)))
    assert not leaks, f"API key found in {leaks}. Move it to .env (git-ignored) and rotate it."

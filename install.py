"""
install.py
----------
A resumable, staged installer for this project's dependencies.

Why this exists: `pip install -r requirements.txt` treats all packages as
one atomic unit conceptually, and `--force-reinstall` (which fixes real
version-pin problems) re-downloads and rebuilds EVERY package every time,
including large ones (torch, transformers) that were already working
fine. On a slow/corporate network, one fragile package (e.g. spacy
needing a source build) means re-paying the cost of everything else too.

This script instead:
  1. Splits dependencies into logical stages (core -> ML -> NLP -> topic
     modeling), ordered lightest/most-reliable first.
  2. Tracks completion per stage in a local state file (.install_state.json).
  3. Skips any stage already marked done AND still verified installed --
     never re-downloads/rebuilds something that already succeeded.
  4. On a stage failure, records it and CONTINUES to the remaining stages
     rather than aborting the whole run -- so one fragile package doesn't
     block everything else.
  5. Prints a clear summary at the end: what's installed, what failed and
     why, and exactly what command to re-run for just the failed part.

Usage:
    python install.py            # run/resume the staged install
    python install.py --status   # show current state without installing
    python install.py --retry spacy   # force-retry one specific stage
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
from importlib.metadata import version as installed_version, PackageNotFoundError

STATE_PATH = Path(__file__).parent / ".install_state.json"

# Ordered stages: lightest/most-reliable first, heaviest/most-fragile last.
# Each stage is independent enough that a later failure shouldn't block
# earlier stages from being usable.
STAGES = [
    ("core", [
        "streamlit>=1.38.0",
        "pandas>=2.0.0",
        "plotly>=5.20.0",
        "yfinance>=0.2.40",
        "feedparser>=6.0.10",
        "schedule>=1.2.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
    ]),
    ("torch", [
        "torch>=2.2.0",
    ]),
    ("transformers", [
        "transformers>=5.0.0,<6.0.0",
    ]),
    ("spacy", [
        "spacy>=3.8.0,<4.0.0",
    ]),
    ("topic_modeling", [
        "sentence-transformers>=3.0.0",
        "bertopic>=0.16.0",
    ]),
    ("news_extras", [
        # Optional: resolves Google News RSS redirect links back to the
        # real publisher URL for full-article summaries. Relies on an
        # undocumented Google internal endpoint that could change or
        # rate-limit without notice -- isolated in its own stage so a
        # failure here never blocks anything else. Dashboard works fine
        # without it, just with headline-only summaries for Google-News-
        # sourced articles instead of full-text ones.
        "googlenewsdecoder",
    ]),
]

SPACY_MODEL = "en_core_web_sm"


def _load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _package_satisfied(spec: str) -> bool:
    """Check if an already-installed package satisfies the given spec, e.g. 'torch>=2.2.0'."""
    req = Requirement(spec)
    try:
        installed = Version(installed_version(req.name))
    except PackageNotFoundError:
        return False
    return req.specifier.contains(installed, prereleases=True)


def _stage_satisfied(packages) -> bool:
    return all(_package_satisfied(p) for p in packages)


def _spacy_model_installed() -> bool:
    try:
        import spacy  # noqa: local import, spacy may not be installed yet
        return spacy.util.is_package(SPACY_MODEL)
    except Exception:
        return False


def run_stage(name, packages, state, force=False):
    # Always check actual environment state first, regardless of recorded
    # history -- the state file is a cache/log, not the source of truth.
    # This matters on a fresh state file too (e.g. first run on a machine
    # that already has some packages installed some other way).
    if not force and _stage_satisfied(packages):
        if state.get(name, {}).get("status") != "done":
            state[name] = {
                "status": "done",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "packages": packages,
                "note": "already satisfied, verified without reinstalling",
            }
            _save_state(state)
        print(f"[skip] {name}: already installed and satisfied.")
        return True

    print(f"[install] {name}: {', '.join(packages)}")
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    if result.returncode == 0:
        state[name] = {"status": "done", "timestamp": timestamp, "packages": packages}
        print(f"[ok] {name} installed successfully.")
        success = True
    else:
        # Keep only the tail of stderr -- pip build failures can be
        # thousands of lines; the last ~40 lines usually contain the
        # actual error, the rest is noise.
        error_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
        state[name] = {
            "status": "failed",
            "timestamp": timestamp,
            "packages": packages,
            "error": error_tail,
        }
        print(f"[FAILED] {name} -- see summary at the end for details.")
        success = False

    _save_state(state)
    return success


def run_spacy_model_step(state, force=False):
    name = "spacy_model"
    if not force and state.get(name, {}).get("status") == "done" and _spacy_model_installed():
        print(f"[skip] {name}: {SPACY_MODEL} already downloaded.")
        return True

    if not _package_satisfied("spacy>=3.8.0"):
        print(f"[skip] {name}: spacy itself isn't installed yet, skipping model download.")
        return False

    print(f"[install] {name}: downloading {SPACY_MODEL} ...")
    cmd = [sys.executable, "-m", "spacy", "download", SPACY_MODEL]
    result = subprocess.run(cmd, capture_output=True, text=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    if result.returncode == 0:
        state[name] = {"status": "done", "timestamp": timestamp}
        print(f"[ok] {SPACY_MODEL} downloaded.")
        success = True
    else:
        error_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
        state[name] = {"status": "failed", "timestamp": timestamp, "error": error_tail}
        print(f"[FAILED] {name} -- see summary at the end for details.")
        success = False

    _save_state(state)
    return success


def print_status(state):
    if not state:
        print("No install state recorded yet -- run `python install.py` to begin.")
        return

    print("Current install state:\n")
    for name, info in state.items():
        status = info.get("status", "unknown")
        ts = info.get("timestamp", "")
        marker = "✓" if status == "done" else "✗"
        print(f"  [{marker}] {name:20s} {status:8s} {ts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show current install state and exit")
    parser.add_argument("--retry", metavar="STAGE", help="Force-retry a specific stage by name")
    args = parser.parse_args()

    state = _load_state()

    if args.status:
        print_status(state)
        return

    results = {}

    for name, packages in STAGES:
        force = args.retry == name
        results[name] = run_stage(name, packages, state, force=force)

    # spaCy's model download is a separate step from the pip package itself
    force_model = args.retry == "spacy_model"
    results["spacy_model"] = run_spacy_model_step(state, force=force_model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(
            f"\n{len(failed)} stage(s) failed: {', '.join(failed)}\n"
            f"Everything else above is installed and usable right now.\n"
            f"To see the actual error for a failed stage:\n"
            f"    python install.py --status\n"
            f"To retry just that stage once you've fixed the issue:\n"
            f"    python install.py --retry {failed[0]}"
        )
    else:
        print("\nAll stages installed successfully.")


if __name__ == "__main__":
    main()

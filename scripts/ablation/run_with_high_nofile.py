#!/usr/bin/env python3
"""Raise this rank's file-descriptor limit, then execute a Python script."""

import os
import resource
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_high_nofile.py TARGET_SCRIPT [ARGS ...]")

    target_script = os.path.abspath(sys.argv[1])
    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(
        f"NOFILE_PREFLIGHT rank={os.environ.get('RANK', 'unset')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'unset')} "
        f"soft={soft_limit} hard={hard_limit}",
        flush=True,
    )

    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(target_script))
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()

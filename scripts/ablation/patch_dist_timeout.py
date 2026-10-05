"""Make the torch.distributed process-group watchdog timeout configurable.

Background
----------
`setup_distributed()` calls `init_process_group(init_method='env://')` with no
`timeout=`, so PyTorch's default of 1800 s applies. On this cluster a rank can
legitimately stall for longer than that inside CPU-only post-processing
(pycocotools `accumulate()` over a large validation split) when the node is
under memory/CPU pressure from other tenants. The sibling rank then hits the
1800 s watchdog and SIGABRTs the whole job.

This patch keeps the default at exactly 1800 s (i.e. it is a no-op unless the
new env var is set) and only adds an opt-in override, so existing experiment
semantics are untouched.

Usage
-----
    /cobot/miniforge3/envs/lw-detr/bin/python scripts/ablation/patch_dist_timeout.py [--revert]
"""

import argparse
import os
import sys

TARGET = "ecdetseg/engine/misc/dist_utils.py"

OLD = "        torch.distributed.init_process_group(init_method='env://')\n"

NEW = (
    "        # Watchdog timeout for collectives. 1800 s is PyTorch's own default;\n"
    "        # EC_DIST_TIMEOUT_SEC raises it for runs whose CPU-side post-processing\n"
    "        # can legitimately outlast the default on a busy/shared node.\n"
    "        _pg_timeout = int(os.getenv('EC_DIST_TIMEOUT_SEC', '1800'))\n"
    "        torch.distributed.init_process_group(\n"
    "            init_method='env://',\n"
    "            timeout=datetime.timedelta(seconds=_pg_timeout),\n"
    "        )\n"
)

OLD_IMPORT = "import atexit\nimport os\n"
NEW_IMPORT = "import atexit\nimport datetime\nimport os\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    root = args.project_root or os.getcwd()
    path = os.path.join(root, TARGET)
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if args.revert:
        if NEW not in src:
            print("revert: patched block not present, nothing to do")
            return 0
        src = src.replace(NEW, OLD, 1)
        src = src.replace(NEW_IMPORT, OLD_IMPORT, 1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print("revert: OK")
        return 0

    if NEW in src:
        print("patch: already applied, nothing to do")
        return 0

    n = src.count(OLD)
    if n != 1:
        print("patch: expected exactly 1 occurrence of the target line, found %d" % n)
        return 1

    src = src.replace(OLD, NEW, 1)
    if "import datetime" not in src:
        if src.count(OLD_IMPORT) != 1:
            print("patch: could not unambiguously insert the datetime import")
            return 1
        src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("patch: OK -> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

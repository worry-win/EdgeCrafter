"""Insert the EC_DIST_TIMEOUT_SEC export into the cmp5L sbatch header block.

Paired with scripts/ablation/patch_dist_timeout.py. That patch makes the
process-group watchdog timeout env-overridable (default stays 1800 s); this one
opts the cmp5L campaign into a 3600 s window, because a rank can spend >30 min
inside CPU-only pycocotools accumulation when the node is memory-starved, and
the sibling rank then SIGABRTs the job on the 1800 s default.

Idempotent: re-running is a no-op.
"""

import argparse
import os
import sys

ANCHOR = 'export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"\n'

BLOCK = (
    "\n"
    "# A rank can legitimately spend >30 min in CPU-only post-processing\n"
    "# (pycocotools accumulate over the full valid split) when the node is under\n"
    "# memory pressure from other tenants; in job 2259_4 that killed the run on\n"
    "# the 1800 s default watchdog. Widen the window to 1 h.\n"
    'export EC_DIST_TIMEOUT_SEC="${EC_DIST_TIMEOUT_SEC:-3600}"\n'
)

MARK = "EC_DIST_TIMEOUT_SEC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sbatch_path")
    args = ap.parse_args()

    if not os.path.isfile(args.sbatch_path):
        print("no such file: %s" % args.sbatch_path)
        return 1

    with open(args.sbatch_path, encoding="utf-8") as fh:
        src = fh.read()

    if MARK in src:
        print("already patched, nothing to do")
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print("expected exactly 1 anchor, found %d" % n)
        return 1

    src = src.replace(ANCHOR, ANCHOR + BLOCK, 1)
    with open(args.sbatch_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("patched %s" % args.sbatch_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

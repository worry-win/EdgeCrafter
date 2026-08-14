"""Run EdgeCrafter training with Torch Elastic child-error recording enabled."""

import argparse
from pathlib import Path
import sys

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record


ECDETSEG_ROOT = Path(__file__).resolve().parents[2] / "ecdetseg"
sys.path.insert(0, str(ECDETSEG_ROOT))

import train  # noqa: E402


@record
def recorded_main() -> None:
    mp.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="")
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-t", "--tuning", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    train.main(parser.parse_args())


if __name__ == "__main__":
    recorded_main()

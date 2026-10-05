"""Read-only initialization observer around the byte-preserved historical trainer."""
import json
import os
import sys
from pathlib import Path


def install_initialization_observer(trainer, audit):
    original = trainer.ECFullSolver.train
    def observed_train(solver):
        result = original(solver)
        audit(solver, solver.args, trainer._module)
        return result
    trainer.ECFullSolver.train = observed_train


def main():
    code = Path(os.environ['EC_HISTORICAL_CODE']).resolve()
    sys.path[:0] = [str(code / 'ecdetseg'), str(code)]
    from scripts.ablation import train_cmp5L_ec_v as trainer
    from scripts.ablation.cmp5L_repro_audit import audit_loaded_initialization
    install_initialization_observer(trainer, audit_loaded_initialization)
    trainer.main(trainer.parse_args())


if __name__ == '__main__':
    main()

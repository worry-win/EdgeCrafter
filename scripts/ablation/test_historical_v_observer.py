import unittest
from types import SimpleNamespace
import torch
from scripts.ablation.run_historical_v_audited import install_initialization_observer

class ObserverTests(unittest.TestCase):
    def test_observer_preserves_rng_and_original_train_result(self):
        events = []
        class Solver:
            def train(self):
                events.append('loaded')
                self.args = SimpleNamespace(smoke=False)
                return 'result'
        trainer = SimpleNamespace(ECFullSolver=Solver, _module=lambda x: x)
        def audit(solver, args, unwrap):
            events.append('audited')
        install_initialization_observer(trainer, audit)
        before = torch.get_rng_state().clone()
        self.assertEqual(Solver().train(), 'result')
        self.assertEqual(events, ['loaded', 'audited'])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

if __name__ == '__main__':
    unittest.main()

import unittest

from ecdetseg.engine.solver.ec_engine import _BatchSlice, _optimizer_step_due


class GradientAccumulationTest(unittest.TestCase):
    def test_steps_every_accumulation_window_and_flushes_tail(self):
        due = [_optimizer_step_due(i, 5, 2) for i in range(5)]
        self.assertEqual(due, [False, True, False, True, True])

    def test_batch_slice_skips_completed_steps_without_changing_total_epoch_size(self):
        batches = _BatchSlice(list(range(8)), start_step=3)

        self.assertEqual(list(batches), [3, 4, 5, 6, 7])
        self.assertEqual(len(batches), 5)


if __name__ == '__main__':
    unittest.main()

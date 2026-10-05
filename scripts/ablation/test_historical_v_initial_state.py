import unittest
import torch
from scripts.ablation.cmp5L_repro_audit import validate_initial_state
class InitialStateTests(unittest.TestCase):
    def test_official_masked_inf_anchors_allowed_but_nan_weights_rejected(self):
        model={'decoder.anchors':torch.tensor([[[0.,0.,0.,0.],[float('inf')]*4]]),
               'decoder.valid_mask':torch.tensor([[[True],[False]]]), 'decoder.weight':torch.ones(2)}
        state={'model':model,'ema':{'module':{k:v.clone() for k,v in model.items()}}}
        result=validate_initial_state(state)
        self.assertTrue(result['model_equals_ema'])
        self.assertEqual(result['masked_inf_anchor_count'],1)
        state['model']['decoder.weight'][0]=float('nan')
        with self.assertRaises(RuntimeError):validate_initial_state(state)
if __name__=='__main__':unittest.main()

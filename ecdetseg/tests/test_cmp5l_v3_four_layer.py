import unittest
import torch
from scripts.ablation.cmp5L_ec_v3_four_layer import four_layer_candidate_set_kd
from scripts.ablation.cmp5L_ec_v_losses import candidate_set_kd

class FourLayerTests(unittest.TestCase):
    def test_independent_layer_matches_mean_and_detached_teacher(self):
        class Matcher:
            def __call__(self, outputs, targets):
                q=int(outputs['pred_logits'][0,:,0].argmax())
                return {'indices':[(torch.tensor([q]),torch.tensor([0]))]}
        target=[{'boxes':torch.tensor([[.5,.5,.2,.2]]),'labels':torch.tensor([0])}]
        far=torch.tensor([[[.1,.1,.1,.1],[.9,.9,.1,.1],[.1,.9,.1,.1]]])
        qs=[0,1,2,0];layers=[];teachers=[]
        for i,q in enumerate(qs):
            logits=torch.tensor([[[0.,9.],[0.,8.],[0.,7.]]]);logits[0,q,0]=1+i*.2
            logits.requires_grad_();boxes=far.clone();boxes[0,q]=target[0]['boxes'][0]
            t=torch.zeros_like(logits);t[0,q,0]=3;t.requires_grad_()
            layers.append({'pred_logits':logits,'pred_boxes':boxes});teachers.append(t)
        # Reverse teacher ranking at two layers: they must still count as zero in /4.
        teachers[1]=torch.tensor([[[4.,0.],[0.,0.],[0.,0.]]],requires_grad=True)
        teachers[2]=torch.tensor([[[4.,0.],[0.,0.],[0.,0.]]],requires_grad=True)
        outputs={**layers[-1],'aux_outputs':layers[:-1],'dn_aux_outputs':[{'bad':'ignored'}]}
        result=four_layer_candidate_set_kd(outputs,teachers,target,Matcher(),global_image_count=2,ddp_world_size=2,normal_query_count=3,negative_topk=2)
        expected=[]
        for layer,t,q in zip(layers,teachers,qs):
            r=candidate_set_kd(layer['pred_logits'],t,layer['pred_boxes'],target,[(torch.tensor([q]),torch.tensor([0]))],global_image_count=2,ddp_world_size=2,normal_query_count=3,negative_topk=2)
            expected.append(r['loss'])
        self.assertTrue(torch.allclose(result['loss'],torch.stack(expected).mean()))
        self.assertEqual([float(x)==0 for x in result['layer_losses']],[False,True,True,False])
        result['loss'].backward()
        self.assertGreater(float(layers[-1]['pred_logits'].grad.abs().sum()),0)
        self.assertEqual(float(layers[-1]['pred_logits'].grad[:,:,1].abs().sum()),0)
        self.assertTrue(all(t.grad is None for t in teachers))

    def test_empty_images_stay_zero_and_keep_gradient_graph(self):
        class Matcher:
            def __call__(self, outputs, targets):
                return {'indices':[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]}
        layers=[{'pred_logits':torch.randn(1,3,2,requires_grad=True),
                 'pred_boxes':torch.rand(1,3,4)} for _ in range(4)]
        outputs={**layers[-1],'aux_outputs':layers[:-1]}
        result=four_layer_candidate_set_kd(outputs,[x['pred_logits'].detach() for x in layers],
            [{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)}],Matcher(),
            global_image_count=4,ddp_world_size=2,normal_query_count=3)
        self.assertEqual(float(result['loss']),0)
        result['loss'].backward()
        self.assertTrue(all(x['pred_logits'].grad is not None and x['pred_logits'].grad.abs().sum()==0 for x in layers))

    def test_teacher_export_preserves_bn_dropout_eval(self):
        from types import SimpleNamespace
        from scripts.ablation import train_cmp5L_ec_v as v
        inner=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Dropout(.7)).eval()
        teacher=SimpleNamespace(decoder=SimpleNamespace(decoder=inner))
        with v.teacher_all_layer_outputs(teacher):
            self.assertTrue(inner.training)
            self.assertFalse(inner[0].training)
            self.assertFalse(inner[1].training)
        self.assertFalse(inner.training)

if __name__=='__main__':unittest.main()

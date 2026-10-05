"""One opt-in four-layer V3 arm around the preserved historical training engine."""
import argparse
import json
import sys
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ecdetseg'),str(ROOT)]
from scripts.ablation import train_cmp5L_ec_v as v
from scripts.ablation.cmp5L_ec_v3_four_layer import four_layer_candidate_set_kd
from scripts.ablation.run_historical_v_audited import install_initialization_observer
from scripts.ablation.cmp5L_repro_audit import audit_loaded_initialization

ORIGINAL_RECIPE=v.recipe
ORIGINAL_OBJECTIVE=v.objective
ORIGINAL_TEACHER=v._teacher_forward


def recipe(arm):
    return ('ECX3','Y1') if arm=='V3_4L' else ORIGINAL_RECIPE(arm)


def objective(arm, outputs, teacher_logits, targets, matcher, *, global_images, ddp_world):
    if v.ACTIVE.args.v_arm!='V3_4L':
        return ORIGINAL_OBJECTIVE(arm,outputs,teacher_logits,targets,matcher,
                                  global_images=global_images,ddp_world=ddp_world)
    result=four_layer_candidate_set_kd(outputs,teacher_logits,targets,matcher,
                                       global_image_count=global_images,ddp_world_size=ddp_world)
    if v.ACTIVE.args.smoke and v.rank()==0 and not getattr(v.ACTIVE,'layer_smoke_saved',False):
        row={'scope':'first smoke objective, possibly calibration; not formal coefficient',
             'layers':[0,1,2,3],'layer_weights':[.25]*4,
             'raw_losses':[float(x.detach()) for x in result['layer_losses']],
             'valid_pair_counts':result['layer_valid_pair_counts'],
             'class_counts':result['layer_matched_class_counts']}
        if not all(torch.isfinite(x).all() for x in result['layer_losses']):
            raise FloatingPointError('nonfinite four-layer KL')
        (v.ACTIVE.output/'four_layer_loss_smoke_audit.json').write_text(json.dumps(row,indent=2)+'\n')
        v.ACTIVE.layer_smoke_saved=True
    return result


def teacher_forward(runtime,replay,topk,samples,absolute_targets,autocast_enabled):
    if v.ACTIVE is None or v.ACTIVE.args.v_arm!='V3_4L':
        return ORIGINAL_TEACHER(runtime,replay,topk,samples,absolute_targets,autocast_enabled)
    # Same validated all-layer emission and final-layer parity path as historical V2.
    with v.teacher_all_layer_outputs(runtime.teacher):
        teacher_layers,audit=v._base_teacher_forward(runtime,replay,topk,samples,
                                                     absolute_targets,autocast_enabled)
    if len(audit['teacher_logits'])!=4:
        raise RuntimeError('V3_4L teacher must export four post-LQE logits')
    if not v.ACTIVE.v2_parity_checked:
        _,normal=v._base_teacher_forward(runtime,replay,topk,samples,
                                         absolute_targets,autocast_enabled)
        parity={name:float((audit[key][-1]-normal[key][-1]).abs().max())
                for name,key in [('logits_max_abs','teacher_logits'),('boxes_max_abs','teacher_boxes')]}
        if max(parity.values())>1e-5:
            raise RuntimeError('V3_4L last-layer teacher eval parity failed: '+str(parity))
        v.ACTIVE.v2_parity=parity;v.ACTIVE.v2_parity_checked=True
        if v.rank()==0:
            (v.ACTIVE.output/'four_layer_teacher_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    return teacher_layers,audit


def install():
    v.recipe=recipe
    v.objective=objective
    v._teacher_forward=teacher_forward
    install_initialization_observer(v,audit_loaded_initialization)
    original_state=v.ECFullSolver.state_dict
    original_load=v.ECFullSolver.load_state_dict
    def state_dict(solver):
        state=original_state(solver)
        state.update(ec_v_arm='V3_4L',ec_v_kd_layers=[0,1,2,3])
        return state
    def load_state_dict(solver,state):
        if state.get('last_epoch',-1)>=0 and state.get('ec_v_arm')!='V3_4L':
            raise RuntimeError('V3_4L refuses a checkpoint from another arm')
        return original_load(solver,state)
    v.ECFullSolver.state_dict=state_dict
    v.ECFullSolver.load_state_dict=load_state_dict


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('-c','--config',required=True);p.add_argument('-r','--resume',required=True)
    p.add_argument('--v-arm',choices=['V3_4L'],default='V3_4L')
    p.add_argument('--manifest',required=True);p.add_argument('--fixed-teacher')
    p.add_argument('--init-sha256',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--use-amp',action='store_true');p.add_argument('--output-dir',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--print-rank',type=int,default=0)
    p.add_argument('--print-method',default='builtin');p.add_argument('--local-rank',type=int)
    return p.parse_args()

if __name__=='__main__':
    install();v.main(parse_args())

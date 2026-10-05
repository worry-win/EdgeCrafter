import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]);code=root/'code_snapshot_v1'
sys.path[:0]=[str(code/'ecdetseg'),str(code)]
from engine.core.yaml_utils import load_config
from scripts.ablation.train_cmp5L_ec_v import validate_v_execution
from scripts.ablation.train_cmp5L_ec_v3_four_layer import recipe

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=Path('/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_V0_V7_history_2026-10-05_v1')
lock=json.loads((base/'LOCKED_PROTOCOL.json').read_text());assert sha(base/'LOCKED_PROTOCOL.json')=='bc73956a53cbdedfa3b4b6b866a6382d90f407cb9d1df739668080011b61a724'
cfg=code/'ecdetseg/configs/ecdet/ecdet_l_dinov2s_cmp5L_ECV3_4L.yml'
resolved=load_config(str(cfg),{});validate_v_execution(resolved,world_size=2)
assert (resolved['ec_fullcycle_arm'],resolved['ec_y_arm'])==recipe('V3_4L')
original=dict(lock['configs']['V3']['resolved']);modified=dict(resolved)
assert original.pop('ec_v_arm')=='V3' and modified.pop('ec_v_arm')=='V3_4L' and original==modified
for rel,h in lock['original_source_manifests']['code_snapshot_v1'].items():
 if not rel.startswith('slurm/'):assert sha(code/rel)==h,rel
assert sha(lock['initialization']['path'])==lock['initialization']['sha256']
assert sha(lock['calibration_manifest']['path'])==lock['calibration_manifest']['sha256']
for ds in lock['datasets'].values():assert sha(ds['annotation'])==ds['annotation_sha256']
protocol={'date':'2026-10-05','arm':'V3_4L','base_protocol':str(base/'LOCKED_PROTOCOL.json'),'base_protocol_sha256':sha(base/'LOCKED_PROTOCOL.json'),'baseline':'V3 Job4526; normal best EMA main','source_commit':sys.argv[2],'config':{'path':str(cfg),'sha256':sha(cfg),'resolved':resolved},'method':{'teacher':'student EMA NP after HE/pre projection, background0.2','loss':'original candidate_set_kd independently L0-L3; each-layer own Hungarian/top20/strict teacher rank','layers':[0,1,2,3],'layer_weights':[.25]*4,'empty_or_invalid_layers':'zero, retained in /4','ddp':'each layer original global image average including empty images; world-size compensation','temperature':1,'box_kd':False,'hidden_kd':False,'student_intervention':False},'initialization':lock['initialization'],'datasets':lock['datasets'],'calibration':{**lock['calibration_manifest'],'box_coefficient':0,'classification_target':'aggregated four-layer KD gradient on decoder L3 parameters / original detection gradient median0.30','not_fixed_whole_model_gradient_budget':True,'formal_t':10,'validation_tuning':False},'execution':{'seed':42,'world_size':2,'per_gpu_batch':16,'global_batch':32,'accumulation':1,'epochs':100,'early_stop':False,'epoch98_best_reload':True,'validation_only':True,'normal_ema_eval':True,'best_primary_final_supplementary':True},'environment':lock['environment'],'source_manifest':{str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'},'source_additions_only':True,'evaluation':lock['evaluation']}
(root/'LOCKED_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
print(sha(root/'LOCKED_PROTOCOL.json'))

import hashlib,json,sys,platform,importlib.metadata
from collections import Counter
from pathlib import Path
import torch
RUN=Path(sys.argv[1]); OLD=Path('/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_V_eight_v1')
SHARED=Path('/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_fullcycle_v1/shared')
code=RUN/'code_snapshot_v1';sys.path[:0]=[str(code/'ecdetseg'),str(code)]
from engine.core.yaml_utils import load_config
from scripts.ablation.train_cmp5L_ec_v import validate_v_execution,recipe
from scripts.ablation.cmp5L_repro_audit import decoder_digest

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def manifest(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.pth']}
oldmanifest={};snapshots={}
for label in ['code_snapshot_v1','code_snapshot_v7_fix1']:
 oldmanifest[label]=manifest(OLD/label);snapshots[label]=manifest(RUN/label)
 for name,h in oldmanifest[label].items():
  if name.startswith('slurm/'):continue
  assert snapshots[label].get(name)==h,(label,name,'historical source drift')
initial=SHARED/'student_init_seed42.pth'; m=json.loads((SHARED/'manifest.json').read_text())
assert sha(initial)==m['student_init_sha256']=='94f3f1876b9923b0263c1ba55ee4dea13ce4c5ff504288f2f85675887f4111e0'
assert sha(m['pretrained_path'])==m['pretrained_sha256']
for batch in m['batches']:
 path=Path('/cobot/Code/wanrui/EdgeCrafter')/batch['path'];assert sha(path)==batch['sha256']
weights=torch.load(initial,map_location='cpu',weights_only=True)
assert weights['last_epoch']==-1 and weights['ema']['updates']==0
assert weights['model'].keys()==weights['ema']['module'].keys()
assert all(torch.equal(v,weights['ema']['module'][k]) and torch.isfinite(v).all() for k,v in weights['model'].items())
decoder,count=decoder_digest(weights['model']);assert decoder=='23116e13844e74391e9a5eafb434809fae8cfc3c1f2f03b28921f74a796ae360' and count==159
configs={};history={}
for arm in [f'V{i}' for i in range(8)]:
 label='code_snapshot_v7_fix1' if arm=='V7' else 'code_snapshot_v1';path=RUN/label/'ecdetseg/configs/ecdet'/f'ecdet_l_dinov2s_cmp5L_EC{arm}.yml'
 c=load_config(str(path),{});validate_v_execution(c,world_size=2)
 assert c['ec_v_arm']==arm and (c['ec_fullcycle_arm'],c['ec_y_arm'])==recipe(arm)
 old=load_config(str(OLD/label/'ecdetseg/configs/ecdet'/path.name),{});assert c==old
 configs[arm]={'path':str(path),'sha256':sha(path),'resolved':c}
 hist=OLD/('V7_fix1' if arm=='V7' else arm)/'COMPLETED.json';history[arm]=json.loads(hist.read_text())
datasets={}
for split in ['train','valid']:
 ann=Path(configs['V0']['resolved']['train_dataloader' if split=='train' else 'val_dataloader']['dataset']['ann_file'])
 ds=json.loads(ann.read_text()); ids=[im['id'] for im in ds['images']];assert len(ids)==len(set(ids))
 cats=ds['categories'];assert {c['id'] for c in cats}=={0,1,2,3}
 counts=Counter(a['category_id'] for a in ds['annotations']);perimage=Counter(a['image_id'] for a in ds['annotations'])
 assert all(a['image_id'] in set(ids) and a['category_id'] in counts for a in ds['annotations'])
 assert all(len(a['bbox'])==4 and a['bbox'][2]>0 and a['bbox'][3]>0 for a in ds['annotations'])
 root=Path(configs['V0']['resolved']['train_dataloader' if split=='train' else 'val_dataloader']['dataset']['img_folder'])
 rows=[]
 for im in ds['images']:
  file=root/im['file_name'];assert file.is_file();rows.append({'image_id':im['id'],'file_name':im['file_name'],'size':file.stat().st_size,'sha256':sha(file)})
 target=RUN/f'{split}_image_manifest.json';target.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 datasets[split]={'annotation':str(ann),'annotation_sha256':sha(ann),'image_root':str(root),'images':len(ids),'annotations':len(ds['annotations']),'empty_images':sum(perimage[i]==0 for i in ids),'categories':cats,'class_gt_counts':dict(counts),'image_manifest_path':str(target),'image_manifest_sha256':sha(target)}
 assert split!='valid' or (len(ids)==2975 and len(ds['annotations'])==2005)
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
protocol={'date':'2026-10-05','arms':list(configs),'split':'validation only','seed':42,'world_size':2,'global_batch':32,'epochs':100,'epoch98_reload_best':True,'source_commit':sys.argv[2],'original_source_manifests':oldmanifest,'execution_source_manifests':snapshots,'configs':configs,'initialization':{'path':str(initial),'sha256':sha(initial),'last_epoch':-1,'model_equals_ema':True,'decoder_tensor_sha256':decoder,'decoder_state_tensors':count},'pretrained':{'path':m['pretrained_path'],'sha256':m['pretrained_sha256']},'calibration_manifest':{'path':str(SHARED/'manifest.json'),'sha256':sha(SHARED/'manifest.json'),'batches':m['batches'],'target_cls_ratio':.30,'target_box_ratio':.10,'box_cap':10},'datasets':datasets,'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'packages':packages},'historical_completed':history,'original_nodes':{'V0':'cu04','V1':'cu04','V2':'cu01','V3':'cu01','V4':'cu02','V5':'cu02','V6':'cu02','V7':'cu01'},'evaluation':{'weights':'ema','forward':'normal student; own topk; no GT intervention','batch_size':8,'num_workers':4,'postprocess':'unchanged historical ECPostProcessor; no extra score truncation','maxDets':[1,10,100],'fixed_score':.5,'fixed_iou':.5},'known_limits':['Historical full image byte manifest and complete environment lock absent; current data bytes locked, no proof of historical image-byte identity.','Seed42 and identical initialization do not guarantee bitwise CUDA training trajectory.','Calibration recomputed at t10 from training only, old coefficients not forced.','Read-only solver initialization observer adds synchronization but no RNG draws, losses, optimizer or CUDA-math changes.']}
(RUN/'LOCKED_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({'protocol_sha256':sha(RUN/'LOCKED_PROTOCOL.json'),'datasets':datasets,'decoder':decoder,'source_match':True},ensure_ascii=False))

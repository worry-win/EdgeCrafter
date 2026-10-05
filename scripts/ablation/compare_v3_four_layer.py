"""Read-only comparison against contemporaneous final-layer V3."""
import json,hashlib,math,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def finite(v):
 if isinstance(v,float):return math.isfinite(v)
 if isinstance(v,list):return all(finite(x) for x in v)
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 return True

root=Path(sys.argv[1]);sources={'V3_4L':root/'V3_4L','V3':Path('/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_V0_V7_history_2026-10-05_v1/V3')}
r={'protocol_sha256':sha(root/'LOCKED_PROTOCOL.json'),'models':{},'pending':[],'intervals':None,'scope':'validation normal best EMA primary; final contains epoch98 reload'};same_ids=None
for name,path in sources.items():
 mark=path/'COMPLETED.json'
 if not mark.exists():r['pending'].append({'arm':name,'failed':(path/'FAILED.json').exists()});continue
 c=json.loads(mark.read_text());audits={}
 for label in ['best','final']:
  v=c['validation'][label];p=Path(v['prediction_path']);ids=[]
  with p.open() as f:
   for line in f:
    row=json.loads(line);assert finite(row);ids.append(row['image_id'])
  assert len(ids)==len(set(ids))==2975
  if same_ids is None:same_ids=set(ids)
  assert set(ids)==same_ids
  assert sha(p)==v['prediction_sha256']
  checkpoint=c['checkpoints'][label];assert sha(checkpoint['path'])==checkpoint['sha256']
  audits[label]={'image_count':len(ids),'finite':True,'prediction_sha256':sha(p),'checkpoint_sha256':checkpoint['sha256']}
 r['models'][name]={'completed':c,'hash_audits':audits,'source':str(path),'completed_sha256':sha(mark)}
r['status']='incomplete' if r['pending'] else 'complete'
if not r['pending']:
 a=r['models']['V3_4L']['completed'];b=r['models']['V3']['completed']
 assert a['init_sha256']==b['init_sha256']
 r['differences_pp']={label:{k:100*(a['validation'][label]['official'][k]-b['validation'][label]['official'][k]) for k in ['ap','ap50','ap75','ar100']} for label in ['best','final']}
 for label in ['best','final']:
  r['differences_pp'][label]['duct_ap50']=100*(a['validation'][label]['per_class_ap50']['3']['ap50']-b['validation'][label]['per_class_ap50']['3']['ap50'])
 r['fixed_count_differences']={label:{k:a['validation'][label]['fixed_rule']['full'][k]-b['validation'][label]['fixed_rule']['full'][k] for k in ['tp','fp','fn']} for label in ['best','final']}
out=root/'comparison';out.mkdir(exist_ok=True);date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d');stem=out/f'cmp5L_V3_4L_vs_V3_{date}'
lines=['# 四层V3与末层V3 — '+date,'','状态：'+r['status'],'','|组|AP|AP50|AP75|AR100|导管AP50|TP|FP|空图FP|分类λ|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,item in r['models'].items():
 c=item['completed'];v=c['validation']['best'];m=v['official'];fixed=v['fixed_rule']
 vals=[name]+[f'{100*m[k]:.3f}' for k in ['ap','ap50','ap75','ar100']]+[f"{100*v['per_class_ap50']['3']['ap50']:.3f}",fixed['full']['tp'],fixed['full']['fp'],fixed['empty']['fp'],c['calibration']['coefficient']]
 lines.append('| '+' | '.join(str(x) for x in vals)+' |')
lines+=['','best为主，final包括epoch98回载收尾。固定规则score≥0.5、同类IoU≥0.5逐图按分数一对一。','四层各自集合KL等权平均、t10分类系数校准到L3梯度比0.30。浅层新增梯度使总模型KD预算不保证等于末层V3；不能将差异全归因于层数而忽略监督预算。','当前只提供单seed固定模型点估计，未计算本批新区间，不宣称显著性、跨seed稳定或消除背景依赖。未运行独立test。','所有best/final原始指标、正式校准、checkpoint/预测哈希与证据路径见JSON。']
stem.with_suffix('.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if r['status']=='complete':(out/'COMPLETED.json').write_text(json.dumps({'report':str(stem.with_suffix('.md')),'json':str(stem.with_suffix('.json')),'report_sha256':sha(stem.with_suffix('.md')),'json_sha256':sha(stem.with_suffix('.json'))},indent=2)+'\n')
print(stem)

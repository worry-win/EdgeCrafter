"""Read-only completion audit and point-estimate comparisons for eight V arms."""
import json,hashlib,math,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return True

def table(head,rs):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rs])

root=Path(sys.argv[1]);lock=json.loads((root/'LOCKED_PROTOCOL.json').read_text());result={'protocol_sha256':sha(root/'LOCKED_PROTOCOL.json'),'source_commit':lock['source_commit'],'models':{},'pending':[],'paired_intervals':None,'comparison_type':'single-seed fixed model point estimates'}
common=None
for arm in lock['arms']:
 p=root/arm;mark=p/'COMPLETED.json'
 if not mark.exists():
  result['pending'].append({'arm':arm,'failed':(p/'FAILED.json').exists(),'path':str(p)});continue
 c=json.loads(mark.read_text());assert c['epochs']==100 and c['init_sha256']==lock['initialization']['sha256'] and c['v_arm']==arm
 a=[json.loads((p/f'decoder_init_rank{i}.json').read_text()) for i in range(2)]
 assert all(r['decoder_tensor_sha256']==lock['initialization']['decoder_tensor_sha256'] and r['decoder_state_tensors']==159 and r['student_equals_initial'] and r['ema_equals_initial'] and r['before_optimizer_update'] for r in a)
 logs=rows(p/'log.txt');epochs=rows(p/'ecfull_epoch_summary.jsonl');assert [r['epoch'] for r in logs]==[r['epoch'] for r in epochs]==list(range(100))
 assert finite(epochs)
 audits={};cal=c.get('calibration') or {}
 for label in ['best','final']:
  v=c['validation'][label];pred=Path(v['prediction_path']);ids=[]
  with pred.open() as f:
   for line in f:
    r=json.loads(line);assert finite(r);ids.append(r['image_id'])
  assert len(ids)==len(set(ids))==lock['datasets']['valid']['images']
  if common is None:common=set(ids)
  assert set(ids)==common
  h=sha(pred);assert h==v['prediction_sha256'];checkpoint=c['checkpoints'][label];assert sha(checkpoint['path'])==checkpoint['sha256']
  audits[label]={'images':len(ids),'unique_ids':len(set(ids)),'finite':True,'prediction_sha256':h,'checkpoint_sha256':checkpoint['sha256']}
 old=lock['historical_completed'][arm]
 difference={label:{k:100*(c['validation'][label]['official'][k]-old['validation'][label]['official'][k]) for k in ['ap','ap50','ap75','ar100']} for label in ['best','final']}
 original=Path('/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/cmp5L_EC_V_eight_v1')/('V7_fix1' if arm=='V7' else arm)
 oldlogs=rows(original/'log.txt')
 prefix={str(i):{'new_ap50':logs[i]['test_coco_eval_bbox'][1], 'old_ap50':oldlogs[i]['test_coco_eval_bbox'][1],'new_updates':epochs[i]['updates'],'new_amp_skips':epochs[i]['amp_skips']} for i in [0,1,2,9,10,49,57,79,98,99]}
 result['models'][arm]={'completed':c,'completed_sha256':sha(mark),'initialization_audits':a,'prediction_audits':audits,'epoch_monitor':epochs,'historical_best_final_differences_pp':difference,'early_and_late_trajectory':prefix,'cuda_probe':json.loads((p/'no_update_diagnostics/cuda_backward_probe.json').read_text()),'fixed_gradient_diagnostics':rows(p/'ec_y_fixed_gradient_ratios.jsonl') if (p/'ec_y_fixed_gradient_ratios.jsonl').exists() else []}
result['status']='incomplete' if result['pending'] else 'complete'
comparisons=[('V1','V0'),('V3','V0')]+[(f'V{i}','V1') for i in [2,3,4,5,6,7]]
result['same_batch_comparisons_pp']={f'{l}-{r}':{label:{k:100*(result['models'][l]['completed']['validation'][label]['official'][k]-result['models'][r]['completed']['validation'][label]['official'][k]) for k in ['ap','ap50','ap75','ar100']} for label in ['best','final']} for l,r in comparisons if l in result['models'] and r in result['models']}
date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d');out=root/'comparison';out.mkdir(exist_ok=True);stem=out/f'cmp5L_V0_V7_history_replay_{date}'
metricrows=[];fixedrows=[];calrows=[];histrows=[]
for arm,item in result['models'].items():
 c=item['completed'];v=c['validation']['best'];m=v['official'];f=v['fixed_rule'];cal=c.get('calibration') or {};oldcal=lock['historical_completed'][arm].get('calibration') or {}
 metricrows.append([arm,c['checkpoints']['best']['last_epoch']]+[f'{100*m[k]:.3f}' for k in ['ap','ap50','ap75','ar100']]+[f"{100*v['per_class_ap50'][str(i)]['ap50']:.3f}" for i in range(4)])
 fixedrows.append([arm]+[f['full'][k] for k in ['tp','fp','fn']]+[f"{100*f['full']['f1']:.3f}",f['positive']['fp'],f['empty']['fp']])
 calrows.append([arm,cal.get('coefficient','无'),oldcal.get('coefficient','无'),cal.get('box_coefficient','无'),oldcal.get('box_coefficient','无'),cal.get('weighted_median_ratio','无')])
 histrows.append([arm]+[f"{item['historical_best_final_differences_pp']['best'][k]:+.3f}" for k in ['ap','ap50','ap75','ar100']])
lines=['# V0–V7 历史回溯复现 — '+date,'','状态：'+result['status']+'。主比较为本批best EMA；final仅补充，包含历史epoch98回载best收尾。只使用validation，不运行独立test。','',table(['组','best epoch(0-based)','AP','AP50','AP75','AR100','实性AP50','囊性AP50','淋巴结AP50','导管AP50'],metricrows),'','## 固定规则：score≥0.5、同类IoU≥0.5、逐图按分数一对一匹配','',table(['组','TP','FP','FN','F1 %','有GT图FP','空图FP'],fixedrows),'','## 正式t10训练集校准（不混用smoke）','',table(['组','本批分类λ','历史分类λ','本批boxλ','历史boxλ','加权校准中位数'],calrows),'','## 本批减历史，同组best点估计差（百分点）','',table(['组','ΔAP','ΔAP50','ΔAP75','ΔAR100'],histrows),'','代码/配置/初始权重一致不保证CUDA轨迹逐位一致。初始化rank审计、0/9等轮轨迹、每轮校准系数、AMP跳步、生产注意力反向重复实验、checkpoint/预测SHA及原始指标见JSON。','当前报告未计算新配对区间，不宣称跨seed稳定性或方法显著胜出。历史完整图像字节和包/驱动锁定记录缺失，不能宣称环境、历史数据字节已完全排除。','NP蒸馏收益与机制解释分别处理；缺少normal EMA同损失对照，不将V3全部变化单独归因于NP。','', 'Pending: '+json.dumps(result['pending'],ensure_ascii=False)]
stem.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if result['status']=='complete':(out/'COMPLETED.json').write_text(json.dumps({'report':str(stem.with_suffix('.md')),'json':str(stem.with_suffix('.json')),'report_sha256':sha(stem.with_suffix('.md')),'json_sha256':sha(stem.with_suffix('.json'))},indent=2)+'\n')
print(stem)

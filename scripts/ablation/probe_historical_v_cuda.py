"""No-update CUDA evidence probe, separate process from all real training."""
import json,sys,subprocess,platform,importlib.metadata
from pathlib import Path
import torch
from engine.edgecrafter.utils import deformable_attention_core_func_v2
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42)
shapes=[(80,80),(40,40),(20,20)]
value=torch.randn(2,sum(h*w for h,w in shapes),8,32,device='cuda',requires_grad=True)
locations=torch.rand(2,300,8,12,2,device='cuda',requires_grad=True)
weights=torch.randn(2,300,8,12,device='cuda').softmax(-1).requires_grad_()
records=[];reference=None
for i in range(10):
 result=deformable_attention_core_func_v2(value,shapes,locations,weights,[3,6,3],value_shape='reshape')
 grads=torch.autograd.grad(result.square().sum(),(value,locations,weights))
 if reference is None:reference=[x.detach().clone() for x in grads]
 records.append({'repeat':i,'output_sum':float(result.sum()),'gradient_max_abs_vs_first':[float((x-r).abs().max()) for x,r in zip(grads,reference)],'gradient_changed_elements':[int((x!=r).sum()) for x,r in zip(grads,reference)]})
blocked=None
try:
 torch.use_deterministic_algorithms(True)
 result=deformable_attention_core_func_v2(value,shapes,locations,weights,[3,6,3],value_shape='reshape')
 torch.autograd.grad(result.square().sum(),(value,locations,weights))
except RuntimeError as exc:
 blocked=str(exc)
finally:
 torch.use_deterministic_algorithms(False)
driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True)
row={'scope':'no training updates; production deformable attention CUDA backward repeated identical inputs','host':platform.node(),'torch':torch.__version__,'cuda':torch.version.cuda,'visible_gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'driver_inventory':driver,'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','scipy','timm','pycocotools']},'repeats':records,'observed_nondeterministic_gradients':any(any(r['gradient_changed_elements']) for r in records),'deterministic_probe_error':blocked,'limitations':'This proves or rules out repeat differences for this probe, not attribution of historical AP drops. Formal training runs in another fresh process with historical default math.'}
(out/'cuda_backward_probe.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps({'host':row['host'],'nondeterministic':row['observed_nondeterministic_gradients'],'probe_error':blocked}))

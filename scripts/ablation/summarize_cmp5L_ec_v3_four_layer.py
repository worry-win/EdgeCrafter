"""Preserve original completion checks; additionally require four-layer replay parity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ecdetseg'),str(ROOT)]
from scripts.ablation import summarize_cmp5L_ec_v as summary
from scripts.ablation.train_cmp5L_ec_v3_four_layer import recipe

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['arm','v-arm','out-dir','config','manifest','init-sha256','ann-file']:
        p.add_argument('--'+name,required=True)
    args=p.parse_args();assert args.v_arm=='V3_4L'
    parity=json.loads((Path(args.out_dir)/'four_layer_teacher_parity.json').read_text())
    assert max(parity.values())<=1e-5
    summary.recipe=recipe;summary.main(args)
    marker=Path(args.out_dir)/'COMPLETED.json';data=json.loads(marker.read_text())
    data.update(method='four-layer original candidate-set KL',layers=[0,1,2,3],
                layer_weights=[.25]*4,box_kd=False,hidden_kd=False,
                teacher_all_layer_eval_parity=parity,baseline_comparison='contemporaneous V3 Job4526')
    marker.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

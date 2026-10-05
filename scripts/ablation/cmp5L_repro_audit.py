import hashlib, json
from pathlib import Path
import torch

def decoder_digest(state):
    digest = hashlib.sha256()
    count = 0
    for key in sorted(state):
        if not key.startswith('decoder.'):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise RuntimeError('decoder state empty')
    return digest.hexdigest(), count

def audit_loaded_initialization(solver, args, unwrap):
    if solver.last_epoch != -1:
        raise RuntimeError('this locked fresh run must start at epoch -1')
    initial = torch.load(args.resume, map_location='cpu', weights_only=True)
    expected, count = decoder_digest(initial['model'])
    student, _ = decoder_digest(unwrap(solver.model).state_dict())
    ema, _ = decoder_digest(solver.ema.module.state_dict())
    if student != expected or ema != expected:
        raise RuntimeError('loaded student/EMA decoder differs from shared initialization')
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    row = {'arm': args.v_arm, 'rank': rank, 'initialization_file_sha256': args.init_sha256, 'decoder_tensor_sha256': expected, 'decoder_state_tensors': count, 'student_equals_initial': student == expected, 'ema_equals_initial': ema == expected, 'before_optimizer_update': True}
    if torch.distributed.is_initialized():
        rows = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rows, row)
        if len({x['decoder_tensor_sha256'] for x in rows}) != 1:
            raise RuntimeError('DDP decoder initialization differs')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / f'decoder_init_rank{rank}.json').write_text(json.dumps(row, indent=2) + '\n')
    print('DECODER_INIT_AUDIT ' + json.dumps(row))

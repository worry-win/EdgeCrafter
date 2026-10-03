from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
lock=json.loads((root/'docs/reproduction/V3_V4_FROZEN.json').read_text())
for rel,expected in lock['files'].items():
 p=root/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
  raise SystemExit('Frozen source differs: '+rel)
print('All frozen source files match:',len(lock['files']))

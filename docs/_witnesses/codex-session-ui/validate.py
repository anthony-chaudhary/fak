#!/usr/bin/env python3
"""Regenerate hashes and validate the issue #8742 deterministic dogfood packet."""
from pathlib import Path
import hashlib,json,re,subprocess,sys

def no_window_creationflags():
 return subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
root=Path(__file__).resolve().parent
required=['session.started','plan.updated','tool.streaming','approval.pending','approval.decided','tool.failed','workspace.edit','transport.disconnected','session.reconnected','turn.cancelled','turn.retried','test.completed','session.completed']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(run):
 r=root/'runs'/run; events=[json.loads(x) for x in (r/'semantic.jsonl').read_text().splitlines()]
 kinds=[x['kind'] for x in events]; assert kinds==required
 assert [x for x in events if x['kind']=='session.reconnected'][0]['replayed_effects']==0
 assert [x for x in events if x['kind']=='session.completed'][0]['duplicate_effects']==0
 assert [x for x in events if x['kind']=='turn.cancelled'][0]['workspace_effects']==0
 assert (r/'final.diff').read_bytes()==(root/'expected.diff').read_bytes()
 assert json.loads((r/'test-result.json').read_text())['exit_code']==0
 for state in ('start','streaming','approval','reconnect','denial','completion'):
  for vp in ('desktop','narrow'): assert (r/'captures'/f'{state}-{vp}.svg').is_file()
 return {'semantic_sha256':sha(r/'semantic.jsonl'),'diff_sha256':sha(r/'final.diff'),'capture_count':len(list((r/'captures').glob('*.svg')))}
a=validate('run-1'); b=validate('run-2'); assert a==b
# Execute the focused test from a fresh temporary copy of the committed seed.
import tempfile,shutil
with tempfile.TemporaryDirectory() as td:
 shutil.copytree(root/'seed',Path(td)/'seed')
 cp=subprocess.run(['go','test','./...'],cwd=Path(td)/'seed',text=True,capture_output=True,creationflags=no_window_creationflags())
 assert cp.returncode==0, cp.stdout+cp.stderr
# Privacy guard: reject common credentials and private absolute paths.
pat=re.compile(r'(ghp_|sk-[A-Za-z0-9]{12,}|BEGIN (RSA|OPENSSH) PRIVATE KEY|[A-Z]:\\Users\\|/home/)',re.I)
for p in root.rglob('*'):
 if p.is_file() and p.name!='validate.py': assert not pat.search(p.read_text(errors='ignore')),f'unscrubbed: {p}'
receipt={'schema':'fak-codex-session-ui-dogfood/1','runs_match':True,**a,'focused_test':'go test ./...: PASS','scrub':'PASS','evidence_class':'deterministic fak-controlled replay; no live Codex behavior claimed'}
(root/'packet-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True))
package projectassets

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This opt-in witness executes the installed FAK/DOS binaries and the actual
// OpenCode plugin against an isolated repository; no model or mocked gate runs.
func TestOpenCodeLeaseAdmissionIntegration(t *testing.T) {
	if os.Getenv("FAK_OPENCODE_LEASE_INTEGRATION") != "1" {
		t.Skip("set FAK_OPENCODE_LEASE_INTEGRATION=1 with node, fak, dos and Python DOS installed")
	}
	plugin, err := filepath.Abs(filepath.Join("..", "..", ".opencode", "plugins", "dos-proof-guard.js"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(plugin); err != nil {
		t.Fatalf("plugin not found at %s: %v", plugin, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "node", "--no-warnings", "--input-type=module", "-", plugin, t.TempDir())
	cmd.Stdin = strings.NewReader(openCodeLeaseWitness)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("real OpenCode lease witness: %v\n%s", err, out)
	} else {
		var witnessFound bool
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, `{"schema":"fak.opencode-lease-witness.v1"`) {
				var witness struct {
					Schema            string `json:"schema"`
					ForeignRefBlocked bool   `json:"foreign_ref_blocked"`
					OwnRefAdmitted    bool   `json:"own_ref_admitted"`
					DisjointAdmitted  bool   `json:"disjoint_admitted"`
					ReleaseAdmitted   bool   `json:"release_admitted"`
					PatchBlocked      bool   `json:"patch_blocked"`
					ForeignDOSBlocked bool   `json:"foreign_dos_blocked"`
					OwnDOSAdmitted    bool   `json:"own_dos_admitted"`
					ReadsAdmitted     bool   `json:"reads_admitted"`
					CorruptDOSBlocked bool   `json:"corrupt_dos_blocked"`
				}
				if err := json.Unmarshal([]byte(line), &witness); err != nil {
					t.Fatalf("failed to parse witness JSON: %v\n%s", err, line)
				}
				if !witness.ForeignRefBlocked || !witness.OwnRefAdmitted || !witness.DisjointAdmitted ||
					!witness.ReleaseAdmitted || !witness.PatchBlocked || !witness.ForeignDOSBlocked ||
					!witness.OwnDOSAdmitted || !witness.ReadsAdmitted || !witness.CorruptDOSBlocked {
					t.Fatalf("witness assertion failure: %+v", witness)
				}
				witnessFound = true
				t.Logf("witness: %s", line)
				break
			}
		}
		if !witnessFound {
			t.Fatalf("real OpenCode lease witness did not emit expected schema:\n%s", out)
		}
	}
}

func TestOpenCodeLeaseFastFailAndCircuitBreaker(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is required for OpenCode lease fast-fail and circuit breaker test")
	}
	root := t.TempDir()
	pluginPath := filepath.Join(root, "dos-proof-guard.js")
	if err := os.WriteFile(pluginPath, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, node, "--no-warnings", "--input-type=module", "-", pluginPath, root)
	cmd.Stdin = strings.NewReader(`
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [pluginFile, root] = process.argv.slice(2);
const plugin = (await import(pathToFileURL(pluginFile).href)).default;
const hooks = await plugin({ directory: root });

// 1. Out-of-workspace scratch writes emit explicit steerage directing to shell commands
{
  const scratchTarget = path.join(root, '..', 'AppData', 'Local', 'Temp', 'opencode', 'scratch.txt');
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-scratch' }, { args: { filePath: scratchTarget } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard] Mutation path is outside this lease workspace'));
      assert.ok(err.message.includes('Temporary scratch files under temp/scratch directories'));
      assert.ok(err.message.includes('use shell commands (e.g. bash or PowerShell)'));
      return true;
    }
  );

  const normalOut = path.join(root, '..', 'sibling', 'file.txt');
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-scratch' }, { args: { filePath: normalOut } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard] Mutation path is outside this lease workspace'));
      assert.ok(!err.message.includes('Temporary scratch files'));
      return true;
    }
  );
}

// 2. Fast-fail rejects malformed path containing null byte without subprocess execution
{
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'write', sessionID: 'sess-malformed' }, { args: { filePath: 'bad\0path.txt' } }),
    /Malformed mutation path: contains null byte/
  );
}

// 3. Structured refusal steerage on COLLISION_RISK containing [RECOVERY] advice
{
  const code = await readFile(pluginFile, 'utf8');
  let execCallCount = 0;
  const mockExec = 'let execCallCount = 0;\n' +
    'const execute = async () => { ' +
    'execCallCount++; ' +
    'const err = new Error("Command failed"); ' +
    'err.stdout = JSON.stringify({schema:"fak.loop-region.v1",admit:false,reason:"COLLISION_RISK",conflict:{holder:"worker-theta"}}); ' +
    'throw err; ' +
    '};';
  const patched = code.replace('const execute = promisify(execFile);', mockExec) +
    '\nexport { execCallCount };';
  const mod = await import('data:text/javascript;base64,' + Buffer.from(patched).toString('base64'));
  const collisionPlugin = await mod.default({ directory: root });

  const collisionSession = 'sess-collision';
  const collisionPath = 'collision-target.go';

  // Failure 1: encounters COLLISION_RISK and receives [RECOVERY] steerage
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('COLLISION_RISK'));
      assert.ok(err.message.includes('[RECOVERY]'));
      assert.ok(err.message.includes("Lane conflict detected with holder 'worker-theta'"));
      assert.ok(err.message.includes("Run 'fak recover COLLISION_RISK' or select a tree-disjoint task."));
      return true;
    }
  );

  // Failure 2: encounters COLLISION_RISK again
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('COLLISION_RISK'));
      assert.ok(err.message.includes('[RECOVERY]'));
      return true;
    }
  );

  // Snapshot execution count before 3rd attempt
  const countBefore3rd = mod.execCallCount;

  // Attempt 3: 3rd consecutive failing edit triggers circuit breaker refusal without dispatching subprocess commands
  await assert.rejects(
    collisionPlugin['tool.execute.before']({ tool: 'edit', sessionID: collisionSession }, { args: { filePath: collisionPath } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard-circuit-breaker]'));
      assert.ok(err.message.includes('Refusing 3rd consecutive failing edit on ' + collisionPath));
      assert.ok(err.message.includes("Stop retrying. Inspect the file with 'read' or pivot to an alternate lane."));
      return true;
    }
  );

  // Verify no subprocess commands were dispatched on 3rd attempt
  assert.equal(mod.execCallCount, countBefore3rd, 'Circuit breaker must refuse without dispatching child CLI commands');
}

// 4. Circuit breaker on string-matching failure and reset on success
{
  const session = 'sess-cb-reset';
  const targetFile = 'cb-reset-test.go';

  // Failure 1 in tool.execute.after
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Could not find oldString in file' }
  );

  // Failure 2 in tool.execute.after
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Found multiple matches for oldString' }
  );

  // Attempt 3: trips circuit breaker in tool.execute.before
  await assert.rejects(
    hooks['tool.execute.before']({ tool: 'edit', sessionID: session }, { args: { filePath: targetFile } }),
    (err) => {
      assert.ok(err.message.includes('[dos-proof-guard-circuit-breaker]'));
      assert.ok(err.message.includes('Refusing 3rd consecutive failing edit on ' + targetFile));
      return true;
    }
  );

  // Successful edit resets the failure counter for that path
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Successfully replaced 1 instance' }
  );

  // Failure 1 after reset
  await hooks['tool.execute.after'](
    { tool: 'edit', sessionID: session, args: { filePath: targetFile } },
    { output: 'Error: Could not find oldString in file' }
  );

  // Next attempt: failure count is 1, so breaker MUST NOT trip
  let tripped = false;
  try {
    await hooks['tool.execute.before']({ tool: 'edit', sessionID: session }, { args: { filePath: targetFile } });
  } catch (err) {
    if (err.message.includes('[dos-proof-guard-circuit-breaker]')) {
      tripped = true;
    }
  }
  assert.equal(tripped, false, 'Circuit breaker must not trip after failure counter reset');
}

console.log(JSON.stringify({ ok: true }));
`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("OpenCode lease fast-fail & circuit breaker test failed: %v\n%s", err, out)
	}
	var receipt struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(out, &receipt); err != nil || !receipt.OK {
		t.Fatalf("unexpected witness output: %s", out)
	}
}

const openCodeLeaseWitness = `
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const pluginPath = process.argv[2], root = process.argv[3];
const invoke = (command, args) => execFileSync(command, args, {
  cwd: root, encoding: 'utf8', timeout: 30000, windowsHide: true,
  stdio: ['ignore', 'pipe', 'pipe'],
});
for (const key of ['FAK_LEASE_OWNER', 'FAK_LEASE_SESSION', 'FAK_LEASE_ID']) delete process.env[key];
// Child commands must never inherit an override pointing at the live fleet WAL.
for (const key of ['DISPATCH_LANE_JOURNAL_PATH', 'JOB_LANE_JOURNAL_PATH', 'DISPATCH_LANE_LEASE_LOCK_PATH', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR']) delete process.env[key];
await mkdir(path.join(root, 'owned'), {recursive:true});
await mkdir(path.join(root, 'free'), {recursive:true});
await writeFile(path.join(root, 'dos.toml'), 'workspace = "."\n[lanes]\nconcurrent = ["owned", "free"]\n[lanes.trees]\nowned = ["owned/**"]\nfree = ["free/**"]\n');
invoke('git', ['init', '--quiet', root]);
const target = path.join(root, 'owned', 'file.txt');
await writeFile(target, 'before');
const plugin = (await import(pathToFileURL(pluginPath).href)).default;
const hooks = await plugin({directory:root});
async function mutate(sessionID, filename, content, tool='write', extra={}) {
  if (hooks['tool.execute.before']) await hooks['tool.execute.before'](
    {tool, sessionID, callID:'witness'}, {args:{filePath:filename, content, ...extra}});
  await writeFile(filename, content);
}
const fak = (args) => JSON.parse(invoke('fak', ['leaseref', ...args, '--dir', root]));
fak(['acquire', '--id', 'codex-witness', '--holder', 'codex:witness', '--session', 'codex-witness', '--tree', 'owned/**', '--ttl', '300', '--announce', 'offline']);
await assert.rejects(mutate('foreign', target, 'clobber', 'write', {owner:'codex:witness', session:'codex-witness', self:'codex-witness'}), (err) => {
  return /dos-proof-guard/.test(err.message) &&
         err.message.includes('[RECOVERY]') &&
         err.message.includes("Lane conflict detected with holder 'codex:witness'") &&
         err.message.includes("Run 'fak recover COLLISION_RISK'");
});
assert.equal(await readFile(target, 'utf8'), 'before');
await mutate('foreign', path.join(root, 'free', 'file.txt'), 'disjoint');
assert.equal(await readFile(path.join(root, 'free', 'file.txt'), 'utf8'), 'disjoint');
fak(['release', '--id', 'codex-witness', '--holder', 'codex:witness', '--announce', 'offline']);
await mutate('foreign', target, 'after-release');
assert.equal(await readFile(target, 'utf8'), 'after-release');
fak(['acquire', '--id', 'own-witness', '--holder', 'opencode:self', '--session', 'self', '--tree', 'owned/**', '--ttl', '300', '--announce', 'offline']);
await mutate('self', target, 'owner');
assert.equal(await readFile(target, 'utf8'), 'owner');
await assert.rejects(mutate('foreign', target, 'patch-clobber', 'apply_patch', {patchText:'*** Begin Patch\n*** Update File: owned/file.txt\n@@\n-owner\n+patch-clobber\n*** End Patch'}), /dos-proof-guard/);
assert.equal(await readFile(target, 'utf8'), 'owner');
fak(['release', '--id', 'own-witness', '--holder', 'opencode:self', '--announce', 'offline']);
invoke('dos', ['lease-lane', '--workspace', root, 'acquire', '--lane', 'owned', '--tree', 'owned/**', '--owner', 'opencode:dos-self', '--run-id', 'dos-self', '--loop-ts', 'witness']);
await assert.rejects(mutate('foreign', target, 'dos-clobber', 'edit'), /dos-proof-guard/);
assert.equal(await readFile(target, 'utf8'), 'owner');
await hooks['tool.execute.before']({tool:'read', sessionID:'foreign'}, {args:{filePath:target}});
await mutate('dos-self', target, 'dos-owner');
assert.equal(await readFile(target, 'utf8'), 'dos-owner');
const journal = path.join(root, '.dos', 'lane-journal.jsonl');
await writeFile(journal, (await readFile(journal, 'utf8')) + '{corrupt}\n');
await assert.rejects(mutate('dos-self', path.join(root, 'free', 'file.txt'), 'corrupt-admit'), /dos-proof-guard/);
assert.equal(await readFile(path.join(root, 'free', 'file.txt'), 'utf8'), 'disjoint');
console.log(JSON.stringify({schema:'fak.opencode-lease-witness.v1', foreign_ref_blocked:true, own_ref_admitted:true, disjoint_admitted:true, release_admitted:true, patch_blocked:true, foreign_dos_blocked:true, own_dos_admitted:true, reads_admitted:true, corrupt_dos_blocked:true}));
`

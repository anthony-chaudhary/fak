package projectassets

import (
	"bytes"
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

func TestOpenCodeProofPlugin_StdoutJSONPurity(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node not found in PATH")
	}

	tmpDir := t.TempDir()
	pluginPath := filepath.Join(tmpDir, "dos-proof-guard.js")
	diskPlugin, err := filepath.Abs(filepath.Join("..", "..", filepath.FromSlash(OpenCodePluginPath)))
	if err == nil {
		if _, statErr := os.Stat(diskPlugin); statErr == nil {
			pluginPath = diskPlugin
		}
	}
	if pluginPath != diskPlugin {
		if err := os.WriteFile(pluginPath, []byte(DefaultOpenCodePlugin), 0644); err != nil {
			t.Fatal(err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "node", "--no-warnings", "--input-type=module", "-", pluginPath, tmpDir)
	cmd.Stdin = strings.NewReader(openCodePurityWitness)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("node process execution failed: %v\nstdout:\n%s\nstderr:\n%s", err, out, stderr.String())
	}

	stdout := string(out)
	lines := strings.Split(stdout, "\n")
	var witnessFound bool

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		var parsed map[string]any
		if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
			if strings.Contains(trimmed, "[dos-proof-guard]") {
				t.Fatalf("unencoded [dos-proof-guard] notice leaked to stdout (not valid JSON): %q", trimmed)
			}
			t.Fatalf("stdout line is not valid JSON (%v): %q", err, trimmed)
		}

		if schema, ok := parsed["schema"].(string); ok && schema == "fak.opencode-purity-witness.v1" {
			var witness struct {
				Schema        string `json:"schema"`
				Status        string `json:"status"`
				EditInjected  bool   `json:"edit_injected"`
				WriteInjected bool   `json:"write_injected"`
				PatchInjected bool   `json:"patch_injected"`
				ReadUnchanged bool   `json:"read_unchanged"`
			}
			if err := json.Unmarshal([]byte(trimmed), &witness); err != nil {
				t.Fatalf("failed to parse witness JSON: %v\n%s", err, trimmed)
			}
			if witness.Status != "ok" || !witness.EditInjected || !witness.WriteInjected || !witness.PatchInjected || !witness.ReadUnchanged {
				t.Fatalf("witness assertion failure: %+v", witness)
			}
			witnessFound = true
		}
	}

	if !witnessFound {
		t.Fatalf("purity witness JSON object not found in stdout:\n%s", stdout)
	}
}

const openCodePurityWitness = `
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const pluginPath = process.argv[2], root = process.argv[3];
const plugin = (await import(pathToFileURL(pluginPath).href)).default;
const hooks = await plugin({ directory: root });

assert.equal(typeof hooks['tool.execute.after'], 'function', 'tool.execute.after hook must be defined');

const reminderSignature = '[dos-proof-guard] Code modified. On-device proof required before completion:';

// 1. tool = "edit" with output = { content: "prior text" }
const editOutput = { content: 'prior text' };
await hooks['tool.execute.after']({ tool: 'edit' }, editOutput);
const editInjected = typeof editOutput.content === 'string' &&
  editOutput.content.startsWith('prior text') &&
  editOutput.content.includes(reminderSignature);

// 2. tool = "write" with output = { content: [{ type: "text", text: "prior" }] }
const writeOutput = { content: [{ type: 'text', text: 'prior' }] };
await hooks['tool.execute.after']({ tool: 'write' }, writeOutput);
const writeInjected = Array.isArray(writeOutput.content) &&
  writeOutput.content.length === 2 &&
  writeOutput.content[0]?.text === 'prior' &&
  typeof writeOutput.content[1]?.text === 'string' &&
  writeOutput.content[1].text.includes(reminderSignature);

// 3. tool = "apply_patch" with output = { content: "prior patch" }
const patchOutput = { content: 'prior patch' };
await hooks['tool.execute.after']({ tool: 'apply_patch' }, patchOutput);
const patchInjected = typeof patchOutput.content === 'string' &&
  patchOutput.content.startsWith('prior patch') &&
  patchOutput.content.includes(reminderSignature);

// 4. tool = "read" (non-mutating) with output = { content: "prior read" }
const readOutput = { content: 'prior read' };
await hooks['tool.execute.after']({ tool: 'read' }, readOutput);
const readUnchanged = readOutput.content === 'prior read';

assert.ok(editInjected, 'edit reminder must be injected into output.content string');
assert.ok(writeInjected, 'write reminder must be injected into output.content array');
assert.ok(patchInjected, 'apply_patch reminder must be injected into output.content string');
assert.ok(readUnchanged, 'read tool output must remain unchanged');

const witness = {
  schema: 'fak.opencode-purity-witness.v1',
  status: 'ok',
  edit_injected: editInjected,
  write_injected: writeInjected,
  patch_injected: patchInjected,
  read_unchanged: readUnchanged,
};
console.log(JSON.stringify(witness));
`

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
await assert.rejects(mutate('foreign', target, 'clobber', 'write', {owner:'codex:witness', session:'codex-witness', self:'codex-witness'}), /dos-proof-guard/);
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

package projectassets

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=2s lane=default
func TestSyncPiGrep(t *testing.T) {
	root := fixture(t)
	before, err := Build(root, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Harnesses["pi"], before.Harnesses["opencode"]) {
		t.Fatal("absent-Pi read-only adapter receipt changed")
	}
	if _, err := os.Stat(filepath.Join(root, ".pi")); !os.IsNotExist(err) {
		t.Fatal("read-only parity created Pi assets")
	}
	after, err := Build(root, true) // actual fak-project-assets sync consumer
	if err != nil || !after.ZeroUnexplainedGaps {
		t.Fatalf("sync: %v, parity=%v", err, after.ZeroUnexplainedGaps)
	}
	file := filepath.Join(root, filepath.FromSlash(PiGrepPath))
	b, err := os.ReadFile(file)
	if err != nil || string(b) != defaultPiGrep {
		t.Fatalf("canonical emitted extension mismatch: %v", err)
	}
	// A clean adapter set with no .pi must preserve nil slices/JSON exactly too.
	for _, p := range []string{file, filepath.Dir(file), filepath.Join(root, ".pi")} {
		if err := os.Remove(p); err != nil {
			t.Fatal(err)
		}
	}
	cleanAbsent, err := Build(root, false)
	if err != nil || !reflect.DeepEqual(cleanAbsent.Harnesses["pi"], cleanAbsent.Harnesses["opencode"]) {
		t.Fatalf("clean absent-Pi receipt changed: err=%v pi=%#v peer=%#v", err, cleanAbsent.Harnesses["pi"], cleanAbsent.Harnesses["opencode"])
	}
	if _, err := os.Stat(filepath.Join(root, ".pi")); !os.IsNotExist(err) {
		t.Fatal("clean read-only parity created Pi assets")
	}
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("// stale\n"), 0644); err != nil {
		t.Fatal(err)
	}
	stale, err := Build(root, false) // actual fak-project-assets parity consumer
	if err != nil || stale.ZeroUnexplainedGaps {
		t.Fatalf("stale Pi asset not reported: %v, parity=%v", err, stale.ZeroUnexplainedGaps)
	}
	if got := stale.Harnesses["pi"].Stale; !reflect.DeepEqual(got, []string{PiGrepPath}) {
		t.Fatalf("wrong Pi stale path: %v", got)
	}
	if _, synced, err := EnsureSync(root); err != nil || !synced {
		t.Fatalf("EnsureSync failed to repair Pi-only drift: synced=%v err=%v", synced, err)
	}
	if _, synced, err := EnsureSync(root); err != nil || synced {
		t.Fatalf("second sync not idempotent: synced=%v err=%v", synced, err)
	}
	if _, err := Ensure(root, false); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	if receipt, err := Build(root, false); err != nil || receipt.ZeroUnexplainedGaps {
		t.Fatalf("missing existing-Pi asset not reported: %v", err)
	}
	if _, err := Build(root, true); err != nil {
		t.Fatal(err)
	}
	t.Run("emitted_tool_call", func(t *testing.T) {
		node, err := exec.LookPath("node")
		if err != nil {
			t.Skip("Node is required for the emitted-extension callback witness; Pi execution remains separate")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, node, "--input-type=module", "-", file)
		cmd.WaitDelay = time.Second
		cmd.Stdin = strings.NewReader(piGrepWitness)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("exact emitted extension callback: %v\n%s", err, out)
		}
	})
}

// This loads the exact sync output and invokes its real registered callback.
// It is a host contract witness, not an installed-Pi/trust/end-to-end claim.
const piGrepWitness = `import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const code = await readFile(process.argv[2], "utf8");
const { default: install } = await import("data:text/javascript;base64," + Buffer.from(code).toString("base64"));
const handlers = new Map(), entries = [], warnings = [];
install({
  on: (name, callback) => { assert.ok(!handlers.has(name)); handlers.set(name, callback); },
  appendEntry: (type, data) => entries.push({ type, data }),
});
assert.deepEqual([...handlers.keys()], ["tool_call"]);
const call = handlers.get("tool_call");
const ctx = { ui: { notify: (message, level) => warnings.push({ message, level }) } };
const cases = [
  ["rg -uu sensitive-pattern /", true],
  ["rg --no-ignore sensitive-pattern /home", true],
  ["rg -uu sensitive-pattern ~", true],
  ["rg -uu sensitive-pattern '~'", false],
  ['rg -uu sensitive-pattern "~"', false],
  ["rg -uu sensitive-pattern \\~", false],
  ["rg -uu sensitive-pattern ''~", false],
  ['rg -uu sensitive-pattern ~"alice"', false],
  ["rg -uu sensitive-pattern ~'alice'", false],
  ["rg -uu sensitive-pattern ~\\alice", false],
  ['rg -uu sensitive-pattern ~""', false],
  ["rg -uu sensitive-pattern ~+", false],
  ["rg -uu sensitive-pattern ~-", false],
  ["rg -uu sensitive-pattern ~alice", false],
  ["rg -uu '~' /", true],
  ["cd '~' && rg -uu sensitive-pattern", false],
  ["cd /work && rg -n 'sensitive-pattern' -uu --glob '!**/.git/**' -l", true],
  ["cd /work && rg -uu -e sensitive-pattern .", true],
  ["rg -uu --files /", true],
  ["rg -uu -e sensitive-pattern /work/repo/../..", true],
  ["rg -uu sensitive-pattern 'C:/'", true],
  ["rg -uu sensitive-pattern /work/repo", false],
  ["rg -uu sensitive-pattern /repo", false, "/repo"],
  ["cd /repo && rg -uu sensitive-pattern .", false, "/repo"],
  ["cd /work && rg -uu sensitive-pattern .", true, "/repo"],
  ["rg -uu sensitive-pattern /work", false, "/work"], // explicit known project, not historical timing context
  ["rg -uu sensitive-pattern /", true, "/"],
  ["rg -uu sensitive-pattern /home", true, "/home"],
  ["rg\u00a0-uu sensitive-pattern /", false],
  ["rg -uu sensitive-pattern /\r", false],
  ["rg -uu sensitive-pattern ./repo", false],
  ["rg -uu sensitive-pattern ~/repo", false],
  ["rg -uu --max-depth 2 sensitive-pattern /work", false],
  ["rg -uu --max-depth=0 sensitive-pattern /", false],
  ["rg -uu --max-filesize 1M sensitive-pattern /work", false],
  ["rg -uu --max-filesize 0 sensitive-pattern /", true],
  ["rg -uu --max-filesize 1M --max-filesize 0 sensitive-pattern /", true],
  ["rg -n sensitive-pattern /work", false],
  ["rg -uu sensitive-pattern", false],
  ["rg -uu /", false], // slash is the pattern, not a path
  ["rg -uu", false], // invalid/incomplete search, not a traversal witness
  ["rg -uu -e '/'", false],
  ["rg -uu -e '--max-depth 2' /", true], // pattern cannot manufacture a bound
  ["rg -- -uu /", false], // -uu is the pattern after --
  ["rg -uu -e sensitive-pattern -- /", true],
  ["rg -uu sensitive-pattern $HOME", false], // unsupported expansion fails open
  ["(cd /work && rg -uu sensitive-pattern)", false],
  ["cd /work || rg -uu sensitive-pattern", false],
  ["echo 'rg -uu sensitive-pattern /'", false],
  ["grep -rn sensitive-pattern /work", false],
  ["grep -e '-recursive-pattern' /work", false],
  ["grep -- '-recursive-pattern' /work", false],
];
for (const [command, blocked, cwd] of cases) {
  const event = { toolName: "bash", input: { command, timeout: 37 } };
  const original = JSON.stringify(event);
  const result = await call(event, cwd === undefined ? ctx : { ...ctx, cwd });
  assert.equal(result?.block === true, blocked, command);
  assert.equal(JSON.stringify(event), original, "command/timeout was mutated");
}
assert.ok(warnings.some(w => w.message.startsWith("FAK_PI_SEARCH_RECURSIVE_GREP")));
assert.equal(await call({ toolName: "grep", input: { pattern: "sensitive-pattern", path: "/" } }, ctx), undefined);
assert.equal(await call({ toolName: "powershell", input: { command: "rg -uu sensitive-pattern /" } }, ctx), undefined);
const bad = { toolName: "bash", input: {} };
Object.defineProperty(bad.input, "command", { get() { throw Error("private-detector-detail"); } });
assert.equal(await call(bad, ctx), undefined);
assert.ok(warnings.some(w => w.message.startsWith("FAK_PI_SEARCH_DETECTOR_ERROR")));
for (let i = 0; i < 40; i++) await call({ toolName: "bash", input: { command: "rg -uu sensitive-pattern /" } }, ctx);
assert.equal(entries.length, 17);
assert.equal(entries[16].data.code, "FAK_PI_SEARCH_TELEMETRY_LIMIT");
const telemetry = JSON.stringify({ entries, warnings });
assert.ok(!telemetry.includes("sensitive-pattern"));
assert.ok(!telemetry.includes("private-detector-detail"));
assert.ok(warnings.length <= 3);
console.log("PASS exact emitted Pi tool_call: original commands, bounded refusal, fail-open, sync lifecycle");
`

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchTickMicroBackendEnrollsIntoHostNotDetachedSpawn is the #2030 acceptance
// witness: a live tick with --backend micro enrolls the routed issue into a REAL
// in-process microagent host (host constructed, agent spawned and retired done, one
// per-agent audit sink saw spawn+done) INSTEAD of exec-spawning a detached CLI, and the
// enrollment holds the SAME lane-lease fence tree the detached path would — so lane-lease
// disjointness (M11) is preserved. The detached spawner is wired to fail the test if the
// micro path ever reaches it.
func TestDispatchTickMicroBackendEnrollsIntoHostNotDetachedSpawn(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{PID: 999, Issue: issue, Lane: lane, Backend: backend}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--backend", "micro", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--live", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for micro host-enroll (stderr: %s)\n%s", code, errb, out)
	}
	if spawned {
		t.Fatal("micro backend called the DETACHED exec spawner; it must enroll into the in-process host instead")
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "enrolled" || got["verdict"] != "ENROLLED" || got["ok"] != true {
		t.Fatalf("micro tick = action %v verdict %v ok %v\n%s", got["action"], got["verdict"], got["ok"], out)
	}
	if got["backend"] != "micro" {
		t.Fatalf("backend = %v, want micro", got["backend"])
	}

	// The enrollment descriptor carries the routed issue + the lease-fence tree.
	enroll := mapAt(got, "host_enrollment")
	if enroll["agent_id"] != "resolve-docs-12" || dispatchMapInt(enroll, "issue") != 12 {
		t.Fatalf("host_enrollment = %#v, want resolve-docs-12/#12", enroll)
	}
	if tree := stringAnySlice(enroll["tree"]); len(tree) != 1 || tree[0] != "docs/**" {
		t.Fatalf("host_enrollment tree = %#v, want [docs/**] (lease fence preserved)", enroll["tree"])
	}

	// Per-agent audit (M11) preserved: the ONE host audit sink saw exactly one spawn and
	// one done for exactly one distinct agent.
	audit := mapAt(got, "host_audit")
	if dispatchMapInt(audit, "spawns") != 1 || dispatchMapInt(audit, "dones") != 1 || dispatchMapInt(audit, "distinct_agents") != 1 {
		t.Fatalf("host_audit = %#v, want 1 spawn / 1 done / 1 distinct agent", audit)
	}
	res := mapAt(got, "host_result")
	if res["done"] != true || dispatchMapInt(res, "steps") < 1 {
		t.Fatalf("host_result = %#v, want done with >=1 step", res)
	}

	// The lane-lease disjointness fence is enforced identically to the detached path: the
	// enrollment acquired the SAME lane lease and was NOT refused (no peer holds it). We
	// assert not-refused rather than acquired==true because a bare temp workspace is not a
	// git repo, so leaseref fail-opens (acquired=false, refused=false) — the invariant that
	// matters for "lane leases still prevent collisions" is that an unheld lane proceeds and
	// a held one would refuse (the shared LANE_LEASE_HELD branch, identical to detached spawn).
	lease := mapAt(got, "lease")
	if lease["refused"] == true {
		t.Fatalf("lease = %#v, want not refused (no peer holds the lane)", lease)
	}

	// #4324: the enrolled path never detaches, so nothing else will witness it. An acquired
	// lease must be released immediately; a bare non-git fixture fail-opens without a lease
	// ID, so there is nothing to release.
	wantRelease := "released"
	if lease["acquired"] != true {
		wantRelease = "no_lease_id"
	}
	if got["lease_release"] != wantRelease {
		t.Fatalf("lease_release = %v, want %s for lease %#v", got["lease_release"], wantRelease, lease)
	}
}

// TestDispatchTickMicroBackendDryRunWouldEnroll pins the dry-run peer of the live path:
// it plans the enrollment (and its lease-fence tree) without constructing a host or
// spawning anything, and never touches the detached spawner.
func TestDispatchTickMicroBackendDryRunWouldEnroll(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--backend", "micro", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	if spawned {
		t.Fatal("dry-run micro tick must not spawn or enroll anything")
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "would_enroll" || got["verdict"] != "WOULD_ENROLL" || got["ok"] != true {
		t.Fatalf("dry-run micro tick = action %v verdict %v ok %v", got["action"], got["verdict"], got["ok"])
	}
	enroll := mapAt(got, "host_enrollment")
	if tree := stringAnySlice(enroll["tree"]); len(tree) != 1 || tree[0] != "docs/**" {
		t.Fatalf("host_enrollment tree = %#v, want [docs/**]", enroll["tree"])
	}
	// A dry run plans but does NOT construct the host, so it records no audit/result.
	if _, ok := got["host_audit"]; ok {
		t.Fatalf("dry-run micro tick recorded host_audit; want none")
	}
}

// mockInProcessToolPlanner is an agent.Planner for testing in-process tool execution.
// It emits a Read tool call on the first turn and reports completion on the second turn.
type mockInProcessToolPlanner struct {
	mu   sync.Mutex
	turn int
}

func (p *mockInProcessToolPlanner) Model() string { return "mock-inprocess-tool" }

func (p *mockInProcessToolPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.turn == 0 {
		p.turn++
		return &agent.Completion{
			Message: agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{
					{
						ID:   "call_read_1",
						Type: "function",
						Function: agent.Func{
							Name:      "Read",
							Arguments: `{"file_path":"docs/test.txt"}`,
						},
					},
				},
			},
			FinishReason: "tool_calls",
		}, nil
	}
	return &agent.Completion{
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: "resolved issue via in-process tool execution",
		},
		FinishReason: "stop",
	}, nil
}

// TestDispatchTickMicroBackendExecutesInProcessToolCalls verifies that the micro backend
// executes tool calls in-process via the owned agent loop (RunGovernedArm) instead of
// spawning a detached CLI process.
func TestDispatchTickMicroBackendExecutesInProcessToolCalls(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	// Seed the file to be read
	docsDir := filepath.Join(root, "docs")
	if err := os.MkdirAll(docsDir, 0o755); err != nil {
		t.Fatalf("mkdir docs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(docsDir, "test.txt"), []byte("sample test content\n"), 0o644); err != nil {
		t.Fatalf("write test.txt: %v", err)
	}

	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{PID: 999, Issue: issue, Lane: lane, Backend: backend}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	oldPlanner := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		return &mockInProcessToolPlanner{}
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldPlanner })

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--backend", "micro", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--live", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for micro host-enroll (stderr: %s)\n%s", code, errb, out)
	}
	if spawned {
		t.Fatal("micro backend called the DETACHED exec spawner; it must execute tool calls in-process")
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "enrolled" || got["verdict"] != "ENROLLED" || got["ok"] != true {
		t.Fatalf("micro tick = action %v verdict %v ok %v\n%s", got["action"], got["verdict"], got["ok"], out)
	}

	res := mapAt(got, "host_result")
	if res["done"] != true {
		t.Fatalf("host_result done = %v, want true", res["done"])
	}

	metricsMap := mapAt(res, "metrics")
	toolCalls := dispatchMapInt(metricsMap, "tool_calls")
	turns := dispatchMapInt(metricsMap, "turns")
	if toolCalls <= 0 {
		t.Fatalf("metrics.tool_calls = %d, want > 0", toolCalls)
	}
	if turns < 2 {
		t.Fatalf("metrics.turns = %d, want >= 2", turns)
	}

	if topToolCalls := dispatchMapInt(res, "tool_calls"); topToolCalls != toolCalls {
		t.Fatalf("host_result.tool_calls = %d, want %d", topToolCalls, toolCalls)
	}
	if topTurns := dispatchMapInt(res, "turns"); topTurns != turns {
		t.Fatalf("host_result.turns = %d, want %d", topTurns, turns)
	}
}

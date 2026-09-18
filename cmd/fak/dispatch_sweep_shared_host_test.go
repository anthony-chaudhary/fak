package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// rotatingSweepRouter hands back ONE lane carrying the next un-drained issue on
// every call. The sweep re-routes each tick (pickDispatchLane -> dispatchRouteIssues),
// so popping one issue per call is what makes tick 1 pick A, tick 2 pick B, tick 3
// pick C. Returning the whole backlog every tick would re-pick the same lowest
// issue and enroll a duplicate agent id, so the rotation is the multi-tick witness.
type rotatingSweepRouter struct {
	mu     sync.Mutex
	issues []int
}

func (r *rotatingSweepRouter) next(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.issues) == 0 {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes:  map[string]dispatchtick.RouterLaneGroup{},
		}, nil
	}
	n := r.issues[0]
	r.issues = r.issues[1:]
	return dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"docs": {Tree: []string{"docs/**"}, Issues: []int{n}, Count: 1},
		},
	}, nil
}

// TestDispatchSweepSharedHostEnrollsAllTicksIntoOneHost is the #13084 acceptance
// witness at the SWEEP front door: a multi-tick `fak dispatch sweep --live
// --backend micro` builds exactly ONE shared microagent host for the whole run
// (host_constructions == 1, counted independently), enrolls every tick's admitted
// agent into it, drains ONCE, and stamps the same counts on the sweep receipt. The
// rotating router forces N distinct ticks; the spawned_count/ticks assertions make a
// test that silently degraded to a single tick fail rather than pass.
func TestDispatchSweepSharedHostEnrollsAllTicksIntoOneHost(t *testing.T) {
	const n = 3
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	// The in-process Read tool targets this file; seed it so the mock planner's tool
	// call succeeds and the agent retires done on its second turn.
	docsDir := filepath.Join(root, "docs")
	if err := os.MkdirAll(docsDir, 0o755); err != nil {
		t.Fatalf("mkdir docs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(docsDir, "test.txt"), []byte("sample test content\n"), 0o644); err != nil {
		t.Fatalf("write docs/test.txt: %v", err)
	}

	// Each tick re-routes; rotate one issue per call so every tick picks a fresh one.
	router := &rotatingSweepRouter{issues: []int{101, 102, 103}}
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = router.next
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })

	oldFetch := dispatchFetchIssue
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{Number: issue, Title: "sweep shared-host issue", Body: "resolve it", Labels: []string{"docs"}}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetch })

	// The micro path must enroll in-process; it must never reach the detached spawner.
	oldSpawner := dispatchIssueWorkerSpawner
	detached := int32(0)
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		atomic.AddInt32(&detached, 1)
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	// ONE shared gateway is constructed per host build; count it to witness
	// host_constructions == 1 independently of the receipt.
	var constructions int32
	oldWorker := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		atomic.AddInt32(&constructions, 1)
		return &mockInProcessToolPlanner{}
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldWorker })

	out, errb, code := runDispatchAt("sweep", "--workspace", root, "--backend", "micro",
		"--lane", "docs", "--live", "--json", "--max-agents", "3", "--max-workers", "8",
		"--settle-s", "0", "--no-loop-ledger")
	if code != 0 {
		t.Fatalf("sweep exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	if got := atomic.LoadInt32(&detached); got != 0 {
		t.Fatalf("micro sweep reached the DETACHED exec spawner %d time(s); it must enroll in-process", got)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad sweep json: %v\n%s", err, out)
	}

	// ONE host for the whole sweep, counted independently of the receipt.
	if got := atomic.LoadInt32(&constructions); got != 1 {
		t.Fatalf("planner constructed %d time(s), want 1 shared gateway for the whole sweep", got)
	}

	sh := mapAt(got, "shared_host")
	if dispatchMapInt(sh, "host_constructions") != 1 {
		t.Fatalf("shared_host.host_constructions = %v, want 1", sh["host_constructions"])
	}
	if dispatchMapInt(sh, "drains") != 1 {
		t.Fatalf("shared_host.drains = %v, want exactly 1 (one drain per sweep)", sh["drains"])
	}
	if dispatchMapInt(sh, "admitted") != n || dispatchMapInt(sh, "enrolled") != n {
		t.Fatalf("shared_host admitted/enrolled = %v/%v, want %d/%d", sh["admitted"], sh["enrolled"], n, n)
	}

	// Multi-tick proof: N spawned across N ticks, not a one-tick pass.
	if got := dispatchMapInt(got, "spawned_count"); got != n {
		t.Fatalf("spawned_count = %d, want %d (one enrolled agent per tick)", got, n)
	}
	ticks, _ := got["ticks"].([]any)
	if len(ticks) != n {
		t.Fatalf("sweep ticks = %d, want %d", len(ticks), n)
	}
	spawnedIssues, _ := got["spawned_issues"].([]any)
	if len(spawnedIssues) != n {
		t.Fatalf("spawned_issues = %#v, want %d distinct issues", got["spawned_issues"], n)
	}
}

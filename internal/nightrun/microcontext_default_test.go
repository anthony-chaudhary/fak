package nightrun

import (
	"context"
	"strings"
	"testing"
	"time"
)

// microcontext_default_test.go is the #5842 witness: the micro-context fabric is
// invoked by the PIPELINE, not by an agent remembering to run it.
//
// The fabric shipped a working spine (cmd/microcontextdemo), a quality ledger, and a
// health scorecard — but no loop ever called it, so every artifact under
// experiments/microcontext/ existed because a human or an agent typed the command.
// The default now lives in the nightrun collection loop's built-in witness registry:
// nightrun is a real Go tick loop AND a registered member of the `manage-benchmarks`
// super loop, so an unattended `run --apply` turn selects the fabric on its own and
// records the outcome as a durable ledger row.
//
// The two tests below split the claim: the first pins the registration (the row is
// real, offline, and auto-runnable), the second drives the ACTUAL loop and proves the
// invocation happens unasked.

// microContextTaskID is the built-in witness row wired in #5842.
const microContextTaskID = "witness-microcontext-fabric-spine"

// findBuiltinTask returns the built-in backlog row with the given id.
func findBuiltinTask(t *testing.T, id string) Task {
	t.Helper()
	tasks, err := Backlog("")
	if err != nil {
		t.Fatalf("assemble built-in backlog: %v", err)
	}
	for _, task := range tasks {
		if task.ID == id {
			return task
		}
	}
	t.Fatalf("task %q is not in the built-in backlog — the loop can never invoke it", id)
	return Task{}
}

// TestMicroContextFabricIsAnAutoRunnableOfflineWitness pins the registration half of
// #5842. A row that is Manual, or gated on hardware this fleet does not have, would be
// SURFACED but never auto-run — the fabric would stay exactly as unwired as before,
// just with a registry entry to point at. So the row must be offline (empty Requires:
// the selfcheck drives a synthetic endpoint) and auto-runnable, and its Run must
// actually invoke the fabric verb rather than describe it.
func TestMicroContextFabricIsAnAutoRunnableOfflineWitness(t *testing.T) {
	task := findBuiltinTask(t, microContextTaskID)

	if len(task.Requires) != 0 {
		t.Errorf("the fabric selfcheck is offline (synthetic endpoint, no weights/GPU/dataset/net); Requires must be empty so every box can collect it, got %v", task.Requires)
	}
	if task.Manual {
		t.Error("Manual:true would make the loop SKIP the row every turn — the fabric would stay hand-run")
	}
	if !task.autoRunnable() {
		t.Errorf("the loop only execs auto-runnable rows; Run=%q is not one (placeholder or prose arrow?)", task.Run)
	}
	if !strings.Contains(task.Run, "./cmd/microcontextdemo") {
		t.Errorf("the row must invoke the micro-context fabric verb itself, got Run=%q", task.Run)
	}
	if !strings.Contains(task.Run, "-selfcheck") {
		t.Errorf("without -selfcheck the spine invariants are not enforced, so a `collected` row would not witness a healthy fabric; got Run=%q", task.Run)
	}
	// A capability-free box (this fleet's common shape: no GPU, no local weights) must
	// find the row feasible, else the default is wired only for hardware nobody has.
	bare := Capabilities{Box: "bare", GPU: "none", Creds: map[string]bool{}}
	if ok, why := bare.Satisfies(task); !ok {
		t.Errorf("the fabric row must be feasible on a bare box, got infeasible: %s", why)
	}
}

// TestNightrunLoopTurnInvokesMicroContextFabricUnasked is the #5842 done condition as a
// test: a loop turn demonstrably invokes the verb WITHOUT being asked.
//
// Nothing here names the fabric task. The loop is handed the real built-in backlog, a
// box with no GPU and no weights, and a ledger in which every OTHER auto-runnable row
// was already collected moments ago (so no other datum is due). The loop then ranks,
// selects, and executes on its own — and the assertion is that what it reached for was
// the micro-context fabric, and that the turn left a durable `collected` ledger row.
func TestNightrunLoopTurnInvokesMicroContextFabricUnasked(t *testing.T) {
	now := time.Date(2026, 8, 12, 3, 0, 0, 0, time.UTC)
	box := "night-box"
	caps := Capabilities{Box: box, GPU: "none", Net: true, Creds: map[string]bool{}}

	tasks, err := Backlog("")
	if err != nil {
		t.Fatalf("assemble built-in backlog: %v", err)
	}

	// Seed the ledger as a box that has already drained everything else it can collect:
	// a fresh `collected` row for every auto-runnable task EXCEPT the fabric. This is the
	// honest steady state of a long-running box — and it isolates the selection, so the
	// task the loop picks is the one the pipeline genuinely considers next.
	var ledger []CollectRow
	for _, task := range tasks {
		if task.ID == microContextTaskID || !task.autoRunnable() {
			continue
		}
		ledger = append(ledger, NewCollectRow(task, box, OutcomeCollected, "", "", time.Second, now.Add(-time.Hour)))
	}

	var invoked []string
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Max: 1,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, task Task, _ string) (Outcome, string, time.Duration, error) {
			invoked = append(invoked, task.Run)
			return OutcomeCollected, "", 2 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatalf("nightrun loop turn: %v", err)
	}

	if len(invoked) != 1 {
		t.Fatalf("one loop turn must execute exactly one task, executed %d: %v (stop reason %q)", len(invoked), invoked, summary.StopReason)
	}
	if !strings.Contains(invoked[0], "./cmd/microcontextdemo") {
		t.Fatalf("the loop turn reached for %q, not the micro-context fabric — the default is not wired into the pipeline", invoked[0])
	}

	// The durable half of the done condition: the turn must leave a ledger row, which is
	// the artifact an operator (or the super-loop walk above nightrun) reads later.
	var row CollectRow
	var found bool
	for _, r := range ledger {
		if r.TaskID == microContextTaskID && !r.IsHeartbeat() {
			row, found = r, true
		}
	}
	if !found {
		t.Fatal("the loop turn left no ledger row for the fabric — an invocation nobody can witness afterwards")
	}
	if row.Outcome != string(OutcomeCollected) {
		t.Errorf("fabric ledger row outcome = %q, want %q", row.Outcome, OutcomeCollected)
	}
	if row.Box != box {
		t.Errorf("fabric ledger row box = %q, want %q (the row must attribute the datum to the box that collected it)", row.Box, box)
	}
}

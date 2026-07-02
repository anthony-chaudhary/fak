package toolprocgate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// TestSupervisorEndToEndDeadlineKill is the seam-1 pipeline witness, all
// in-process and clock-free: spawn a cancellable call with a deadline → Tick
// past the deadline → the context is CANCELLED, the call is revoked, the
// journal records the kill — and the late completion is QUARANTINED through
// the real kernel fold. Monitoring, control, enforcement in one causal chain.
func TestSupervisorEndToEndDeadlineKill(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})

	ctx, cancel := context.WithCancel(context.Background())
	if err := sup.Spawn("t-dl", "bg_fetch", "s1", 10_000, 0, 1_000, cancel); err != nil {
		t.Fatal(err)
	}

	// Within deadline: no action, context live.
	rep, err := sup.Tick(9_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 || ctx.Err() != nil {
		t.Fatalf("premature enforcement: actions=%v ctxErr=%v", rep.Actions, ctx.Err())
	}

	// Past deadline: cancel fires, revocation lands, journal shows KILLED.
	rep, err = sup.Tick(12_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].Advice != toolproc.AdviceKill || !rep.Actions[0].Cancelled {
		t.Fatalf("want one cancelled kill action, got %+v", rep.Actions)
	}
	if ctx.Err() == nil {
		t.Fatal("in-flight context must be cancelled")
	}
	if r, ok := KilledReason("t-dl"); !ok || r != toolproc.ReasonToolDeadlineExceededName {
		t.Fatalf("revocation table: got %q/%t", r, ok)
	}
	if p := procOf(t, rep.Table, "t-dl"); p.State != toolproc.StateKilled {
		t.Fatalf("post-enforcement table must show KILLED, got %s", p.State)
	}

	// The late completion, admitted through the REAL kernel chain, quarantines.
	c := &abi.ToolCall{Tool: "bg_fetch", TraceID: "t-dl"}
	r := &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("late bytes")}}
	v := kernel.New("").AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictQuarantine || v.Reason != toolproc.ReasonToolResultAfterKill {
		t.Fatalf("late completion: want Quarantine/TOOL_RESULT_AFTER_KILL, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// Idempotent: a later Tick takes no further action on the killed call.
	rep, err = sup.Tick(20_000)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range rep.Actions {
		if a.CallID == "t-dl" && a.Advice != toolproc.AdviceQuarantineResult {
			t.Fatalf("killed call must yield no further destructive action, got %+v", a)
		}
	}
}

// TestSupervisorOrphanReap: the owning session ends under a running call —
// Tick reaps it (cancel + revoke citing TOOL_ORPHANED).
func TestSupervisorOrphanReap(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	if err := sup.Spawn("t-orph", "watch_dir", "s2", 0, 0, 1_000, cancel); err != nil {
		t.Fatal(err)
	}
	if err := sup.SessionEnd("s2", 5_000); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(6_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].Advice != toolproc.AdviceReap || !rep.Actions[0].Cancelled {
		t.Fatalf("want one cancelled reap, got %+v", rep.Actions)
	}
	if ctx.Err() == nil {
		t.Fatal("orphaned call's context must be cancelled")
	}
	if r, ok := KilledReason("t-orph"); !ok || r != toolproc.ReasonToolOrphanedName {
		t.Fatalf("revocation reason: got %q/%t", r, ok)
	}
}

// TestSupervisorStallIsAdvisory: a stalled heartbeat yields a probe action —
// reported, never cancelled, never revoked.
func TestSupervisorStallIsAdvisory(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	if err := sup.Spawn("t-stall", "bg_tail", "s1", 0, 5_000, 1_000, cancel); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(60_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].Advice != toolproc.AdviceProbe || rep.Actions[0].Cancelled {
		t.Fatalf("want one advisory probe, got %+v", rep.Actions)
	}
	if ctx.Err() != nil {
		t.Fatal("probe must not cancel")
	}
	if _, ok := KilledReason("t-stall"); ok {
		t.Fatal("probe must not revoke")
	}
	// A pulse recovers it: next Tick is clean.
	if err := sup.Pulse("t-stall", 61_000, "poll-1"); err != nil {
		t.Fatal(err)
	}
	rep, err = sup.Tick(62_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 {
		t.Fatalf("recovered proc must be clean, got %+v", rep.Actions)
	}
}

// TestSupervisorHealthyLifecycleAndPrune: a clean call runs, exits, and its
// journal rows prune after the cutoff; running procs are never pruned.
func TestSupervisorHealthyLifecycleAndPrune(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("t-ok", "search", "s1", 0, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.Spawn("t-run", "monitor", "s1", 0, 0, 1_500, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.Exit("t-ok", 2_000, "ok"); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(3_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 {
		t.Fatalf("healthy table must need no action, got %+v", rep.Actions)
	}
	if err := sup.PruneTerminal(3_000, 2_500); err != nil {
		t.Fatal(err)
	}
	tab, err := sup.Table(4_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(tab.Procs) != 1 || tab.Procs[0].CallID != "t-run" {
		t.Fatalf("want only the running proc after prune, got %+v", tab.Procs)
	}
}

// TestSupervisorRejectsBadObservations: the fail-closed journal contract holds
// at the live entry points too.
func TestSupervisorRejectsBadObservations(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("c1", "x", "s1", 0, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.Spawn("c1", "x", "s1", 0, 0, 2_000, nil); err == nil {
		t.Error("duplicate spawn must refuse")
	}
	if err := sup.Pulse("ghost", 2_000, ""); err == nil {
		t.Error("pulse for unknown call must refuse")
	}
	if err := sup.Exit("c1", 3_000, "fine"); err == nil {
		t.Error("bad exit status must refuse")
	}
	if err := sup.Spawn("", "x", "s1", 0, 0, 1_000, nil); err == nil {
		t.Error("empty call id must refuse")
	}
}

func procOf(t *testing.T, tab toolproc.Table, id string) toolproc.Proc {
	t.Helper()
	for _, p := range tab.Procs {
		if p.CallID == id {
			return p
		}
	}
	t.Fatalf("proc %s not in table", id)
	return toolproc.Proc{}
}

package toolprocgate

import (
	"context"
	"testing"
	"time"

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

// TestSupervisorSettlementGrace verifies bounded settlement grace handling:
// (a) A cooperative / responsive process that exits within the settlement grace
//     window is not forcefully reaped (Reaped is false, Settled is true).
// (b) A non-responsive process that ignores cancel and stays alive past the
//     settlement grace window is forcefully reaped when grace expires.
func TestSupervisorSettlementGrace(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	t.Run("cooperative_process_settles_within_grace", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		if err := sup.Spawn("p-coop", "fetch_data", "s1", 10_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-coop", 1234); err != nil {
			t.Fatal(err)
		}

		// Within deadline: no enforcement.
		rep, err := sup.Tick(9_000)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 0 || cancelled {
			t.Fatalf("premature enforcement: actions=%v cancelled=%v", rep.Actions, cancelled)
		}

		// Past deadline (12_000): cancel lever triggered, forceful OS reap deferred.
		rep, err = sup.Tick(12_000)
		if err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must be invoked upon deadline exceeded")
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action on initial cancel request, got %+v", rep.Actions)
		}
		act1 := rep.Actions[0]
		if !act1.Cancelled || act1.Reaped || !act1.SettlementPending {
			t.Fatalf("want cancelled=true, reaped=false, pending=true; got %+v", act1)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called while grace is pending, got pids=%v", rec.pids)
		}
		if !sup.SettlementPending("p-coop") {
			t.Fatal("SettlementPending must be true")
		}

		// Responsive process answers cancel and terminates cleanly at 12_200 (within 500ms grace window ending at 12_500).
		if err := sup.Exit("p-coop", 12_200, "ok"); err != nil {
			t.Fatal(err)
		}

		// Tick at 12_300: process settled cleanly without forceful reap.
		rep, err = sup.Tick(12_300)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 settled action, got %+v", rep.Actions)
		}
		act2 := rep.Actions[0]
		if act2.Reaped || !act2.Settled {
			t.Fatalf("want Reaped=false, Settled=true; got Reaped=%v, Settled=%v", act2.Reaped, act2.Settled)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("cooperative process must never be reaped, got pids=%v", rec.pids)
		}
		if _, ok := KilledReason("p-coop"); ok {
			t.Fatal("settled process must not be entered into revocation table")
		}
		if p := procOf(t, rep.Table, "p-coop"); p.State != toolproc.StateDone {
			t.Fatalf("want StateDone, got %s", p.State)
		}
		if !sup.IsSettled("p-coop") {
			t.Fatal("IsSettled must report true")
		}
		settled, reaped, pending := sup.SettlementStatus("p-coop")
		if !settled || reaped || pending {
			t.Fatalf("SettlementStatus: want (true, false, false), got (%v, %v, %v)", settled, reaped, pending)
		}
		if act2.TerminalClassification != TerminalDeliveredAndSettled {
			t.Fatalf("want TerminalDeliveredAndSettled, got %q", act2.TerminalClassification)
		}
		if tc := sup.TerminalClassification("p-coop"); tc != TerminalDeliveredAndSettled {
			t.Fatalf("TerminalClassification: want %q, got %q", TerminalDeliveredAndSettled, tc)
		}
		if sup.SettledAtMS("p-coop") != 12_200 {
			t.Fatalf("SettledAtMS: want 12200, got %d", sup.SettledAtMS("p-coop"))
		}
		info, ok := sup.SettlementInfo("p-coop")
		if !ok || !info.Settled || info.Reaped || info.Pending || info.TerminalClassification != TerminalDeliveredAndSettled {
			t.Fatalf("SettlementInfo: unexpected info %+v", info)
		}

		// Subsequent ticks: idempotent, no further reap or action.
		rep, err = sup.Tick(14_000)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must remain uncalled on later ticks, got pids=%v", rec.pids)
		}
	})

	t.Run("cooperative_process_exits_synchronously_in_cancel", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancel := func() {
			// Process cleanly exits immediately upon receiving cancel lever
			_ = sup.Exit("p-sync", 12_050, "ok")
		}

		if err := sup.Spawn("p-sync", "fetch_data", "s1", 10_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-sync", 1235); err != nil {
			t.Fatal(err)
		}

		rep, err := sup.Tick(12_000)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action, got %+v", rep.Actions)
		}
		act := rep.Actions[0]
		if !act.Cancelled || act.Reaped || !act.Settled {
			t.Fatalf("want Cancelled=true, Reaped=false, Settled=true; got %+v", act)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called, got %v", rec.pids)
		}
	})

	t.Run("non_responsive_process_reaped_on_grace_expiry", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
			// Non-responsive: ignores cancel, does NOT exit.
		}

		if err := sup.Spawn("p-unresp", "heavy_work", "s1", 10_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-unresp", 5678); err != nil {
			t.Fatal(err)
		}

		// Past deadline (12_000): cancel lever triggered, forceful OS reap deferred.
		rep, err := sup.Tick(12_000)
		if err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must be invoked")
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action, got %+v", rep.Actions)
		}
		act1 := rep.Actions[0]
		if !act1.Cancelled || act1.Reaped || act1.Settled {
			t.Fatalf("want Cancelled=true, Reaped=false, Settled=false; got %+v", act1)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called before grace expiry, got %v", rec.pids)
		}

		// Tick within grace window (12_200 < 12_500): process still alive, still deferred.
		rep, err = sup.Tick(12_200)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called within grace window, got %v", rec.pids)
		}

		// Tick past grace window (12_600 >= 12_500): grace expired! Forceful reap executed.
		rep, err = sup.Tick(12_600)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 reap action, got %+v", rep.Actions)
		}
		actExpired := rep.Actions[0]
		if !actExpired.Reaped || actExpired.Settled || actExpired.ReapDetail != "tree terminated" {
			t.Fatalf("want Reaped=true, Settled=false, Detail='tree terminated'; got %+v", actExpired)
		}
		if len(rec.pids) != 1 || rec.pids[0] != 5678 {
			t.Fatalf("reaper must be invoked with PID 5678, got %v", rec.pids)
		}
		if r, ok := KilledReason("p-unresp"); !ok || r != toolproc.ReasonToolDeadlineExceededName {
			t.Fatalf("revocation table must cite deadline exceeded, got %q/%t", r, ok)
		}
		if p := procOf(t, rep.Table, "p-unresp"); p.State != toolproc.StateKilled {
			t.Fatalf("want StateKilled, got %s", p.State)
		}
		if actExpired.TerminalClassification != TerminalDeliveredButUnknown {
			t.Fatalf("want TerminalDeliveredButUnknown, got %q", actExpired.TerminalClassification)
		}
		if tc := sup.TerminalClassification("p-unresp"); tc != TerminalDeliveredButUnknown {
			t.Fatalf("TerminalClassification: want %q, got %q", TerminalDeliveredButUnknown, tc)
		}
		if sup.ReapedAtMS("p-unresp") != 12_600 {
			t.Fatalf("ReapedAtMS: want 12600, got %d", sup.ReapedAtMS("p-unresp"))
		}
		info, ok := sup.SettlementInfo("p-unresp")
		if !ok || info.Settled || !info.Reaped || info.Pending || info.TerminalClassification != TerminalDeliveredButUnknown {
			t.Fatalf("SettlementInfo: unexpected info %+v", info)
		}

		// Late result submitted for reaped process is quarantined through kernel.AdmitResult.
		c := &abi.ToolCall{Tool: "heavy_work", TraceID: "p-unresp"}
		r := &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("late output")}}
		v := kernel.New("").AdmitResult(context.Background(), c, r)
		if v.Kind != abi.VerdictQuarantine || v.Reason != toolproc.ReasonToolResultAfterKill {
			t.Fatalf("late result must be quarantined with TOOL_RESULT_AFTER_KILL, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
		}

		// Subsequent tick: idempotent, process already reaped.
		rep, err = sup.Tick(15_000)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.pids) != 1 {
			t.Fatalf("reaper must not be invoked again, got %v", rec.pids)
		}
	})

	t.Run("cooperative_process_settles_via_Settle_call", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		if err := sup.Spawn("p-settle", "fetch_data", "s1", 10_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-settle", 4321); err != nil {
			t.Fatal(err)
		}

		// Past deadline (12_000): cancel lever triggered, settlement pending.
		rep, err := sup.Tick(12_000)
		if err != nil {
			t.Fatal(err)
		}
		if !cancelled || len(rep.Actions) != 1 || !rep.Actions[0].SettlementPending {
			t.Fatalf("want pending settlement action, got %+v", rep.Actions)
		}

		// Process directly invokes Settle to report clean termination without OS reap.
		if err := sup.Settle("p-settle", 12_200); err != nil {
			t.Fatal(err)
		}
		if !sup.IsSettled("p-settle") {
			t.Fatal("IsSettled must be true after Settle")
		}

		// Tick at 12_300: process settled, reap skipped.
		rep, err = sup.Tick(12_300)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action, got %+v", rep.Actions)
		}
		act := rep.Actions[0]
		if act.Reaped || !act.Settled {
			t.Fatalf("want Reaped=false, Settled=true; got %+v", act)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called, got %v", rec.pids)
		}
		if p := procOf(t, rep.Table, "p-settle"); p.State != toolproc.StateDone {
			t.Fatalf("Table must show StateDone, got %s", p.State)
		}
		settled, reaped, pending := sup.SettlementStatus("p-settle")
		if !settled || reaped || pending {
			t.Fatalf("want (true, false, false), got (%v, %v, %v)", settled, reaped, pending)
		}
	})

	t.Run("explicit_cancel_with_per_process_override_and_settle", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 100 * time.Millisecond
		sup.SetProcessSettlementGrace("p-override", 800*time.Millisecond)

		if g := sup.EffectiveGrace("p-override"); g != 800*time.Millisecond {
			t.Fatalf("want effective grace 800ms, got %v", g)
		}
		if g := sup.EffectiveGrace("other"); g != 100*time.Millisecond {
			t.Fatalf("want default grace 100ms for other, got %v", g)
		}

		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		if err := sup.Spawn("p-override", "long_job", "s1", 60_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-override", 7777); err != nil {
			t.Fatal(err)
		}

		// Explicit Cancel at 5_000 with custom reason.
		if err := sup.Cancel("p-override", 5_000, "OPERATOR_ABORT"); err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must fire immediately on Cancel")
		}
		if reqAt := sup.CancelRequestedAt("p-override"); reqAt.UnixMilli() != 5_000 {
			t.Fatalf("CancelRequestedAt: want 5000, got %v", reqAt.UnixMilli())
		}
		if !sup.SettlementPending("p-override") {
			t.Fatal("SettlementPending must be true")
		}

		// Tick at 5_200: past default 100ms grace, but well within 800ms override (deadline 5_800).
		rep, err := sup.Tick(5_200)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called within overridden grace window, got %v", rec.pids)
		}
		if len(rep.Actions) != 1 || !rep.Actions[0].SettlementPending {
			t.Fatalf("want pending action on first tick, got %+v", rep.Actions)
		}

		// Process settles at 5_300 via Settle.
		if err := sup.Settle("p-override", 5_300); err != nil {
			t.Fatal(err)
		}

		// Tick at 5_400: reports clean settlement.
		rep, err = sup.Tick(5_400)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action, got %+v", rep.Actions)
		}
		act := rep.Actions[0]
		if act.Reaped || !act.Settled {
			t.Fatalf("want Reaped=false, Settled=true; got %+v", act)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called, got %v", rec.pids)
		}
		if p := procOf(t, rep.Table, "p-override"); p.State != toolproc.StateDone {
			t.Fatalf("Table must show StateDone, got %s", p.State)
		}
		settled, reaped, pending := sup.SettlementStatus("p-override")
		if !settled || reaped || pending {
			t.Fatalf("want (true, false, false), got (%v, %v, %v)", settled, reaped, pending)
		}
	})

	t.Run("explicit_cancel_non_responsive_grace_expiry_reap", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		if err := sup.Spawn("p-exp-reap", "worker", "s1", 60_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-exp-reap", 8888); err != nil {
			t.Fatal(err)
		}

		// Explicit Cancel at 5_000.
		if err := sup.Cancel("p-exp-reap", 5_000, "OPERATOR_KILL"); err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must fire")
		}

		// Tick at 5_200 (within 500ms grace window ending at 5_500): not reaped.
		rep, err := sup.Tick(5_200)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("reaper must not be called within grace window, got %v", rec.pids)
		}

		// Tick at 5_600 (past grace deadline 5_500): grace expired! Forceful reap executed.
		rep, err = sup.Tick(5_600)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action on grace expiry, got %+v", rep.Actions)
		}
		act := rep.Actions[0]
		if !act.Reaped || act.Settled || act.Reason != "OPERATOR_KILL" {
			t.Fatalf("want Reaped=true, Settled=false, Reason=OPERATOR_KILL; got %+v", act)
		}
		if len(rec.pids) != 1 || rec.pids[0] != 8888 {
			t.Fatalf("reaper must be invoked with PID 8888, got %v", rec.pids)
		}
		if r, ok := KilledReason("p-exp-reap"); !ok || r != "OPERATOR_KILL" {
			t.Fatalf("revocation table must record OPERATOR_KILL, got %q/%t", r, ok)
		}
		if p := procOf(t, rep.Table, "p-exp-reap"); p.State != toolproc.StateKilled {
			t.Fatalf("post-reap table must show StateKilled, got %s", p.State)
		}
		settled, reaped, pending := sup.SettlementStatus("p-exp-reap")
		if settled || !reaped || pending {
			t.Fatalf("SettlementStatus: want (false, true, false), got (%v, %v, %v)", settled, reaped, pending)
		}
	})

	t.Run("per_process_grace_override_zero_disables_grace", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		sup.SetProcessSettlementGrace("p-nograce", 0)

		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		if err := sup.Spawn("p-nograce", "worker", "s1", 60_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}
		if err := sup.BindPID("p-nograce", 9999); err != nil {
			t.Fatal(err)
		}

		// Cancel immediately reaps when grace is 0.
		if err := sup.Cancel("p-nograce", 5_000, "IMMEDIATE_KILL"); err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must fire")
		}
		if len(rec.pids) != 1 || rec.pids[0] != 9999 {
			t.Fatalf("reaper must fire immediately, got %v", rec.pids)
		}
		if sup.SettlementPending("p-nograce") {
			t.Fatal("SettlementPending must be false when grace is 0")
		}
		settled, reaped, pending := sup.SettlementStatus("p-nograce")
		if settled || !reaped || pending {
			t.Fatalf("SettlementStatus: want (false, true, false), got (%v, %v, %v)", settled, reaped, pending)
		}
	})

	t.Run("pre_dispatch_abort_distinct_terminal_outcome", func(t *testing.T) {
		Reset()
		t.Cleanup(Reset)

		sup := NewSupervisor(toolproc.Config{})
		sup.SettlementGrace = 500 * time.Millisecond
		rec := &recordingReaper{ok: true, detail: "tree terminated"}
		sup.SetReaper(rec.reap)

		cancelled := false
		cancel := func() {
			cancelled = true
		}

		// Spawn without binding PID (pre-dispatch).
		if err := sup.Spawn("p-predisp", "unbound_tool", "s1", 10_000, 0, 1_000, cancel); err != nil {
			t.Fatal(err)
		}

		// Cancelled before dispatch / before PID binding.
		if err := sup.Cancel("p-predisp", 2_000, "PRE_DISPATCH_ABORT"); err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Fatal("cancel lever must be invoked")
		}

		// Settles cleanly at 2_100 via Settle.
		if err := sup.Settle("p-predisp", 2_100); err != nil {
			t.Fatal(err)
		}

		rep, err := sup.Tick(2_200)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Actions) != 1 {
			t.Fatalf("want 1 action, got %+v", rep.Actions)
		}
		act := rep.Actions[0]
		if act.TerminalClassification != TerminalPreDispatchAbort {
			t.Fatalf("want TerminalPreDispatchAbort, got %q", act.TerminalClassification)
		}
		if tc := sup.TerminalClassification("p-predisp"); tc != TerminalPreDispatchAbort {
			t.Fatalf("TerminalClassification: want %q, got %q", TerminalPreDispatchAbort, tc)
		}
		if len(rec.pids) != 0 {
			t.Fatalf("unbound process must never invoke OS reaper, got %v", rec.pids)
		}
		info, ok := sup.SettlementInfo("p-predisp")
		if !ok || info.TerminalClassification != TerminalPreDispatchAbort || info.RequestedAtMS != 2_000 || info.SettledAtMS != 2_100 {
			t.Fatalf("SettlementInfo: unexpected info %+v", info)
		}
	})
}

// TestSupervisorCancelAlreadyExitedCall verifies that Cancel() on an already-exited
// call does not retroactively revoke or kill it.
func TestSupervisorCancelAlreadyExitedCall(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	sup := NewSupervisor(toolproc.Config{})
	sup.SettlementGrace = 500 * time.Millisecond
	rec := &recordingReaper{ok: true, detail: "tree terminated"}
	sup.SetReaper(rec.reap)

	cancelled := false
	cancel := func() {
		cancelled = true
	}

	// 1. Spawn call with PID binding.
	if err := sup.Spawn("p-exited", "worker", "s1", 10_000, 0, 1_000, cancel); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("p-exited", 1111); err != nil {
		t.Fatal(err)
	}

	// 2. Call exits cleanly.
	if err := sup.Exit("p-exited", 2_000, "ok"); err != nil {
		t.Fatal(err)
	}

	// 3. Cancel called on already-exited call.
	if err := sup.Cancel("p-exited", 3_000, "OPERATOR_ABORT"); err != nil {
		t.Fatal(err)
	}

	// 4. Must not create pending settlement record, trigger cancel lever, or reap.
	if sup.SettlementPending("p-exited") {
		t.Fatal("already-exited call must not enter settlement pending")
	}
	if cancelled {
		t.Fatal("cancel lever must not be invoked on already-exited call")
	}
	if len(rec.pids) != 0 {
		t.Fatalf("reaper must not be called on already-exited call, got %v", rec.pids)
	}
	if _, ok := KilledReason("p-exited"); ok {
		t.Fatal("already-exited call must not be entered into revocation table")
	}

	// 5. Tick past deadline/grace must not revoke or reap.
	rep, err := sup.Tick(12_000)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range rep.Actions {
		if a.CallID == "p-exited" && (a.Advice == toolproc.AdviceKill || a.Advice == toolproc.AdviceReap || a.Reaped) {
			t.Fatalf("already-exited call must not yield kill/reap action, got %+v", a)
		}
	}
	if len(rec.pids) != 0 {
		t.Fatalf("reaper must not be called on tick for already-exited call, got %v", rec.pids)
	}
	if _, ok := KilledReason("p-exited"); ok {
		t.Fatal("already-exited call must not be entered into revocation table after tick")
	}
	tab, err := sup.Table(12_000)
	if err != nil {
		t.Fatal(err)
	}
	p := procOf(t, tab, "p-exited")
	if p.State != toolproc.StateDone {
		t.Fatalf("want StateDone, got %s", p.State)
	}

	// 6. Also verify immediate kill path (SettlementGrace <= 0).
	if err := sup.Spawn("p-exited-nograce", "worker", "s1", 10_000, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.Exit("p-exited-nograce", 2_000, "ok"); err != nil {
		t.Fatal(err)
	}
	sup.SettlementGrace = 0
	if err := sup.Cancel("p-exited-nograce", 3_000, "IMMEDIATE_KILL"); err != nil {
		t.Fatal(err)
	}
	if _, ok := KilledReason("p-exited-nograce"); ok {
		t.Fatal("already-exited call must not be entered into revocation table even with grace=0")
	}
}

// TestSupervisorLateExitPreventsPhantomReap verifies that a late Exit()
// transitions state out of settlementPending and prevents phantom reap.
func TestSupervisorLateExitPreventsPhantomReap(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	sup := NewSupervisor(toolproc.Config{})
	sup.SettlementGrace = 500 * time.Millisecond
	rec := &recordingReaper{ok: true, detail: "tree terminated"}
	sup.SetReaper(rec.reap)

	cancelled := false
	cancel := func() {
		cancelled = true
	}

	// 1. Spawn call with deadline 10_000 and bind PID.
	if err := sup.Spawn("p-late-exit", "worker", "s1", 10_000, 0, 1_000, cancel); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("p-late-exit", 2222); err != nil {
		t.Fatal(err)
	}

	// 2. Tick at 12_000 (deadline exceeded): cancel lever fires, grace begins (deadline 12_500).
	rep, err := sup.Tick(12_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || !rep.Actions[0].SettlementPending {
		t.Fatalf("want 1 pending settlement action, got %+v", rep.Actions)
	}
	if !cancelled {
		t.Fatal("cancel lever must fire at deadline")
	}
	if !sup.SettlementPending("p-late-exit") {
		t.Fatal("call must be pending settlement")
	}
	if len(rec.pids) != 0 {
		t.Fatalf("reaper must not be called during grace, got %v", rec.pids)
	}

	// 3. Late Exit at 12_600 (nowMS 12_600 > DeadlineMS 12_500).
	if err := sup.Exit("p-late-exit", 12_600, "ok"); err != nil {
		t.Fatal(err)
	}

	// 4. Must transition out of settlementPending into settled.
	if sup.SettlementPending("p-late-exit") {
		t.Fatal("SettlementPending must be false after Exit()")
	}
	if !sup.IsSettled("p-late-exit") {
		t.Fatal("IsSettled must be true after Exit()")
	}
	settled, reaped, pending := sup.SettlementStatus("p-late-exit")
	if !settled || reaped || pending {
		t.Fatalf("SettlementStatus: want (true, false, false), got (%v, %v, %v)", settled, reaped, pending)
	}

	// 5. Tick at 12_700: must NOT forcefully reap the exited process!
	rep, err = sup.Tick(12_700)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.pids) != 0 {
		t.Fatalf("reaper must NOT be invoked on late-exited process (phantom reap!), got %v", rec.pids)
	}
	if _, ok := KilledReason("p-late-exit"); ok {
		t.Fatal("late-exited process must NOT be revoked")
	}
	tab, err := sup.Table(12_700)
	if err != nil {
		t.Fatal(err)
	}
	p := procOf(t, tab, "p-late-exit")
	if p.State != toolproc.StateDone {
		t.Fatalf("want StateDone, got %s", p.State)
	}
}

// TestSupervisorSettleTransitionsToDoneAndNoLoop verifies that Settle()
// transitions the process to StateDone and does not loop forever on subsequent ticks.
func TestSupervisorSettleTransitionsToDoneAndNoLoop(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	sup := NewSupervisor(toolproc.Config{})
	sup.SettlementGrace = 500 * time.Millisecond

	if err := sup.Spawn("p-settle-loop", "worker", "s1", 10_000, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}

	// Tick past deadline: cancel triggered, settlement pending.
	rep, err := sup.Tick(12_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || !rep.Actions[0].SettlementPending {
		t.Fatalf("want pending settlement action, got %+v", rep.Actions)
	}

	// Process settles at 12_200 via Settle().
	if err := sup.Settle("p-settle-loop", 12_200); err != nil {
		t.Fatal(err)
	}

	// Authority table must immediately reflect StateDone.
	tab, err := sup.Table(12_200)
	if err != nil {
		t.Fatal(err)
	}
	p := procOf(t, tab, "p-settle-loop")
	if p.State != toolproc.StateDone {
		t.Fatalf("authoritative table must transition to StateDone, got %s", p.State)
	}

	// Tick at 12_300: first tick after settle reports Settled=true.
	rep, err = sup.Tick(12_300)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || !rep.Actions[0].Settled {
		t.Fatalf("want 1 settled action, got %+v", rep.Actions)
	}

	// Tick at 13_000: must NOT loop or emit duplicate settled actions!
	rep, err = sup.Tick(13_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 {
		t.Fatalf("subsequent tick must emit 0 actions (no infinite loop), got %+v", rep.Actions)
	}

	// Tick at 14_000: still clean.
	rep, err = sup.Tick(14_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 {
		t.Fatalf("subsequent tick must emit 0 actions, got %+v", rep.Actions)
	}
}

// TestSupervisorProcessGraceOverride verifies per-process grace override behavior.
func TestSupervisorProcessGraceOverride(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	sup := NewSupervisor(toolproc.Config{})
	sup.SettlementGrace = 200 * time.Millisecond
	rec := &recordingReaper{ok: true, detail: "tree terminated"}
	sup.SetReaper(rec.reap)

	sup.SetProcessSettlementGrace("p-longer", 1000*time.Millisecond)
	sup.SetProcessSettlementGrace("p-zero", 0)

	if g := sup.EffectiveGrace("p-longer"); g != 1000*time.Millisecond {
		t.Fatalf("want 1000ms, got %v", g)
	}
	if g := sup.EffectiveGrace("p-zero"); g != 0 {
		t.Fatalf("want 0, got %v", g)
	}
	if g := sup.EffectiveGrace("p-default"); g != 200*time.Millisecond {
		t.Fatalf("want 200ms, got %v", g)
	}

	if err := sup.Spawn("p-longer", "task", "s1", 60_000, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("p-longer", 101); err != nil {
		t.Fatal(err)
	}
	if err := sup.Spawn("p-zero", "task", "s1", 60_000, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("p-zero", 102); err != nil {
		t.Fatal(err)
	}
	if err := sup.Spawn("p-default", "task", "s1", 60_000, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("p-default", 103); err != nil {
		t.Fatal(err)
	}

	// 1. Cancel p-zero: immediately reaped because grace is 0.
	if err := sup.Cancel("p-zero", 10_000, "ABORT"); err != nil {
		t.Fatal(err)
	}
	if len(rec.pids) != 1 || rec.pids[0] != 102 {
		t.Fatalf("p-zero must be reaped immediately, got %v", rec.pids)
	}

	// 2. Cancel p-default: grace is 200ms (deadline 10_200).
	if err := sup.Cancel("p-default", 10_000, "ABORT"); err != nil {
		t.Fatal(err)
	}
	// 3. Cancel p-longer: grace is 1000ms (deadline 11_000).
	if err := sup.Cancel("p-longer", 10_000, "ABORT"); err != nil {
		t.Fatal(err)
	}

	// Tick at 10_300:
	// - p-default grace expired at 10_200 -> reaped!
	// - p-longer grace expires at 11_000 -> still pending, not reaped!
	rep, err := sup.Tick(10_300)
	if err != nil {
		t.Fatal(err)
	}
	var longerAct, defaultAct *TickAction
	for i := range rep.Actions {
		switch rep.Actions[i].CallID {
		case "p-longer":
			longerAct = &rep.Actions[i]
		case "p-default":
			defaultAct = &rep.Actions[i]
		}
	}
	if longerAct == nil || !longerAct.SettlementPending {
		t.Fatalf("want p-longer pending action at 10_300, got %+v", rep.Actions)
	}
	if defaultAct == nil || !defaultAct.Reaped {
		t.Fatalf("want p-default reaped action at 10_300, got %+v", rep.Actions)
	}
	if len(rec.pids) != 2 || rec.pids[1] != 103 {
		t.Fatalf("p-default must be reaped at 10_300, got %v", rec.pids)
	}
	if !sup.SettlementPending("p-longer") {
		t.Fatal("p-longer must still be pending settlement")
	}

	// p-longer exits cleanly before 11_000 deadline.
	if err := sup.Exit("p-longer", 10_500, "ok"); err != nil {
		t.Fatal(err)
	}

	// Tick at 11_500: p-longer settled cleanly, never reaped.
	rep, err = sup.Tick(11_500)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].CallID != "p-longer" || !rep.Actions[0].Settled {
		t.Fatalf("want 1 settled action for p-longer, got %+v", rep.Actions)
	}
	if len(rec.pids) != 2 {
		t.Fatalf("p-longer must never be reaped, got %v", rec.pids)
	}
	if !sup.IsSettled("p-longer") {
		t.Fatal("p-longer must be settled")
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

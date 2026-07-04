package toolprocgate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// TestMonitorKilled_LateOutputQuarantined is the monitor seam's full causal
// chain, in-process and clock-free: arm an event-stream monitor with a declared
// cadence -> it goes silent past cadence -> Tick KILLS it (a monitor stall is a
// kill, not a probe) citing TOOL_HEARTBEAT_STALLED and journals the revocation
// -> a late completion from the dead monitor is QUARANTINED through the real
// kernel result floor. Silence produced a verdict, and the straggler race is
// closed for free by the existing revocation table.
func TestMonitorKilled_LateOutputQuarantined(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})

	// A progress-only monitor is refused before anything is journaled.
	if err := sup.ArmMonitor(toolproc.MonitorSpec{
		CallID: "m-happy", Filter: "elapsed_steps=", HeartbeatEveryMS: 5_000, AtMS: 1_000,
	}, nil); err == nil {
		t.Fatal("progress-only monitor must be refused at arm time")
	}

	ctx, cancel := context.WithCancel(context.Background())
	if err := sup.ArmMonitor(toolproc.MonitorSpec{
		CallID: "m-late", Tool: "monitor:deploy", Session: "s1",
		Filter: "Ready in|Traceback|Error|FAILED", HeartbeatEveryMS: 5_000, AtMS: 1_000,
	}, cancel); err != nil {
		t.Fatal(err)
	}

	// Within cadence: no action, the monitor is live.
	rep, err := sup.Tick(3_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 0 || ctx.Err() != nil {
		t.Fatalf("premature enforcement: actions=%v ctxErr=%v", rep.Actions, ctx.Err())
	}

	// Silent past cadence: the monitor is KILLED (not merely probed), the cancel
	// lever fires, and the journal records TOOL_HEARTBEAT_STALLED as the reason.
	rep, err = sup.Tick(60_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].Advice != toolproc.AdviceKill || !rep.Actions[0].Cancelled {
		t.Fatalf("want one cancelled kill action, got %+v", rep.Actions)
	}
	if rep.Actions[0].Reason != toolproc.ReasonToolHeartbeatStalledName {
		t.Fatalf("kill must cite the stall verdict, got %q", rep.Actions[0].Reason)
	}
	if ctx.Err() == nil {
		t.Fatal("the monitor's in-flight context must be cancelled")
	}
	if r, ok := KilledReason("m-late"); !ok || r != toolproc.ReasonToolHeartbeatStalledName {
		t.Fatalf("revocation table: got %q/%t", r, ok)
	}
	if p := procOf(t, rep.Table, "m-late"); p.State != toolproc.StateKilled {
		t.Fatalf("post-enforcement table must show KILLED, got %s", p.State)
	}

	// The late completion, admitted through the REAL kernel chain, is quarantined.
	c := &abi.ToolCall{Tool: "monitor:deploy", TraceID: "m-late"}
	r := &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("late monitor bytes")}}
	v := kernel.New("").AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictQuarantine || v.Reason != toolproc.ReasonToolResultAfterKill {
		t.Fatalf("late completion: want Quarantine/TOOL_RESULT_AFTER_KILL, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// Idempotent: a later Tick takes no further destructive action on the dead monitor.
	rep, err = sup.Tick(70_000)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range rep.Actions {
		if a.CallID == "m-late" && a.Advice != toolproc.AdviceQuarantineResult {
			t.Fatalf("killed monitor must yield no further destructive action, got %+v", a)
		}
	}
}

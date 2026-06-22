package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestRunEmbedsPerCallTrace is the witness that agent-report.json carries a
// structured, per-call decision trace for BOTH arms — so a bad run is debuggable
// from the artifact alone, with no opt-in --log side file. It pins the kernel
// actions the offline A/B deterministically produces (an in-syscall grammar
// TRANSFORM, a vDSO dedup hit, a quarantine) as per-call rows, the deciding rung
// (by) being captured, and the reason+by invariant on any deny.
func TestRunEmbedsPerCallTrace(t *testing.T) {
	res, _, err := Run(context.Background(), NewMockPlanner("trace"), DefaultTask, 12)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res.Calls) == 0 {
		t.Fatal("RunResult.Calls is empty — per-call trace not embedded in the artifact")
	}

	var sawFak, sawBaseline, sawTransform, sawVDSO, sawKernelAction, sawRungBy bool
	for _, c := range res.Calls {
		switch c.Arm {
		case "fak":
			sawFak = true
		case "baseline":
			sawBaseline = true
			if c.Verdict != "naive-exec" {
				t.Errorf("baseline row should be naive-exec, got %q (%+v)", c.Verdict, c)
			}
		default:
			t.Errorf("call has unexpected arm %q", c.Arm)
		}
		if c.Turn <= 0 || c.Tool == "" || c.Verdict == "" {
			t.Errorf("call row missing core fields: %+v", c)
		}
		if c.Arm == "fak" {
			if c.By != "" {
				sawRungBy = true // the deciding rung is captured (monitor/vdso/grammar/...)
			}
			if c.Verdict == "TRANSFORM" {
				sawTransform = true
			}
			if c.By == "vdso" {
				sawVDSO = true
			}
			if strings.Contains(c.Note, "QUARANTINED") || c.Verdict == "DENY" {
				sawKernelAction = true
			}
			// Invariant: a deny must name WHY (reason) and WHO (by) — the two fields
			// the old text-only trace dropped.
			if c.Verdict == "DENY" && (c.Reason == "" || c.By == "") {
				t.Errorf("fak DENY row missing reason/by: %+v", c)
			}
		}
	}
	if !sawFak || !sawBaseline {
		t.Errorf("expected both arms in the trace; sawFak=%v sawBaseline=%v", sawFak, sawBaseline)
	}
	if !sawRungBy {
		t.Error("expected at least one fak row to record the deciding rung (by)")
	}
	if !sawTransform {
		t.Error("expected the in-syscall grammar TRANSFORM to appear as a per-call row")
	}
	if !sawVDSO {
		t.Error("expected the vDSO dedup hit (by=vdso) to appear as a per-call row")
	}
	if !sawKernelAction {
		t.Error("expected the poison handling (quarantine or deny) to be visible in the trace")
	}

	// The trace must serialize cleanly and not leak unbounded args.
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"calls"`) {
		t.Error("serialized report has no calls array")
	}
	for _, c := range res.Calls {
		if len(c.Args) > 200 {
			t.Errorf("call args preview is unbounded (%d bytes): %q", len(c.Args), c.Args)
		}
	}
}

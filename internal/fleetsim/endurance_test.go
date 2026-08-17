package fleetsim

import (
	"encoding/json"
	"testing"
)

func TestReplayEnduranceCloses200WithoutPrematureDoneOrPeerTheft(t *testing.T) {
	cfg := EnduranceConfig{
		Issues: 200, Workers: 32, ClosePerCycle: 8, MaxCycles: 80,
		RefusalCycles:   map[int]bool{2: true, 3: true, 11: true},
		OwnedWIPCycles:  map[int]bool{7: true},
		PeerWIPCycles:   map[int]bool{15: true, 16: true, 17: true, 18: true, 19: true},
		AbandonedCycles: map[int]bool{24: true},
	}
	rep := ReplayEndurance(cfg)
	if rep.ClosedIssues != 200 || !rep.EventuallyDrain {
		t.Fatalf("closed=%d drain=%v, want 200 and eventual drain", rep.ClosedIssues, rep.EventuallyDrain)
	}
	if rep.PrematureDone || rep.TouchedPeerWIP {
		t.Fatalf("unsafe report: premature_done=%v touched_peer=%v", rep.PrematureDone, rep.TouchedPeerWIP)
	}
	if rep.MaxNoProgress != 5 {
		t.Fatalf("max no-progress = %d, want peer wait to reach capped decision rung", rep.MaxNoProgress)
	}
	seenDecision, seenReset := false, false
	for i, cycle := range rep.Cycles {
		if cycle.Stage == "operator-decision" {
			seenDecision = true
		}
		if i > 0 && rep.Cycles[i-1].NoProgress > 0 && cycle.Closed > 0 && cycle.NoProgress == 0 {
			seenReset = true
		}
		if cycle.OpenAfter > 0 && !cycle.KeepGoing {
			t.Fatalf("cycle %d declared done with %d open issues", cycle.Cycle, cycle.OpenAfter)
		}
	}
	if !seenDecision || !seenReset {
		t.Fatalf("decision=%v reset=%v, want capped escalation and progress reset", seenDecision, seenReset)
	}
	b, err := json.Marshal(rep)
	if err != nil || !json.Valid(b) || len(b) == 0 {
		t.Fatalf("machine-readable report invalid: err=%v bytes=%q", err, b)
	}
	t.Logf("ENDURANCE_REPORT %s", b)
}

func TestReplayEnduranceResidualWorkKeepsGoingAfterIssueDrain(t *testing.T) {
	rep := ReplayEndurance(EnduranceConfig{Issues: 1, ClosePerCycle: 1, OwnedWIPCycles: map[int]bool{2: true}, MaxCycles: 4})
	if len(rep.Cycles) != 3 || !rep.Cycles[1].KeepGoing || rep.Cycles[1].Residual != "owned-reconcile" {
		t.Fatalf("cycles=%+v, want owned residual to delay done", rep.Cycles)
	}
	if !rep.EventuallyDrain {
		t.Fatal("expected drain after residual cleanup")
	}
}

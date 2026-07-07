package gateway

import "testing"

// TestAnnotateToolLivelockFusesRepeatedAdmittedCall proves the structural backstop:
// an identical ADMITTED call repeated past the fuse count stops being admitted. The
// advisory-only note (attach an envelope, keep admitting) let an unresponsive loop
// burn tokens forever; the fuse converts the call into a retryable LIVELOCK_FUSE
// refusal and reports its id so the caller drops it from the kept slice.
func TestAnnotateToolLivelockFusesRepeatedAdmittedCall(t *testing.T) {
	s := &Server{}
	mk := func() []ToolAdjudication {
		return []ToolAdjudication{{
			ToolCallID: "tc-loop",
			Tool:       "Bash",
			ArgsDigest: "sha256:abc",
			Admitted:   true,
			Verdict:    WireVerdict{Kind: "ALLOW"},
		}}
	}

	// Advisory threshold is 3, default fuse factor 2 -> fuse arms at repeat 6.
	for i := 1; i <= 5; i++ {
		adjs := mk()
		fused := s.annotateToolLivelock("sess-fuse", adjs)
		if len(fused) != 0 {
			t.Fatalf("repeat %d fused before the fuse count: %v", i, fused)
		}
		if !adjs[0].Admitted {
			t.Fatalf("repeat %d de-admitted before the fuse count", i)
		}
		if i >= 3 && (adjs[0].Livelock == nil || adjs[0].Livelock.Fuse) {
			t.Fatalf("repeat %d: want advisory-without-fuse envelope, got %+v", i, adjs[0].Livelock)
		}
	}

	// Repeat 6: the fuse arms and the call is converted to a refusal.
	adjs := mk()
	fused := s.annotateToolLivelock("sess-fuse", adjs)
	if _, ok := fused["tc-loop"]; !ok {
		t.Fatalf("repeat 6 did not report the fused tool-call id: %v", fused)
	}
	if adjs[0].Admitted {
		t.Fatal("repeat 6 must de-admit the fused call")
	}
	if adjs[0].Verdict.Kind != "DENY" || adjs[0].Verdict.Reason != ReasonLivelockFuse {
		t.Fatalf("fused verdict = %+v, want DENY/%s", adjs[0].Verdict, ReasonLivelockFuse)
	}
	// The refusal must be retryable per-tool feedback, never a deny-all session stop.
	if !toolRejectionIsRetryableFeedback(adjs[0].Verdict) {
		t.Fatalf("fused refusal is not retryable feedback: %+v", adjs[0].Verdict)
	}
	if adjs[0].Livelock == nil || !adjs[0].Livelock.Fuse {
		t.Fatalf("fused adjudication should carry a fuse=armed envelope, got %+v", adjs[0].Livelock)
	}
}

// TestAnnotateToolLivelockFuseDropsKeptCall proves the whole served path: a fused
// call is removed from kept (never reaches the wire) and counted in dropped.
func TestAnnotateToolLivelockFuseDropsKeptCall(t *testing.T) {
	// The fused verdict must be classed as retryable feedback so a fully-fused turn
	// becomes tool-feedback (keep going, change approach), not a deny-all stop.
	v := WireVerdict{Kind: "DENY", Reason: ReasonLivelockFuse, Disposition: "RETRYABLE"}
	adjs := []ToolAdjudication{{Tool: "Bash", Admitted: false, Verdict: v}}
	if got := adjudicationOutcomeForTurn(adjs, 0, 0); got != adjudicationOutcomeToolFeedback {
		t.Fatalf("a fully-fused turn = %v, want tool-feedback (not a deny-all session stop)", got)
	}
}

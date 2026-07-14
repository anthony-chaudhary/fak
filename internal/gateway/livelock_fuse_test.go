package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

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

// TestAnnotateToolLivelockEscalatesToTerminalDenyAll proves the terminal rung: an
// identical admitted call that survives the retryable fuse and keeps repeating past the
// abort count (threshold*3 = 9) is stamped TERMINAL, so the all-refused turn becomes a
// deny-all session stop instead of the retryable feedback that let #2704 spin forever.
func TestAnnotateToolLivelockEscalatesToTerminalDenyAll(t *testing.T) {
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
	// Repeats 1..8: the fuse (6) arms but the disposition stays RETRYABLE feedback.
	var adjs []ToolAdjudication
	for i := 1; i <= 8; i++ {
		adjs = mk()
		s.annotateToolLivelock("sess-abort", adjs)
		if i >= 6 && !toolRejectionIsRetryableFeedback(adjs[0].Verdict) {
			t.Fatalf("repeat %d: fused refusal must stay retryable before the abort count, got %+v", i, adjs[0].Verdict)
		}
	}
	// Repeat 9: the terminal rung arms — the refusal is now non-retryable, so the
	// whole-turn outcome is a deny-all stop.
	adjs = mk()
	s.annotateToolLivelock("sess-abort", adjs)
	if adjs[0].Verdict.Kind != "DENY" || adjs[0].Verdict.Reason != ReasonLivelockFuse {
		t.Fatalf("repeat 9 verdict = %+v, want DENY/%s", adjs[0].Verdict, ReasonLivelockFuse)
	}
	if toolRejectionIsRetryableFeedback(adjs[0].Verdict) {
		t.Fatalf("repeat 9 refusal must NOT be retryable (terminal escalation), got %+v", adjs[0].Verdict)
	}
	if adjs[0].Livelock == nil || !adjs[0].Livelock.Escalate {
		t.Fatalf("repeat 9 adjudication should carry an escalate=true envelope, got %+v", adjs[0].Livelock)
	}
	if got := adjudicationOutcomeForTurn(adjs, 0, 0); got != adjudicationOutcomeDenyAll {
		t.Fatalf("a terminally-escalated turn = %v, want deny-all session stop", got)
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

func TestRepeatedTerminalSecretDenialSaturatesAndStaysBroken(t *testing.T) {
	s := &Server{}
	var last []ToolAdjudication
	for i := 0; i < 110; i++ {
		last = []ToolAdjudication{{ToolCallID: "tc-secret", Tool: "PowerShell", ArgsDigest: "sha256:44136fa355b3", Admitted: false,
			Verdict: WireVerdict{Kind: "DENY", Reason: "SECRET_EXFIL", By: "policy", Disposition: "TERMINAL"}}}
		s.annotateToolLivelock("interactive-36304", last)
	}
	if last[0].Livelock == nil || !last[0].Livelock.Escalate {
		t.Fatalf("110 identical terminal denials must carry a loop-break envelope: %+v", last[0])
	}
	want := guardrsi.DefaultLivelockThreshold*guardrsi.DefaultLivelockAbortFactor + 1
	if last[0].Livelock.RepeatCount != want {
		t.Fatalf("repeat counter=%d, want saturated abort threshold %d", last[0].Livelock.RepeatCount, want)
	}
	if last[0].Verdict.By != "livelock-abort" || last[0].Verdict.Disposition != "TERMINAL" {
		t.Fatalf("loop break verdict=%+v", last[0].Verdict)
	}
	if got := adjudicationOutcomeForTurn(last, 0, 0); got != adjudicationOutcomeDenyAll {
		t.Fatalf("terminal repeated denial outcome=%v, want deny-all stop", got)
	}
}

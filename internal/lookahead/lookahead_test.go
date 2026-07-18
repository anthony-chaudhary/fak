package lookahead

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// longTranscript is any transcript over the minLessonTranscriptChars decline floor, so a test
// exercises the GATE rather than the short-transcript decline.
var longTranscript = strings.Repeat("turn ", 60)

// seam builds a DistillFunc that always proposes the given claim+kind (ok=true).
func seam(claim string, kind LessonKind) DistillFunc {
	return func(string) (Proposal, bool) { return Proposal{Claim: claim, Kind: kind}, true }
}

// TestDistillLessonWitnessGate is the golden table: a proposal's asserted kind is admitted only
// at the rung its evidence earned. W3 may assert FACT or RISK; W2 may assert only RISK (a FACT is
// refused LESSON_OVERCLAIMS); W1/W0 may assert nothing.
func TestDistillLessonWitnessGate(t *testing.T) {
	cases := []struct {
		name        string
		rung        trajctl.WitnessRung
		proposed    LessonKind
		wantOK      bool
		wantRefused bool
		wantKind    LessonKind
	}{
		{"W3 fact -> witnessed fact", trajctl.W3, KindFact, true, false, KindFact},
		{"W3 risk -> risk", trajctl.W3, KindRisk, true, false, KindRisk},
		{"W3 blank kind -> risk floor", trajctl.W3, "", true, false, KindRisk},
		{"W2 risk -> risk", trajctl.W2, KindRisk, true, false, KindRisk},
		{"W2 blank kind -> risk floor", trajctl.W2, "", true, false, KindRisk},
		{"W2 fact -> OVERCLAIMS", trajctl.W2, KindFact, false, true, ""},
		{"W1 risk -> OVERCLAIMS", trajctl.W1, KindRisk, false, true, ""},
		{"W1 fact -> OVERCLAIMS", trajctl.W1, KindFact, false, true, ""},
		{"W0 risk -> OVERCLAIMS", trajctl.W0, KindRisk, false, true, ""},
		{"unknown rung -> OVERCLAIMS", trajctl.WitnessRung("W9"), KindRisk, false, true, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ev := RolloutEvidence{ForkSessionID: "fork-1", BaseSHA: "base", Turns: 3, Rung: tc.rung}
			lesson, out := DistillLesson(ev, longTranscript, "expire-sha", seam("the claim", tc.proposed))
			if out.OK != tc.wantOK {
				t.Fatalf("OK = %v, want %v (out=%+v)", out.OK, tc.wantOK, out)
			}
			if out.Refused != tc.wantRefused {
				t.Fatalf("Refused = %v, want %v (out=%+v)", out.Refused, tc.wantRefused, out)
			}
			if tc.wantRefused && out.Reason != ReasonLessonOverclaims {
				t.Fatalf("refusal reason = %q, want %q", out.Reason, ReasonLessonOverclaims)
			}
			if tc.wantOK {
				if lesson.Kind != tc.wantKind {
					t.Fatalf("lesson kind = %q, want %q", lesson.Kind, tc.wantKind)
				}
				if lesson.Rung != tc.rung {
					t.Fatalf("lesson rung = %q, want %q", lesson.Rung, tc.rung)
				}
				if lesson.Evidence.ForkSessionID != "fork-1" || lesson.ExpiresSHA != "expire-sha" {
					t.Fatalf("lesson did not carry its evidence/expiry: %+v", lesson)
				}
			}
		})
	}
}

// TestDistillLessonDeclinesGracefully proves every non-gate fall-through yields a DECLINE (never
// a refusal, never a poisoned seed): a nil seam, a too-short transcript, and a seam that errors.
func TestDistillLessonDeclinesGracefully(t *testing.T) {
	ev := RolloutEvidence{Rung: trajctl.W3}
	// nil seam
	if _, out := DistillLesson(ev, longTranscript, "sha", nil); !out.Declined || out.Reason != DeclineNoSeam {
		t.Fatalf("nil seam => %+v, want Declined/%s", out, DeclineNoSeam)
	}
	// short transcript
	if _, out := DistillLesson(ev, "tiny", "sha", seam("c", KindFact)); !out.Declined || out.Reason != DeclineTranscriptTiny {
		t.Fatalf("short transcript => %+v, want Declined/%s", out, DeclineTranscriptTiny)
	}
	// seam declines
	declining := DistillFunc(func(string) (Proposal, bool) { return Proposal{}, false })
	if _, out := DistillLesson(ev, longTranscript, "sha", declining); !out.Declined || out.Reason != DeclineModelError {
		t.Fatalf("declining seam => %+v, want Declined/%s", out, DeclineModelError)
	}
	// A decline is never a refusal.
	if _, out := DistillLesson(ev, "tiny", "sha", nil); out.Refused || out.OK {
		t.Fatalf("decline must not set Refused/OK: %+v", out)
	}
}

// TestLessonRenderShowsRung pins that the rung is visible in the rendered string so a consumer
// cannot read a W2 risk as a witnessed fact.
func TestLessonRenderShowsRung(t *testing.T) {
	fact := Lesson{Claim: "the head is warm", Kind: KindFact, Rung: trajctl.W3}
	if got := fact.Render(); got != "Witnessed (W3): the head is warm" {
		t.Fatalf("fact render = %q", got)
	}
	risk := Lesson{Claim: "may stall", Kind: KindRisk, Rung: trajctl.W2}
	if got := risk.Render(); got != "Risk flag (W2): may stall" {
		t.Fatalf("risk render = %q", got)
	}
}

// TestLessonStale pins the staleness horizon: an un-pinned lesson never goes stale; a pinned one
// goes stale exactly when the caller says trunk moved past its ExpiresSHA.
func TestLessonStale(t *testing.T) {
	unpinned := Lesson{ExpiresSHA: ""}
	if unpinned.Stale(true) {
		t.Fatal("an un-pinned lesson (empty ExpiresSHA) must never be stale")
	}
	pinned := Lesson{ExpiresSHA: "base-sha"}
	if pinned.Stale(false) {
		t.Fatal("pinned lesson: trunk has NOT moved past -> fresh")
	}
	if !pinned.Stale(true) {
		t.Fatal("pinned lesson: trunk moved past ExpiresSHA -> stale")
	}
}

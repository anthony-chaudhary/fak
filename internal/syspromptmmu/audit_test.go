package syspromptmmu

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// TestAuditOK asserts a freshly built fak base context audits clean: present, not
// diverged, the re-derived digest equals the plan digest, and every resident segment
// witness matches.
func TestAuditOK(t *testing.T) {
	plan := BaseContextPlan()
	body := bodyWith(t, BuildSystemValue(plan, []cachemeta.PromptSegment{overlaySeg("a card")}), nil)

	a := AuditBaseContext(body)
	if !a.Present || a.Diverged || a.Status != AuditOK {
		t.Fatalf("clean body: present=%v diverged=%v status=%q, want present/ok", a.Present, a.Diverged, a.Status)
	}
	if a.GotDigest != a.ExpectDigest {
		t.Errorf("digest mismatch on a clean spine:\n got=%s\n exp=%s", a.GotDigest, a.ExpectDigest)
	}
	if len(a.Segments) != len(plan) {
		t.Fatalf("audited %d segments, want %d", len(a.Segments), len(plan))
	}
	for i, s := range a.Segments {
		if !s.Match {
			t.Errorf("segment %d should match", i)
		}
	}
	if a.BreakIdx != len(plan)-1 {
		t.Errorf("breakIdx = %d, want %d", a.BreakIdx, len(plan)-1)
	}
}

// TestAuditDivergedAlarm asserts a tampered resident block fires the loud divergence
// alarm with the culprit segment flagged and the digest changed.
func TestAuditDivergedAlarm(t *testing.T) {
	plan := BaseContextPlan()
	// Build a fak-SHAPED body (breakpoint on last resident) but with spine block 1
	// tampered — the accidental head mutation.
	mutated := make([]cachemeta.PromptSegment, len(plan))
	copy(mutated, plan)
	mutated[1].Content = append(append([]byte(nil), plan[1].Content...), " DRIFT"...)
	body := bodyWith(t, BuildSystemValue(mutated, nil), nil)

	a := AuditBaseContext(body)
	if !a.Present {
		t.Fatal("a fak-shaped body must be Present even when diverged")
	}
	if !a.Diverged || a.Status != AuditDiverged {
		t.Fatalf("expected the divergence alarm, got diverged=%v status=%q", a.Diverged, a.Status)
	}
	if a.GotDigest == a.ExpectDigest {
		t.Error("digest must differ on a diverged spine")
	}
	if a.Segments[1].Match {
		t.Error("segment 1 (tampered) must be flagged as a mismatch")
	}
	if !a.Segments[0].Match {
		t.Error("segment 0 (untouched) should still match")
	}
}

// TestAuditAbsentHarnessBody asserts a harness-authored body (no fak base context) is a
// neutral AuditAbsent, NOT a divergence alarm — the WITNESSED/PLANNED fence.
func TestAuditAbsentHarnessBody(t *testing.T) {
	cases := map[string][]byte{
		"bare-string-system": bodyWith(t, []byte(`"a harness-authored system prompt"`), nil),
		"no-breakpoint":      bodyWith(t, []byte(`[{"type":"text","text":"harness rule"}]`), nil),
		"no-system":          []byte(`{"model":"x","messages":[]}`),
		"empty":              nil,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			a := AuditBaseContext(raw)
			if a.Present {
				t.Error("a non-fak body must not be Present")
			}
			if a.Diverged {
				t.Error("a non-fak body must NOT raise the divergence alarm")
			}
			if a.Status != AuditAbsent {
				t.Errorf("status = %q, want %q", a.Status, AuditAbsent)
			}
		})
	}
}

// TestAuditMisplacedBreakpointAbsent asserts a body whose breakpoint is not on the last
// resident block is AuditAbsent (not fak-shaped), never a false alarm or false OK.
func TestAuditMisplacedBreakpointAbsent(t *testing.T) {
	plan := BaseContextPlan()
	var blocks []json.RawMessage
	for i, seg := range plan {
		blocks = append(blocks, marshalBlock(seg.Content, i == len(plan)-1))
	}
	blocks = append(blocks, marshalBlock([]byte("overlay with a stray breakpoint"), true))
	sys, err := json.Marshal(blocks)
	if err != nil {
		t.Fatal(err)
	}
	a := AuditBaseContext(bodyWith(t, sys, nil))
	if a.Present || a.Status != AuditAbsent {
		t.Fatalf("misplaced breakpoint: present=%v status=%q, want absent", a.Present, a.Status)
	}
}

// TestAuditSurvivesOverlaySwap ties Rung 6 to Rung 2/3: swapping the overlay via
// SpliceSystemOverlay must NOT trip the spine alarm — the audit still reads AuditOK.
func TestAuditSurvivesOverlaySwap(t *testing.T) {
	plan := BaseContextPlan()
	body := bodyWith(t, BuildSystemValue(plan, []cachemeta.PromptSegment{overlaySeg("old")}), nil)

	res := SpliceSystemOverlay(body, plan, []cachemeta.PromptSegment{overlaySeg("new-1"), overlaySeg("new-2")}, decodeOK)
	if !res.Changed {
		t.Fatalf("expected a splice, got identity (%s)", res.SkipReason)
	}
	a := AuditBaseContext(res.Body)
	if !a.Present || a.Diverged || a.Status != AuditOK {
		t.Fatalf("audit after an overlay swap: present=%v diverged=%v status=%q, want ok", a.Present, a.Diverged, a.Status)
	}
	if a.GotDigest != a.ExpectDigest {
		t.Error("an overlay swap must not change the realized spine digest")
	}
}

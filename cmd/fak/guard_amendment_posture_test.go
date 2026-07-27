package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestFormatAmendmentPosture proves the exit summary's AMENDMENT POSTURE section
// reports, straight off the real policy.PolicyKnobRegistry: the FROZEN / RATCHET
// / GATED_WIDEN counts, and that the SELF_AMENDABLE (agent-writable) frontier is
// empty. Driven off the live registry (no fixture) so the section can never
// drift from the classification the conformance tests in internal/policy enforce.
func TestFormatAmendmentPosture(t *testing.T) {
	out := formatAmendmentPosture()

	// The section header follows the shared exit-summary grammar.
	if !strings.Contains(out, "── guard · amendment posture ") {
		t.Fatalf("missing amendment-posture section header:\n%s", out)
	}

	frozen := len(policy.KnobsByClass(policy.AmendFrozen))
	ratchet := len(policy.KnobsByClass(policy.AmendRatchet))
	gated := len(policy.KnobsByClass(policy.AmendGatedWiden))
	if frozen == 0 || ratchet == 0 || gated == 0 {
		t.Fatalf("registry sanity: want nonzero FROZEN/RATCHET/GATED_WIDEN, got %d/%d/%d",
			frozen, ratchet, gated)
	}
	want := fmt.Sprintf("%d FROZEN, %d RATCHET (tighten-only), %d GATED_WIDEN (operator-gated)",
		frozen, ratchet, gated)
	if !strings.Contains(out, want) {
		t.Fatalf("posture row missing per-class counts %q:\n%s", want, out)
	}

	// The load-bearing claim: the agent-writable frontier is empty today.
	if got := policy.KnobsByClass(policy.AmendSelfAmendable); len(got) != 0 {
		t.Fatalf("registry declares %d SELF_AMENDABLE knob(s); this test (and the summary's empty-frontier row) assume 0: %+v", len(got), got)
	}
	if !strings.Contains(out, "0 — empty frontier") {
		t.Fatalf("missing empty self-amendable frontier row:\n%s", out)
	}
	if !strings.Contains(out, "could not widen anything on its own") {
		t.Fatalf("missing the operator-legibility note (agent could not widen anything on its own):\n%s", out)
	}
	if strings.Contains(out, "⚠ self-amendable") {
		t.Fatalf("non-empty-frontier warning must not print while the registry frontier is empty:\n%s", out)
	}
}

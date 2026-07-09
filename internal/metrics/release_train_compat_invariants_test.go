package metrics

import "testing"

// Invariant tests for the release-train compatibility gate (issue #1658,
// gen/second-next). These pin contracts the package doc *asserts* but the
// scenario tests never made checkable on their own: totality/order of the
// finding list, and the fail-closed defaults that guard a future edit which
// adds a check without a rule. They are the compatibility-test artifact the
// generation frame names, and they harden the gate before a release-decide
// caller ever indexes its findings by position.

// TestCheckReleaseTrainIsTotalAndOrdered pins the "pure and total" contract:
// for ANY candidate — including the fail-closed unknown-generation path —
// CheckReleaseTrain returns exactly one finding per check, in CompatChecks
// order, and every finding carries a verdict and a reason. A refactor that
// reorders, drops, or short-circuits a finding trips this before a caller that
// reads findings[i] by position silently reads the wrong check.
func TestCheckReleaseTrainIsTotalAndOrdered(t *testing.T) {
	candidates := []ReleaseTrainCandidate{
		{}, // empty generation "" -> unknown -> fail-closed path
		{Generation: "now", ReleaseNoteSection: "shipped"},
		{Generation: "next", ExperimentalFlag: "FAK_X", ReleaseNoteSection: "next"},
		{Generation: "second-next", ExperimentalFlag: "FAK_Y", ReleaseNoteSection: "research"},
		{Generation: "future", DefaultExposed: true, BreaksShippedSurface: true, SupersedesShipped: true, ReleaseNoteSection: "shipped"},
		{Generation: "someday"}, // non-empty but outside the closed vocabulary
	}
	for _, c := range candidates {
		r := CheckReleaseTrain(c)
		if len(r.Findings) != len(CompatChecks) {
			t.Fatalf("candidate %+v: got %d findings, want one per check (%d)", c, len(r.Findings), len(CompatChecks))
		}
		for i, check := range CompatChecks {
			f := r.Findings[i]
			if f.Check != check {
				t.Fatalf("candidate %+v: findings[%d].Check = %q, want %q (order must match CompatChecks)", c, i, f.Check, check)
			}
			if f.Verdict == "" {
				t.Fatalf("candidate %+v: findings[%d] (%q) has an empty verdict; every check must yield one", c, i, check)
			}
			if f.Reason == "" {
				t.Fatalf("candidate %+v: findings[%d] (%q) has an empty reason", c, i, check)
			}
		}
	}
}

// TestEvaluateUnknownCheckFailsClosed exercises the defensive default branches
// that CheckReleaseTrain cannot reach (it only iterates the closed CompatChecks
// set). A CompatCheck value with no rule must BLOCK — "fail closed", never a
// silent pass — and its human label must fall back to the raw key rather than
// render blank. This is the guard for a future edit that adds a CompatCheck
// constant but forgets to give it a rule in evaluate.
func TestEvaluateUnknownCheckFailsClosed(t *testing.T) {
	const bogus = CompatCheck("no-such-check")

	f := evaluate(bogus, ReleaseTrainCandidate{Generation: "now"})
	if f.Verdict != CompatBlock {
		t.Fatalf("a check with no rule must fail closed with %q, got %q", CompatBlock, f.Verdict)
	}
	if f.Check != bogus {
		t.Fatalf("finding should carry the offending check %q, got %q", bogus, f.Check)
	}
	if f.Reason == "" {
		t.Fatalf("a fail-closed finding must explain itself, got an empty reason")
	}
	if got := bogus.Label(); got != string(bogus) {
		t.Fatalf("Label() for an unknown check should echo the raw key %q, got %q", string(bogus), got)
	}
}

// TestQuoteRendersEmptyVisibly pins the reason-string helper's empty case: an
// absent value renders as "" so a "names no gate" / "unplaced" reason reads as a
// visible empty string, not a gap the reader has to notice. The unknown-flag and
// unplaced-note refusals depend on this being legible.
func TestQuoteRendersEmptyVisibly(t *testing.T) {
	if got := quote(""); got != `""` {
		t.Fatalf("quote(\"\") = %q, want %q", got, `""`)
	}
	if got := quote("FAK_X"); got != `"FAK_X"` {
		t.Fatalf("quote(%q) = %q, want %q", "FAK_X", got, `"FAK_X"`)
	}
}

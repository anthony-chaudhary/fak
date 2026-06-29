package dojocal

import (
	"strings"
	"testing"
)

// registryFixture mirrors the real internal/dojo/claims.go literal format exactly,
// including the two traps that make a naive value-replace wrong: the same number
// recurs across cells (1.0 appears three times here) and the claimed value is also
// quoted in the basis prose ("~85% ... (share = 0.85)"). A correct rewrite is
// anchored on the (lever, metric) key and touches ONLY the constructor's first
// numeric argument.
const registryFixture = `var Registry = ClaimRegistry{
	{"resume-posture", "posture_accuracy"}: claim(1.0,
		"the resume projection's per-boundary cold/warm posture call assumed correct"),
	{"resume-posture", "cold_write_share"}: claim(0.85,
		"the projection prices ~85% of the resident at the cold-write premium (share = 0.85)"),
	{"resume-posture", "cross_session_warm_hit_rate"}: floor(0.0, false,
		"~0% of large first-turn resumes hit a still-warm cross-session prefix by default"),
	{"vcache-warmth", "false_warm_rate"}: floor(0.0, true,
		"the warmth belief never predicts warm on a call the provider bills cache_read=0"),
	{"vcache-warmth", "warm_recall"}: claim(1.0,
		"the warmth belief calls warm every read the provider bills cache_read>0"),
}`

func TestRewriteClaim_AnchorsToCellNotProse(t *testing.T) {
	// Re-point cold_write_share 0.85 -> 0.9. The basis prose ("~85%", "share = 0.85")
	// must survive byte-for-byte; only the claim(...) numeric argument changes.
	out, old, err := RewriteClaim([]byte(registryFixture), "resume-posture", "cold_write_share", 0.9)
	if err != nil {
		t.Fatalf("RewriteClaim: %v", err)
	}
	if old != 0.85 {
		t.Fatalf("old value = %v, want 0.85", old)
	}
	s := string(out)
	if !strings.Contains(s, `"cold_write_share"}: claim(0.9,`) {
		t.Errorf("claim literal not rewritten to 0.9:\n%s", s)
	}
	// The prose number is NOT a claim argument and must be untouched.
	if !strings.Contains(s, "~85% of the resident") || !strings.Contains(s, "(share = 0.85)") {
		t.Errorf("basis prose was corrupted by the rewrite:\n%s", s)
	}
	if strings.Contains(s, "claim(0.85,") {
		t.Errorf("the old claim literal 0.85 still present after rewrite:\n%s", s)
	}
}

func TestRewriteClaim_DisambiguatesSameValueCells(t *testing.T) {
	// Three cells claim 1.0 (posture_accuracy, warm_recall, and the floors at 0.0).
	// Rewriting posture_accuracy must leave warm_recall's 1.0 intact.
	out, _, err := RewriteClaim([]byte(registryFixture), "resume-posture", "posture_accuracy", 0.965)
	if err != nil {
		t.Fatalf("RewriteClaim: %v", err)
	}
	s := string(out)
	if !strings.Contains(s, `"posture_accuracy"}: claim(0.965,`) {
		t.Errorf("posture_accuracy not rewritten:\n%s", s)
	}
	if !strings.Contains(s, `"warm_recall"}: claim(1.0,`) {
		t.Errorf("warm_recall's 1.0 was clobbered by the posture_accuracy rewrite:\n%s", s)
	}
}

func TestRewriteClaim_FloorFormPreservesTrailingArgs(t *testing.T) {
	// The floor(<float>, <bool>, ...) form has a trailing bool the rewrite must not
	// touch. (The proposer never RECALIBRATEs a floor, but the rewriter must still
	// handle the constructor shape correctly if ever handed one.)
	out, old, err := RewriteClaim([]byte(registryFixture), "vcache-warmth", "false_warm_rate", 0.01)
	if err != nil {
		t.Fatalf("RewriteClaim: %v", err)
	}
	if old != 0.0 {
		t.Fatalf("old = %v, want 0.0", old)
	}
	if !strings.Contains(string(out), `"false_warm_rate"}: floor(0.01, true,`) {
		t.Errorf("floor float not rewritten or trailing bool clobbered:\n%s", string(out))
	}
}

func TestRewriteClaim_UnknownCellErrors(t *testing.T) {
	if _, _, err := RewriteClaim([]byte(registryFixture), "no-such-lever", "no-metric", 0.5); err == nil {
		t.Fatal("expected an error for an unregistered cell, got nil")
	}
}

func TestRewriteClaim_NoOpErrors(t *testing.T) {
	// Rewriting to the same value is not a real recalibration — it must fail closed,
	// the text twin of treeChangedOnly's "changed nothing" refusal.
	_, old, err := RewriteClaim([]byte(registryFixture), "resume-posture", "cold_write_share", 0.85)
	if err == nil {
		t.Fatal("expected a no-op error rewriting 0.85 -> 0.85, got nil")
	}
	if old != 0.85 {
		t.Fatalf("no-op error should still report old=0.85, got %v", old)
	}
}

func TestReadClaim(t *testing.T) {
	v, err := ReadClaim([]byte(registryFixture), "resume-posture", "cold_write_share")
	if err != nil {
		t.Fatalf("ReadClaim: %v", err)
	}
	if v != 0.85 {
		t.Fatalf("ReadClaim = %v, want 0.85", v)
	}
	// After a rewrite, ReadClaim reflects the new value (round-trip).
	out, _, err := RewriteClaim([]byte(registryFixture), "resume-posture", "cold_write_share", 0.9)
	if err != nil {
		t.Fatalf("RewriteClaim: %v", err)
	}
	if v, err := ReadClaim(out, "resume-posture", "cold_write_share"); err != nil || v != 0.9 {
		t.Fatalf("ReadClaim after rewrite = %v, %v; want 0.9, nil", v, err)
	}
}

func TestClaimChangeLine_SingleLineDiff(t *testing.T) {
	lineNo, before, after, err := ClaimChangeLine([]byte(registryFixture), "resume-posture", "cold_write_share", 0.9)
	if err != nil {
		t.Fatalf("ClaimChangeLine: %v", err)
	}
	if lineNo <= 0 {
		t.Fatalf("lineNo = %d, want a positive 1-based line", lineNo)
	}
	if !strings.Contains(before, "claim(0.85,") || !strings.Contains(after, "claim(0.9,") {
		t.Fatalf("diff lines wrong:\n- %s\n+ %s", before, after)
	}
	// Exactly the claim argument differs between the two lines — the prose tail is
	// identical, proving the change is surgical.
	if strings.TrimSpace(strings.Replace(before, "0.85", "0.9", 1)) != strings.TrimSpace(after) {
		t.Fatalf("the only difference between the lines must be 0.85->0.9:\n- %s\n+ %s", before, after)
	}
}

func TestFormatClaim(t *testing.T) {
	cases := map[float64]string{
		1.0:   "1.0",
		0.0:   "0.0",
		0.85:  "0.85",
		0.893: "0.893",
		0.965: "0.965",
	}
	for in, want := range cases {
		if got := formatClaim(in); got != want {
			t.Errorf("formatClaim(%v) = %q, want %q", in, got, want)
		}
	}
}

package headlesslint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestScanLeftoversBothArms is #3670's done-condition, verbatim: a run that ends
// with "two more things worth doing" prose and filed ZERO issues is refused; the
// SAME summary once those follow-ups were filed as open gh issues passes clean.
func TestScanLeftoversBothArms(t *testing.T) {
	summary := "Shipped the retry fix, tests pass, committed abc1234, pushed.\n" +
		"There are two more things worth doing: exponential backoff and a docs pass."

	// Arm 1 — narrated leftovers, zero issues filed -> refused.
	unfiled := ScanLeftovers(summary, 0, false)
	if unfiled.Verdict != LeftoversUnfiled || !unfiled.Refused() {
		t.Fatalf("arm1 (narrated + 0 filed): want %s/refused, got %s/refused=%v %+v",
			LeftoversUnfiled, unfiled.Verdict, unfiled.Refused(), unfiled.Hits)
	}
	if unfiled.Narrated == 0 {
		t.Fatalf("arm1: expected at least one narrated leftover, got 0")
	}
	if unfiled.Resolve == "" {
		t.Errorf("arm1: a refused report should carry the remediation string")
	}

	// Arm 2 — the same summary, but the two follow-ups were filed as gh issues.
	filed := ScanLeftovers(summary, 2, false)
	if filed.Verdict != LeftoversClean || filed.Refused() {
		t.Fatalf("arm2 (narrated + 2 filed): want %s/not-refused, got %s/refused=%v",
			LeftoversClean, filed.Verdict, filed.Refused())
	}
}

// TestScanLeftoversOperatorEscape: the "genuinely nothing left" escape forces clean
// even when leftovers are narrated and nothing was filed.
func TestScanLeftoversOperatorEscape(t *testing.T) {
	rep := ScanLeftovers("A couple more things are still out of scope and left to do.", 0, true)
	if rep.Verdict != LeftoversClean || rep.Refused() {
		t.Fatalf("operator escape: want %s/not-refused, got %s/refused=%v", LeftoversClean, rep.Verdict, rep.Refused())
	}
	if !rep.Overridden {
		t.Errorf("operator escape: report should record Overridden=true")
	}
}

// TestScanLeftoversCleanSummary: a summary that only reports completed work carries
// no leftover narration, so it is clean even with zero issues filed.
func TestScanLeftoversCleanSummary(t *testing.T) {
	rep := ScanLeftovers("Implemented the parser, committed abc123, tests pass, pushed.", 0, false)
	if rep.Verdict != LeftoversClean || rep.Narrated != 0 {
		t.Fatalf("clean summary: want clean/0 narrated, got %s/%d %+v", rep.Verdict, rep.Narrated, rep.Hits)
	}
}

// TestScanLeftoversTicketedLineNotNarration: a leftover line that itself cites a
// filed ticket is honest scoping, not bare narration, so it does not flag even with
// the issues-filed count at zero (the per-line ticket cross-check).
func TestScanLeftoversTicketedLineNotNarration(t *testing.T) {
	rep := ScanLeftovers("Out of scope for this change: exponential backoff, filed #4001.", 0, false)
	if rep.Verdict != LeftoversClean {
		t.Fatalf("ticketed leftover: want clean, got %s %+v", rep.Verdict, rep.Hits)
	}
}

// TestLeftoversDoctrineBindsAgentsMd couples code to doctrine: the fold quotes the
// AGENTS.md spine-first rule verbatim, and this asserts AGENTS.md still carries that
// exact line. If the doctrine text moves, this reds — forcing the constant and the
// rule to stay in lockstep rather than drifting silently.
func TestLeftoversDoctrineBindsAgentsMd(t *testing.T) {
	path := filepath.Join("..", "..", "AGENTS.md")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(b), Doctrine) {
		t.Fatalf("AGENTS.md must carry the doctrine line %q that ScanLeftovers binds to (code↔doctrine coupling broke)", Doctrine)
	}
}

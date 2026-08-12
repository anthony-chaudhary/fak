package hooks

import (
	"strings"
	"testing"
)

// repairorder_test.go — the hand-off a gate run gives a committing agent must lead with the work
// that actually refuses the commit (#5972). The scenario the axis names is the one where
// registry order and impact order DISAGREE: PUBLIC_LEAK is registered FIRST but softened to
// warn, DUPLICATION is registered LAST but hardened to block. Registry order alone puts the
// non-blocking item at the head of the fix list.

// gateIndex returns a gate's position in the real pre-commit registry, so the witness below is
// anchored to the shipped order rather than to an assumption about it.
func gateIndex(t *testing.T, name string) int {
	t.Helper()
	for i, g := range PreCommitGates() {
		if g.Name == name {
			return i
		}
	}
	t.Fatalf("gate %q not registered in PreCommitGates()", name)
	return -1
}

// collectAsPreCommitDoes mirrors the cmd/fak pre-commit loop: walk the registry IN ORDER, mark
// each gate's findings with the disposition its resolved mode gives them, append flat, then
// order the whole list for repair. modes maps gate name -> resolved mode; a gate absent from
// findingsByGate contributed nothing this run.
func collectAsPreCommitDoes(modes map[string]string, findingsByGate map[string][]Finding) []Finding {
	var all []Finding
	for _, g := range PreCommitGates() {
		fs := findingsByGate[g.Name]
		if len(fs) == 0 {
			continue
		}
		all = append(all, MarkDisposition(fs, modes[g.Name])...)
	}
	OrderForRepair(all)
	return all
}

func TestOrderForRepairLeadsWithBlockingWhenRegistryOrderDisagrees(t *testing.T) {
	if gateIndex(t, "PUBLIC_LEAK") >= gateIndex(t, "DUPLICATION") {
		t.Fatalf("premise: PUBLIC_LEAK must be registered before DUPLICATION for this witness to bite")
	}
	leak := []Finding{{Gate: "PUBLIC_LEAK", File: "docs/leak.md", Line: 2, Detail: "redact the customer name"}}
	dup := []Finding{{Gate: "DUPLICATION", File: "internal/x/new.go", Line: 9, Detail: "copied block", Severity: 91}}

	got := collectAsPreCommitDoes(
		map[string]string{"PUBLIC_LEAK": "warn", "DUPLICATION": "block"},
		map[string][]Finding{"PUBLIC_LEAK": leak, "DUPLICATION": dup},
	)
	if len(got) != 2 {
		t.Fatalf("findings = %#v, want two", got)
	}
	if got[0].Gate != "DUPLICATION" || got[0].Advisory {
		t.Fatalf("findings[0] = %#v, want the binding DUPLICATION finding", got[0])
	}
	if got[1].Gate != "PUBLIC_LEAK" || !got[1].Advisory {
		t.Fatalf("findings[1] = %#v, want the advisory PUBLIC_LEAK finding", got[1])
	}
	// The gate's own graded magnitude rides through untouched: marking a disposition is not a
	// re-grade, and #5972 explicitly leaves Severity out of the rank.
	if got[0].Severity != 91 {
		t.Fatalf("severity = %d, want the gate's 91 preserved", got[0].Severity)
	}
	// MarkDisposition copies: the gate's own slice must not have been rewritten under it.
	if dup[0].Advisory || leak[0].Advisory {
		t.Fatalf("gate-owned findings were mutated: dup=%#v leak=%#v", dup[0], leak[0])
	}
}

func TestOrderForRepairKeepsRegistryOrderInsideEachDisposition(t *testing.T) {
	// Two binding and two advisory findings, interleaved on input. The partition must not
	// disturb the relative order the registry produced within either half.
	findings := []Finding{
		{Gate: "PUBLIC_LEAK", File: "a.md", Advisory: true},
		{Gate: "SECRET_SHAPE", File: "b.go"},
		{Gate: "PRIOR_ART", File: "c.md", Advisory: true},
		{Gate: "DUPLICATION", File: "d.go"},
	}
	OrderForRepair(findings)
	got := []string{findings[0].Gate, findings[1].Gate, findings[2].Gate, findings[3].Gate}
	// Binding half keeps SECRET_SHAPE before DUPLICATION; advisory half keeps PUBLIC_LEAK
	// before PRIOR_ART.
	want := []string{"SECRET_SHAPE", "DUPLICATION", "PUBLIC_LEAK", "PRIOR_ART"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("order = %v, want %v", got, want)
	}
}

func TestOrderForRepairIsIdempotentAndSafeOnEmpty(t *testing.T) {
	OrderForRepair(nil)
	OrderForRepair([]Finding{})
	findings := []Finding{
		{Gate: "PRIOR_ART", Advisory: true},
		{Gate: "SECRET_SHAPE"},
	}
	OrderForRepair(findings)
	first := findings[0].Gate
	OrderForRepair(findings)
	if findings[0].Gate != first || findings[0].Gate != "SECRET_SHAPE" {
		t.Fatalf("re-ordering changed the list: %#v", findings)
	}
}

func TestMarkDispositionUsesModeNotRegistryPosition(t *testing.T) {
	in := []Finding{{Gate: "PRIOR_ART", File: "x.md"}}
	if out := MarkDisposition(in, "block"); out[0].Advisory {
		t.Fatalf("block mode must mark binding, got %#v", out[0])
	}
	if out := MarkDisposition(in, "warn"); !out[0].Advisory {
		t.Fatalf("warn mode must mark advisory, got %#v", out[0])
	}
	// An env var arrives with whatever whitespace the shell left on it.
	if out := MarkDisposition(in, " block "); out[0].Advisory {
		t.Fatalf("padded block mode must mark binding, got %#v", out[0])
	}
	// An unrecognized mode degrades to ADVISORY. The unsafe direction would be to promote an
	// unknown mode to binding and refuse a commit no gate actually blocked.
	if out := MarkDisposition(in, "loud"); !out[0].Advisory {
		t.Fatalf("unknown mode must degrade to advisory, got %#v", out[0])
	}
	if got := MarkDisposition(nil, "block"); got != nil {
		t.Fatalf("MarkDisposition(nil) = %#v, want nil", got)
	}
	if !ModeIsBinding(BindingMode) || ModeIsBinding("warn") || ModeIsBinding("off") {
		t.Fatalf("ModeIsBinding disagrees with the gate modes")
	}
}

func TestRepairSummaryStatesDispositionAndLeadsWithBinding(t *testing.T) {
	if RepairSummary(nil) != "" {
		t.Fatalf("empty run must print nothing, got %q", RepairSummary(nil))
	}
	findings := []Finding{
		{Gate: "PUBLIC_LEAK", File: "docs/leak.md", Line: 2, Detail: "redact", Advisory: true},
		{Gate: "DUPLICATION", File: "internal/x/new.go", Line: 9, Detail: "copied block"},
	}
	out := RepairSummary(findings)
	lines := strings.Split(out, "\n")
	if len(lines) != 3 {
		t.Fatalf("summary = %q, want a header plus two items", out)
	}
	if !strings.Contains(lines[0], "1 binding of 2") {
		t.Fatalf("header = %q, want the binding count", lines[0])
	}
	if !strings.Contains(lines[1], "[binding]") || !strings.Contains(lines[1], "DUPLICATION") ||
		!strings.Contains(lines[1], "internal/x/new.go:9") {
		t.Fatalf("first item = %q, want the binding DUPLICATION finding with its file:line", lines[1])
	}
	if !strings.Contains(lines[2], "[advisory]") || !strings.Contains(lines[2], "PUBLIC_LEAK") {
		t.Fatalf("second item = %q, want the advisory PUBLIC_LEAK finding", lines[2])
	}
	// Rendering must not reorder the caller's slice — the JSON payload and the printed block are
	// built from the same list, and a renderer with a side effect on it is a trap.
	if findings[0].Gate != "PUBLIC_LEAK" {
		t.Fatalf("RepairSummary mutated its input: %#v", findings)
	}
}

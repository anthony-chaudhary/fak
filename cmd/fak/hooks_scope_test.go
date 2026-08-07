package main

// hooks_scope_test.go — the Done condition of #5603, stated as executable claims.
//
// The issue asks for exactly two things to be provable: that a clean staged-only run and a clean
// whole-tree run emit DIFFERENT scope strings, and that a run with a gate softened to `warn` is
// distinguishable from one where every gate ran in `block`. Both are claims about what one report
// says in isolation — a reader holding a single line or a single payload, with no second run to
// compare it against — so every test here reads the rendered output rather than the struct that
// produced it.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestCleanStagedAndCleanTreeRunsEmitDifferentScopeStrings is the first half of the Done
// condition. Both halves of the gate system print the word "clean" over registries that share
// gate names (UNTIERED_LEAF / TIER_DECLARED, GOFMT / gofmt-check), so a scope string that did not
// differ would be worse than none: it would assert sameness where the populations genuinely
// differ, and on a shared trunk the gap between them is where a peer's unstaged drift lives.
func TestCleanStagedAndCleanTreeRunsEmitDifferentScopeStrings(t *testing.T) {
	staged := runScope{Population: scopePopulationStaged}.note()
	tree := runScope{Population: scopePopulationTree}.note()

	if staged == tree {
		t.Fatalf("staged and whole-tree runs emit the SAME scope string %q; a reader holding one\n"+
			"report cannot tell which population it was clean over", staged)
	}
	// Difference alone is not enough — two arbitrary distinct strings would pass that and teach a
	// reader nothing. Each must name the population it EXCLUDES, which is the fact a clean verdict
	// silently omits.
	if !strings.Contains(staged, "unstaged") {
		t.Errorf("staged scope %q never says unstaged edits went unjudged — the exclusion is the\n"+
			"whole reason the staged verdict is narrower than it reads", staged)
	}
	if !strings.Contains(staged, "untracked") {
		t.Errorf("staged scope %q never says untracked paths went unjudged", staged)
	}
	if !strings.Contains(tree, "untracked") {
		t.Errorf("tree scope %q never says untracked files went unjudged; the whole-tree sweep is\n"+
			"not the superset a reader would assume", tree)
	}
	// The tree note must not simply be the staged note with a word swapped: the sets differ in
	// BOTH directions, and a template would hide that a whole-tree sweep also misses something.
	if !strings.Contains(tree, "no commit has staged") {
		t.Errorf("tree scope %q never says it covers edits no commit staged — the direction in\n"+
			"which the tree sweep is WIDER than the staged one", tree)
	}
}

// TestSoftenedRunIsDistinguishableFromAllBlockRun is the second half of the Done condition. A gate
// in `warn` still runs, still judges real candidates, and still appears in the per-gate ledger
// #5602 added — it is invisible in every number the report already prints, and the only thing it
// has lost is the ability to refuse. That is precisely the state an operator needs named.
func TestSoftenedRunIsDistinguishableFromAllBlockRun(t *testing.T) {
	allBlock := runScope{Population: scopePopulationStaged}
	softened := runScope{
		Population: scopePopulationStaged,
		Narrowing: gateNarrowing{
			Advisory:   []string{"PUBLIC_LEAK"},
			ByOperator: []string{"PUBLIC_LEAK"},
		},
	}

	if allBlock.note() == softened.note() {
		t.Fatalf("a run with PUBLIC_LEAK softened to warn reads identically to one where every gate\n"+
			"could refuse: %q", allBlock.note())
	}
	if strings.Contains(allBlock.note(), "NARROWED") {
		t.Errorf("a full-strength run claims it was narrowed: %q", allBlock.note())
	}
	for _, want := range []string{"NARROWED", "PUBLIC_LEAK", "advisory-only"} {
		if !strings.Contains(softened.note(), want) {
			t.Errorf("softened run's scope missing %q\ngot: %s", want, softened.note())
		}
	}
}

// TestOperatorSoftenedIsDistinguishableFromShippedAdvisory guards the distinction that makes the
// previous test worth anything. Several gates ship advisory BY DESIGN (GOFMT, PRIOR_ART), so
// "some gate is advisory" is the normal posture and cannot by itself mean somebody weakened this
// run. Collapsing the two would report fak's shipped defaults as a hollowed-out run — an alarm
// that fires on every commit is one nobody reads — and, worse, would let a genuinely softened run
// hide inside that noise.
func TestOperatorSoftenedIsDistinguishableFromShippedAdvisory(t *testing.T) {
	shipped := runScope{
		Population: scopePopulationStaged,
		Narrowing:  gateNarrowing{Advisory: []string{"GOFMT"}}, // advisory by compiled default
	}
	quietened := runScope{
		Population: scopePopulationStaged,
		Narrowing: gateNarrowing{
			Advisory:   []string{"GOFMT"},
			ByOperator: []string{"GOFMT"}, // FLEET_GOFMT_GUARD=warn set this run
		},
	}

	if shipped.note() == quietened.note() {
		t.Fatalf("a gate advisory by design and one an operator quietened this run read the same:\n%s",
			shipped.note())
	}
	if strings.Contains(shipped.note(), "moved off the compiled default") {
		t.Errorf("GOFMT ships advisory; the report blames an operator who changed nothing:\n%s", shipped.note())
	}
	if !strings.Contains(quietened.note(), "moved off the compiled default") {
		t.Errorf("an operator override went unnamed:\n%s", quietened.note())
	}
}

// TestGateNotRunIsNotReportedAsSkipped keeps #5603 from eating #5299. `skipped` means a checker
// BROKE and the run degraded; a gate the operator turned off is intent. Both narrow the verdict,
// and both must be visible, but a report that files them under one heading either cries
// degradation over a deliberate choice or hides a broken checker behind one.
func TestGateNotRunIsNotReportedAsSkipped(t *testing.T) {
	s := runScope{
		Population: scopePopulationStaged,
		Narrowing:  gateNarrowing{NotRun: []string{"DUPLICATION (off)"}, ByOperator: []string{"DUPLICATION"}},
	}
	note := s.note()
	if strings.Contains(note, "skipped") || strings.Contains(note, "DEGRADED") {
		t.Errorf("an operator-disabled gate is reported in #5299's degraded vocabulary:\n%s", note)
	}
	if !strings.Contains(note, "not run") || !strings.Contains(note, "DUPLICATION") {
		t.Errorf("the disabled gate is not named at all:\n%s", note)
	}
}

// TestScopeClauseIsSilentWhenNothingWasNarrowed keeps the line from becoming wallpaper. A clause
// that appears on every single run reporting zero is one a reader learns to skip — and the runs
// where it is the entire story are exactly the ones they would then skip it on.
func TestScopeClauseIsSilentWhenNothingWasNarrowed(t *testing.T) {
	note := runScope{Population: scopePopulationStaged}.note()
	if strings.Contains(note, "NARROWED") || strings.Contains(note, "0 not run") {
		t.Errorf("full-strength run still prints a narrowing clause:\n%s", note)
	}
	if !strings.Contains(note, "STAGED-ONLY") {
		t.Errorf("population dropped when there was no narrowing to report:\n%s", note)
	}
}

// TestScopeNamesCapsTheListButNeverTheCount covers `fak hygiene --gates INDEX_SYNC`, which
// deselects three dozen gates. The list is truncated for readability; the COUNT is not, because a
// truncated count would be the same understatement this epic exists to remove.
func TestScopeNamesCapsTheListButNeverTheCount(t *testing.T) {
	names := []string{"A_GATE", "B_GATE", "C_GATE", "D_GATE", "E_GATE", "F_GATE", "G_GATE"}
	got := scopeNames("not run", names)

	if !strings.HasPrefix(got, "7 not run") {
		t.Errorf("count was truncated along with the list: %q", got)
	}
	if !strings.Contains(got, "+3 more") {
		t.Errorf("list was not capped: %q", got)
	}
	if strings.Contains(got, "G_GATE") {
		t.Errorf("cap did not apply: %q", got)
	}
}

// TestCleanRunSummaryCarriesScope proves the wiring, not just the renderer: the pre-commit clean
// line is where a human meets this, and a scope vocabulary no summary calls is decoration.
func TestCleanRunSummaryCarriesScope(t *testing.T) {
	n := func(i int) *int { return &i }
	reports := []gateReport{{Gate: "GOFMT", Candidates: n(3), Unit: "staged .go file(s)"}}

	plain := cleanRunSummary(reports, 3, runScope{Population: scopePopulationStaged})
	if !strings.Contains(plain, "STAGED-ONLY") {
		t.Errorf("clean pre-commit summary never names its population:\n%s", plain)
	}
	// The #5602 content must survive intact beside it — this leaf adds, it does not replace.
	if !strings.Contains(plain, "GOFMT 3") {
		t.Errorf("scope clause displaced the candidate denominator:\n%s", plain)
	}

	narrowed := cleanRunSummary(reports, 3, runScope{
		Population: scopePopulationStaged,
		Narrowing:  gateNarrowing{NotRun: []string{"SECRET_SHAPE (escaped)"}, ByOperator: []string{"SECRET_SHAPE"}},
	})
	if plain == narrowed {
		t.Fatalf("a commit that escaped SECRET_SHAPE prints the same clean line as one that ran it:\n%s", plain)
	}
	if !strings.Contains(narrowed, "SECRET_SHAPE") {
		t.Errorf("escaped gate unnamed on the clean line:\n%s", narrowed)
	}
}

// TestBothJSONPayloadsCarryTheirOwnPopulation is the machine-readable form of the first Done
// claim. Under --json stderr is silent, so the payload is the ONLY channel; the two commands emit
// deliberately identical shapes, which is what makes the differing `scope` value load-bearing
// rather than cosmetic.
func TestBothJSONPayloadsCarryTheirOwnPopulation(t *testing.T) {
	staged := decodeFindingsJSON(t, nil, nil, nil)

	var out, errb bytes.Buffer
	emitHygieneJSON(&out, &errb, nil, nil, nil, runScope{Population: scopePopulationTree})
	if errb.Len() > 0 {
		t.Fatalf("emitHygieneJSON wrote to stderr: %s", errb.String())
	}
	var tree map[string]any
	if err := json.Unmarshal(out.Bytes(), &tree); err != nil {
		t.Fatalf("hygiene payload is not valid JSON: %v\n%s", err, out.String())
	}

	if staged["scope"] != scopePopulationStaged {
		t.Errorf("pre-commit scope = %#v, want %q", staged["scope"], scopePopulationStaged)
	}
	if tree["scope"] != scopePopulationTree {
		t.Errorf("hygiene scope = %#v, want %q", tree["scope"], scopePopulationTree)
	}
	if staged["scope"] == tree["scope"] {
		t.Fatalf("both commands report the same scope %#v", staged["scope"])
	}

	// Every narrowing key present with an empty list on an unnarrowed run, in BOTH payloads: a
	// consumer must never have to decide whether an absent key means "none" or "this build does
	// not report it" — the same rule #5299's skipped_gates already follows.
	for label, payload := range map[string]map[string]any{"pre-commit": staged, "hygiene": tree} {
		nar, ok := payload["scope_narrowing"].(map[string]any)
		if !ok {
			t.Fatalf("%s: scope_narrowing = %#v, want an object", label, payload["scope_narrowing"])
		}
		for _, k := range []string{"gates_not_run", "gates_advisory", "gates_operator_changed"} {
			arr, isArr := nar[k].([]any)
			if !isArr {
				t.Errorf("%s: scope_narrowing.%s = %#v, want an empty array not null/absent", label, k, nar[k])
				continue
			}
			if len(arr) != 0 {
				t.Errorf("%s: scope_narrowing.%s = %v on an unnarrowed run", label, k, arr)
			}
		}
		if nar["narrowed"] != false {
			t.Errorf("%s: scope_narrowing.narrowed = %#v on an unnarrowed run, want false", label, nar["narrowed"])
		}
	}
}

// TestScopeNarrowingPayloadReportsSoftenedGates is the second Done claim on the wire. A dashboard
// consuming --json sees no stderr at all, so if this key did not carry the softening, a fleet
// running with SECURITY gates quietened would look green in aggregate — the exact class of silent
// green this epic was opened for.
func TestScopeNarrowingPayloadReportsSoftenedGates(t *testing.T) {
	s := runScope{
		Population: scopePopulationStaged,
		Narrowing: gateNarrowing{
			NotRun:     []string{"DUPLICATION (off)"},
			Advisory:   []string{"PUBLIC_LEAK"},
			ByOperator: []string{"DUPLICATION", "PUBLIC_LEAK"},
		},
	}
	nar := s.narrowingPayload()

	if nar["narrowed"] != true {
		t.Errorf("narrowed = %#v on a run with two gates weakened, want true", nar["narrowed"])
	}
	if got := nar["gates_operator_changed"].([]string); len(got) != 2 {
		t.Errorf("gates_operator_changed = %v, want both weakened gates", got)
	}
	// not-run and advisory must stay in SEPARATE keys: one gate delivered no verdict at all, the
	// other delivered one that could not refuse, and a consumer that cannot tell them apart cannot
	// tell how much refusal power the run actually had.
	if got := nar["gates_not_run"].([]string); len(got) != 1 || !strings.Contains(got[0], "DUPLICATION") {
		t.Errorf("gates_not_run = %v, want just DUPLICATION", got)
	}
	if got := nar["gates_advisory"].([]string); len(got) != 1 || got[0] != "PUBLIC_LEAK" {
		t.Errorf("gates_advisory = %v, want just PUBLIC_LEAK", got)
	}
}

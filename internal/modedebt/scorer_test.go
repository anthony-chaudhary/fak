package modedebt

// scorer_test.go carries the #4415 witnesses. The two the issue names
// (TestModeDebtFlagsUnliftedDial / TestModeDebtPassesLiftedRegime) pin the
// doctrine at the two ends of the vocabulary: an implicit dial is ranked debt, a
// legitimately-lifted dial and a correctly harness-held safety dial are not.
// TestModeDebtGradeInvariants is the exhaustive backstop scorer.go's header cites --
// it enumerates every criteria combination so the HARD => un-lifted invariant that
// makes dispatch.go's SelectHardUnlifted a safe filter cannot rot silently.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestModeDebtFlagsUnliftedDial is the first witness: an un-lifted, floor-wideable,
// un-journaled dial is flagged as debt and ranked -- HARD when nothing clamps it,
// SOFT when the floor already holds it. Both are ranked debt; only the HARD one
// reaches the #4416 dispatcher.
func TestModeDebtFlagsUnliftedDial(t *testing.T) {
	// The full-implicitness case: read raw, un-clamped, un-journaled. Nothing stops
	// it widening the mandatory floor, so it heads the lift queue.
	wideable := Facts{
		Slug:    "FAK_ALLOW_WIDEN",
		Name:    "FAK_ALLOW_WIDEN",
		Surface: "env",
		File:    "cmd/fak/guard.go",
		Line:    42,
		// Orthogonal only: it names one axis, but clears no other bar.
		Criteria: Criteria{Orthogonal: true},
	}
	// The same implicitness, but the adjudicator's sanitizeProfile clamp already
	// holds it, so it cannot widen the floor. Still debt -- just later in the queue.
	clamped := Facts{
		Slug:     "FAK_NARROW_ONLY",
		Name:     "FAK_NARROW_ONLY",
		Surface:  "env",
		File:     "cmd/fak/guard.go",
		Line:     57,
		Criteria: Criteria{FloorClamped: true, Orthogonal: true},
	}

	if grade, reason := Grade(wideable); grade != GradeHard {
		t.Fatalf("un-lifted AND un-clamped dial: grade = %q, want %q (reason %q)", grade, GradeHard, reason)
	}
	if grade, reason := Grade(clamped); grade != GradeSoft {
		t.Fatalf("un-lifted but floor-clamped dial: grade = %q, want %q (reason %q)", grade, GradeSoft, reason)
	}

	sc := Score([]Facts{wideable, clamped})
	if sc.Schema != Schema {
		t.Errorf("scorecard schema = %q, want %q", sc.Schema, Schema)
	}
	if sc.Hard != 1 || sc.Soft != 1 || sc.Clean != 0 || sc.NotDebt != 0 {
		t.Errorf("rollup = clean %d / hard %d / soft %d / not-debt %d, want 0/1/1/0",
			sc.Clean, sc.Hard, sc.Soft, sc.NotDebt)
	}
	// Both un-lifted dials are DEBT: that is the worklist #2405 burns down.
	if sc.Debt != 2 {
		t.Errorf("ranked debt = %d, want 2 (both un-lifted dials)", sc.Debt)
	}

	// Every ranked dial must carry its provenance and an explanation, or the dump is
	// a number without a next action.
	for _, d := range sc.Dials {
		if d.Lifted {
			t.Errorf("dial %s: Lifted = true, want false", d.Slug)
		}
		if d.Detail == "" {
			t.Errorf("dial %s: empty Detail, want the unmet-bar reason", d.Slug)
		}
		if d.File == "" || d.Line == 0 || d.Surface == "" {
			t.Errorf("dial %s: provenance = %s:%d surface %q, want all three set",
				d.Slug, d.File, d.Line, d.Surface)
		}
	}

	// Only the un-clamped one is the dispatcher's candidate (#4416 files an issue
	// per HARD un-lifted dial); the floor-clamped one must not be fanned out.
	hard := SelectHardUnlifted(sc)
	if len(hard) != 1 {
		t.Fatalf("SelectHardUnlifted returned %d dials, want 1", len(hard))
	}
	if hard[0].Slug != slug(wideable.Slug) {
		t.Errorf("SelectHardUnlifted picked %q, want %q", hard[0].Slug, slug(wideable.Slug))
	}
}

// TestModeDebtPassesLiftedRegime is the second witness: a legitimately typed +
// floor-clamped + journaled dial scores CLEAN even though it is persistent (the
// target is implicitness, not statefulness), and a correctly harness-held,
// model-unreachable safety dial is NOT counted as debt at all.
func TestModeDebtPassesLiftedRegime(t *testing.T) {
	// A persistent dial that is nonetheless fully lifted: typed [regimes] entry,
	// clamped by sanitizeProfile, its flip journaled, one axis. Persistence is fine.
	lifted := Facts{
		Slug:     "FAK_WRITE_REGIME",
		Name:     "FAK_WRITE_REGIME",
		Surface:  "env",
		File:     "internal/adjudicator/rungprofile.go",
		Line:     11,
		Regime:   "write",
		Criteria: Criteria{Lifted: true, FloorClamped: true, Journaled: true, Orthogonal: true},
	}
	// A safety dial the harness holds and the model cannot reach. Holding a safety
	// floor outside the model's reach is the design, not a gap -- so it is excluded
	// from debt entirely, never ranked, even though it clears no other bar.
	held := Facts{
		Slug:       "FAK_SELF_MODIFY_FLOOR",
		Name:       "FAK_SELF_MODIFY_FLOOR",
		Surface:    "env",
		File:       "internal/adjudicator/floor.go",
		Line:       9,
		Criteria:   Criteria{},
		SafetyHold: "harness-only floor; no model-reachable route sets it",
	}

	if grade, reason := Grade(lifted); grade != GradeClean {
		t.Fatalf("fully lifted dial: grade = %q, want %q (reason %q)", grade, GradeClean, reason)
	}
	if grade, reason := Grade(held); grade != GradeNotDebt {
		t.Fatalf("harness-held safety dial: grade = %q, want %q (reason %q)", grade, GradeNotDebt, reason)
	}

	sc := Score([]Facts{lifted, held})
	if sc.Clean != 1 || sc.NotDebt != 1 {
		t.Errorf("rollup = clean %d / not-debt %d, want 1/1", sc.Clean, sc.NotDebt)
	}
	// The load-bearing assertion: NEITHER contributes to the debt count.
	if sc.Debt != 0 || sc.Hard != 0 || sc.Soft != 0 {
		t.Errorf("ranked debt = %d (hard %d, soft %d), want 0 -- a CLEAN dial and a "+
			"harness-held safety dial are both zero debt", sc.Debt, sc.Hard, sc.Soft)
	}
	// And neither is ever fanned out as a lift candidate.
	if hard := SelectHardUnlifted(sc); len(hard) != 0 {
		t.Errorf("SelectHardUnlifted returned %d dials, want 0", len(hard))
	}

	// The excluded dial must NAME its hold, so a reader can audit the exclusion
	// rather than take it on faith.
	for _, d := range sc.Dials {
		if d.Grade == GradeNotDebt && d.Excluded == "" {
			t.Errorf("dial %s: NOT-DEBT with no recorded hold reason", d.Slug)
		}
	}
}

// TestModeDebtGradeInvariants enumerates every criteria combination (and both
// safety-hold states) and pins the properties the rest of the pair relies on. This
// is the test scorer.go's header cites for the HARD => un-lifted invariant.
func TestModeDebtGradeInvariants(t *testing.T) {
	var facts []Facts
	for i := 0; i < 16; i++ {
		c := Criteria{
			Lifted:       i&1 != 0,
			FloorClamped: i&2 != 0,
			Journaled:    i&4 != 0,
			Orthogonal:   i&8 != 0,
		}
		for _, hold := range []string{"", "reviewed harness-held safety dial"} {
			f := Facts{
				Slug:       "dial-" + string(rune('a'+i)) + map[bool]string{true: "-held", false: ""}[hold != ""],
				Surface:    "flag",
				File:       "cmd/fak/x.go",
				Line:       i + 1,
				Criteria:   c,
				SafetyHold: hold,
			}
			facts = append(facts, f)

			grade, reason := Grade(f)
			if reason == "" {
				t.Errorf("%+v: empty reason", f.Criteria)
			}
			switch grade {
			case GradeClean, GradeHard, GradeSoft, GradeNotDebt:
			default:
				t.Fatalf("%+v hold=%q: grade %q is outside the closed vocabulary", c, hold, grade)
			}

			// The exclusion wins over every other fact.
			if hold != "" && grade != GradeNotDebt {
				t.Errorf("%+v with a safety hold: grade = %q, want %q", c, grade, GradeNotDebt)
			}
			if hold == "" && grade == GradeNotDebt {
				t.Errorf("%+v without a safety hold: graded %q -- only a reviewed hold excludes", c, GradeNotDebt)
			}
			// THE invariant dispatch.go leans on: HARD implies un-lifted (and
			// un-clamped). A lifted dial can never join the lift worklist.
			if grade == GradeHard && (c.Lifted || c.FloorClamped) {
				t.Errorf("%+v: graded HARD but lifted=%v clamped=%v -- HARD must mean "+
					"un-lifted AND un-clamped", c, c.Lifted, c.FloorClamped)
			}
			// CLEAN is exactly "all four bars, no hold" -- never a laundered partial.
			if got, want := grade == GradeClean, c.AllMet() && hold == ""; got != want {
				t.Errorf("%+v hold=%q: CLEAN = %v, want %v", c, hold, got, want)
			}
		}
	}

	sc := Score(facts)
	if n := sc.Clean + sc.Hard + sc.Soft + sc.NotDebt; n != len(sc.Dials) {
		t.Errorf("rollup counts sum to %d, want %d dials -- every dial gets exactly one grade",
			n, len(sc.Dials))
	}
	if sc.Debt != sc.Hard+sc.Soft {
		t.Errorf("Debt = %d, want hard+soft = %d", sc.Debt, sc.Hard+sc.Soft)
	}
	for _, d := range SelectHardUnlifted(sc) {
		if !d.IsHard() || d.Lifted {
			t.Errorf("SelectHardUnlifted yielded %s (grade %q, lifted %v)", d.Slug, d.Grade, d.Lifted)
		}
	}

	// Deterministic ordering: the same fact set scores byte-identically, which is
	// what lets a re-run dedup onto the same issues instead of churning them.
	first, err := json.Marshal(sc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	second, err := json.Marshal(Score(facts))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(first) != string(second) {
		t.Error("Score is not deterministic: two runs over the same facts differ")
	}
}

// TestModeDebtScanCensusesLiveTree drives the whole census over a fixture tree: a
// dial lifted into a typed [regimes] entry and mentioned by the floor and journal
// packages scores CLEAN, while a raw un-lifted one is ranked debt. It also pins the
// read-only determinism claim -- scanning twice must marshal to identical bytes.
func TestModeDebtScanCensusesLiveTree(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", rel, err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", rel, err)
		}
	}

	// #2405's typed lift, as this reader expects to find it.
	write("dos.toml", "[regimes.write]\nconsent = true\n")
	// Two dials on the CLI surface: one lifted, one raw. Both names are ones
	// knobcensus's classify recognizes as behavior dials (it is deliberately
	// vocabulary-narrow), and neither carries an omnibus token.
	write("cmd/fak/dials.go", `package main

func dials() {
	fs.Bool("consent", false, "lifted dial")
	fs.Bool("target", false, "un-lifted dial")
}
`)
	// The floor package names the lifted dial, so the sanitizeProfile clamp holds it.
	write("internal/adjudicator/clamp.go", "package adjudicator\n\nvar clamped = []string{\"consent\"}\n")
	// The journal package names it too, so its transition leaves a durable row.
	write("internal/journal/rows.go", "package journal\n\nvar journaled = []string{\"consent\"}\n")

	sc, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}

	byName := map[string]Dial{}
	for _, d := range sc.Dials {
		byName[d.Name] = d
	}
	lifted, ok := byName["consent"]
	if !ok {
		t.Fatalf("census missed the lifted dial; saw %v", sc.Dials)
	}
	if lifted.Grade != GradeClean {
		t.Errorf("lifted dial: grade = %q (%s), want %q", lifted.Grade, lifted.Detail, GradeClean)
	}
	if lifted.Regime != "write" {
		t.Errorf("lifted dial: regime = %q, want %q", lifted.Regime, "write")
	}
	raw, ok := byName["target"]
	if !ok {
		t.Fatalf("census missed the un-lifted dial; saw %v", sc.Dials)
	}
	if raw.Grade != GradeHard {
		t.Errorf("un-lifted dial: grade = %q (%s), want %q", raw.Grade, raw.Detail, GradeHard)
	}
	if raw.Lifted {
		t.Error("un-lifted dial reported Lifted = true")
	}

	// Read-only and deterministic: the same tree yields the same bytes.
	again, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan (second): %v", err)
	}
	a, _ := json.Marshal(sc)
	b, _ := json.Marshal(again)
	if string(a) != string(b) {
		t.Error("Scan is not deterministic: two scans of the same tree differ")
	}
}

// TestModeDebtRegimeIndexReadsTypedLift pins the #2405 reader on its own: an absent
// dos.toml means nothing is lifted (not an error), a [regimes.<name>] table lifts
// both its bare keys and its quoted dial list, and a following table ends the block.
func TestModeDebtRegimeIndexReadsTypedLift(t *testing.T) {
	empty := t.TempDir()
	idx, err := RegimeIndex(empty)
	if err != nil {
		t.Fatalf("RegimeIndex on a tree with no dos.toml: %v", err)
	}
	if len(idx) != 0 {
		t.Errorf("no dos.toml should lift nothing, got %v", idx)
	}

	root := t.TempDir()
	body := "[regimes.write]\nFAK_ALLOW_WIDEN = true\ndials = [\"self-modify\"]\n\n" +
		"[stamp]\nprefix = \"not-a-dial\"\n"
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(body), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}
	idx, err = RegimeIndex(root)
	if err != nil {
		t.Fatalf("RegimeIndex: %v", err)
	}
	// Both spellings of the same control fold to one dial identity.
	if got := idx[foldDial("--allow-widen")]; got != "write" {
		t.Errorf("regime for allow-widen = %q, want %q", got, "write")
	}
	if got := idx[foldDial("self-modify")]; got != "write" {
		t.Errorf("regime for self-modify = %q, want %q", got, "write")
	}
	// A later table is not a regime: its keys must not be laundered into lifted.
	if got, ok := idx[foldDial("prefix")]; ok {
		t.Errorf("key from the [stamp] table leaked in as regime %q", got)
	}
}

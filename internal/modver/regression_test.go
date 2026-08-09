package modver

import (
	"bytes"
	"strings"
	"testing"
)

// regression_test.go — the named witness for #2470, exercised in BOTH directions:
// a fixture regression must warn, and everything that is not a regression must
// stay silent.

// scoreOf is a local pointer helper: the Module/LedgerRow score field is *float64
// so an absent score stays distinguishable from a zero one.
func scoreOf(v float64) *float64 { return &v }

// regressionLedger is the fixture prior ledger: internal/alpha last scored 0.80 at
// rev 3, internal/beta 0.40 at rev 2, cmd/fak 0.55 at rev 7 — plus a scar line, which
// an append-only ledger a fleet writes will have.
const regressionLedger = `{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/alpha","kind":"internal","rev":3,"score":0.8}
this line is scar tissue and must be tolerated
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/beta","kind":"internal","rev":2,"score":0.4}
{"schema":"fak-module-versions/1","ts":"2026-07-02T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":7,"score":0.55}
`

// TestScoreDropsWarnsOnFixtureRegression is the WARN direction: one module scored
// lower than its last stamped score, and the advisory names it with both ends of the
// movement. The modules beside it — one improved, one flat, one never scored before —
// prove the check is a drop detector, not a "score changed" detector.
func TestScoreDropsWarnsOnFixtureRegression(t *testing.T) {
	rep := Report{Head: "deadbee1", Modules: []Module{
		{Name: "cmd/fak", Kind: "cmd", Rev: 9, Score: scoreOf(0.55)},                 // flat: silent
		{Name: "internal/alpha", Kind: "internal", Rev: 5, Score: scoreOf(0.5)},      // DROP 0.8 -> 0.5
		{Name: "internal/beta", Kind: "internal", Rev: 4, Score: scoreOf(0.9)},       // improved: silent
		{Name: "internal/gamma", Kind: "internal", Rev: 1, Score: scoreOf(0.1)},      // first observation: silent
		{Name: "internal/delta", Kind: "internal", Rev: 2, Score: nil},               // nothing joined: silent
	}}
	drops := ScoreDrops(rep, []byte(regressionLedger))
	if len(drops) != 1 {
		t.Fatalf("ScoreDrops = %+v, want exactly 1 drop (internal/alpha)", drops)
	}
	got := drops[0]
	// prev/cur are VARIABLES, not constants, so Delta below is computed the way
	// ScoreDrops computes it — a runtime float64 subtraction yielding
	// -0.30000000000000004. Written as the constant expression `0.5 - 0.8` the
	// compiler folds it at arbitrary precision to exactly -0.3, which no runtime
	// subtraction of these two float64s ever equals, and the struct compare below
	// could never pass. This mirrors the artifact ScoreDropAdvisory documents.
	prev, cur := 0.8, 0.5
	want := ScoreDrop{
		Module:  "internal/alpha",
		Prev:    prev,
		Current: cur,
		Delta:   cur - prev,
		PrevRev: 3,
		Rev:     5,
		PrevTS:  "2026-07-01T00:00:00Z",
	}
	if got != want {
		t.Errorf("drop = %+v, want %+v", got, want)
	}
}

// TestScoreDropsSilentWithoutRegression is the SILENT direction: a report whose every
// scored module held or improved yields no findings at all, and the renderer writes
// nothing — a clean run must not print a "0 regressions" line.
func TestScoreDropsSilentWithoutRegression(t *testing.T) {
	rep := Report{Head: "deadbee1", Modules: []Module{
		{Name: "cmd/fak", Kind: "cmd", Rev: 9, Score: scoreOf(0.55)},            // exactly equal
		{Name: "internal/alpha", Kind: "internal", Rev: 5, Score: scoreOf(0.81)}, // up
		{Name: "internal/beta", Kind: "internal", Rev: 4, Score: scoreOf(0.4)},   // exactly equal
	}}
	if drops := ScoreDrops(rep, []byte(regressionLedger)); len(drops) != 0 {
		t.Fatalf("ScoreDrops = %+v, want none (no module scored lower)", drops)
	}
	var b bytes.Buffer
	if wrote := ScoreDropAdvisory(&b, nil); wrote || b.Len() != 0 {
		t.Errorf("advisory over no drops wrote %q (wrote=%v), want silence", b.String(), wrote)
	}
}

// TestScoreDropsSilentWithoutPriorLedger pins the no-memory case: with an empty (or
// absent) ledger every module is a first observation, so nothing can have regressed.
func TestScoreDropsSilentWithoutPriorLedger(t *testing.T) {
	rep := Report{Modules: []Module{{Name: "internal/alpha", Rev: 5, Score: scoreOf(0.1)}}}
	if drops := ScoreDrops(rep, nil); len(drops) != 0 {
		t.Errorf("ScoreDrops against an empty ledger = %+v, want none", drops)
	}
}

// TestScoreDropsAnchorsOnLastScoredRow pins the anchor choice: a later stamp taken
// with no score join writes a scoreless row, and that row must NOT erase the memory of
// what the module last scored. Anchoring on the literal last row would swallow this
// regression silently — the exact failure the version-everything loop needs caught.
func TestScoreDropsAnchorsOnLastScoredRow(t *testing.T) {
	ledger := `{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/alpha","kind":"internal","rev":3,"score":0.8}
{"schema":"fak-module-versions/1","ts":"2026-07-05T00:00:00Z","module":"internal/alpha","kind":"internal","rev":4}
`
	rep := Report{Modules: []Module{{Name: "internal/alpha", Rev: 6, Score: scoreOf(0.2)}}}
	drops := ScoreDrops(rep, []byte(ledger))
	if len(drops) != 1 {
		t.Fatalf("ScoreDrops = %+v, want the drop across the unscored stamp", drops)
	}
	if drops[0].Prev != 0.8 || drops[0].PrevRev != 3 {
		t.Errorf("anchored on prev=%g rev=%d, want the last SCORED row (0.8 at rev 3)", drops[0].Prev, drops[0].PrevRev)
	}
}

// TestScoreDropsOrderWorstFirst pins the ordering contract: the biggest fall leads, so
// truncating or skimming the advisory keeps the movement most worth looking at.
func TestScoreDropsOrderWorstFirst(t *testing.T) {
	rep := Report{Modules: []Module{
		{Name: "internal/beta", Rev: 4, Score: scoreOf(0.35)},  // -0.05
		{Name: "internal/alpha", Rev: 5, Score: scoreOf(0.1)},  // -0.70
		{Name: "cmd/fak", Rev: 9, Score: scoreOf(0.35)},        // -0.20
	}}
	drops := ScoreDrops(rep, []byte(regressionLedger))
	var names []string
	for _, d := range drops {
		names = append(names, d.Module)
	}
	want := []string{"internal/alpha", "cmd/fak", "internal/beta"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Errorf("drop order = %v, want %v (worst first)", names, want)
	}
}

// TestScoreDropAdvisoryWritesWarning pins the rendered warning: it names the module,
// both scores, the signed delta, and says out loud that it blocks nothing.
func TestScoreDropAdvisoryWritesWarning(t *testing.T) {
	rep := Report{Modules: []Module{{Name: "internal/alpha", Rev: 5, Score: scoreOf(0.5)}}}
	var b bytes.Buffer
	if wrote := ScoreDropAdvisory(&b, ScoreDrops(rep, []byte(regressionLedger))); !wrote {
		t.Fatal("advisory over a real drop wrote nothing, want a warning")
	}
	out := b.String()
	for _, want := range []string{"internal/alpha", "0.8 -> 0.5", "-0.3", "r3 -> r5", "nothing is blocked"} {
		if !strings.Contains(out, want) {
			t.Errorf("advisory = %q, want it to contain %q", out, want)
		}
	}
}

package scorecardpane

// paneparity_test.go — the Go-pane <-> Python-pane reconciler.
//
// The scorecard control pane exists in TWO implementations that are meant to fold
// the SAME portfolio (this file's package doc: the Go fold is "byte-compatible with
// the Python contract"):
//
//   - Go pane:     internal/scorecardpane.Cards           (controlpane.go)
//   - Python pane: tools/scorecard_control_pane.py SCORECARDS
//
// Nothing reconciled the two, and they drifted: a card folded into only one pane
// regresses freely under whichever loop runs the other pane, invisible to its
// ratchet. The Python side already guards its own breadth (every tools/*_scorecard.py
// is registered-or-excluded: test_every_scorecard_is_registered_or_excluded in
// tools/scorecard_control_pane_test.py). This test guards the previously-unguarded
// Go<->Python axis: the two panes must fold the same set of debt keys.
//
// It keys on the DEBT key (what actually sums into total_debt and is pinned per-key
// in the baseline), not the card key, so a cosmetic card-key spelling difference does
// not read as membership drift — that spelling is checked separately (knownKeyDrift)
// because per-key baselines depend on it.

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// repoRoot is two levels up from internal/scorecardpane.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	return root
}

// knownPaneDivergence enumerates debt keys DELIBERATELY (for now) folded into only
// one pane. Each is real drift the coverage audit surfaced; the entry is the
// burn-down worklist, and its presence keeps the gate green on today's tree while
// hard-blocking any NEW divergence. Mirrors EXCLUDED_SCORECARDS in
// tools/scorecard_control_pane_test.py — a documented, reasoned allow-list.
//
// To retire an entry: fold the card into the missing pane (or delete it from the
// pane that has it) and remove the line here; the stale-entry guard below fails if a
// line no longer corresponds to a live divergence, so this list cannot rot.
var knownPaneDivergence = map[string]string{
	"lightgap_debt": "python-only: lightgap remains a Python control-pane producer until its Go card is wired",
	// Python-pane-only — port into internal/scorecardpane.Cards, or retire from Python:
	"popularization_debt": "python-only: popularization_readiness_scorecard.py folded in the Python pane only",
	"guard_accuracy_debt": "python-only: `go run ./cmd/fak score guard-accuracy` folded in the Python pane only",
	"commit_debt":         "python-only: commit_subject_coverage.py folded in the Python pane only",
	// Go-pane-only — port into tools/scorecard_control_pane.py SCORECARDS, or retire from Go:
	"antipattern_debt":       "go-only: antipattern-scorecard folded in the Go pane only",
	"negframe_debt":          "go-only: negframe folded in the Go pane only",
	"negation_tax_debt":      "go-only: negation-tax folded in the Go pane only",
	"negation_operator_debt": "go-only: negation-operator folded in the Go pane only",
	"residual_count":         "go-only: the osp_residual card (#5022) folded in the Go pane only — its SCORECARDS row lands under the `tools` lane, which a live peer lease held when the card was wired; add the row and delete this line",
	"flow_debt":              "go-only: flow-metrics is folded in the Go pane; register and pin it in the Python pane before deleting this line",
}

// knownKeyDrift enumerates debt keys folded in BOTH panes but under a different card
// KEY. The per-metric baseline (tools/scorecard_baseline.json) keys on the card key,
// so a spelling mismatch silently mis-aligns that card's pinned debt/grade across the
// two panes. Reconcile the spelling, then delete the entry.
var knownKeyDrift = map[string]string{
	"sota_debt": "card key 'sota' (python) vs 'sota_coverage' (go) — reconcile so per-key baselines align",
}

// scorecardsEntryRe matches one SCORECARDS dict literal in the Python pane, capturing
// (card key, debt key). The shape `{"key": "...", "debt": "..."` is unique to
// SCORECARDS rows — EXCLUDED_SCORECARDS uses `"file.py": "reason"` (no "key":) and
// GO_BACKED_KEYS is a comprehension, so neither is captured.
var scorecardsEntryRe = regexp.MustCompile(`\{"key":\s*"([^"]+)",\s*"debt":\s*"([^"]+)"`)

// TestGoPythonPaneParity fails when the Go and Python control panes fold a different
// set of debt keys (a membership divergence) or spell the same card's key differently,
// unless the difference is an annotated entry in knownPaneDivergence / knownKeyDrift.
func TestGoPythonPaneParity(t *testing.T) {
	root := repoRoot(t)
	pyPath := filepath.Join(root, "tools", "scorecard_control_pane.py")
	src, err := os.ReadFile(pyPath)
	if err != nil {
		t.Fatalf("read python pane %s: %v", pyPath, err)
	}

	// Python pane: debt key -> card key.
	pyKeyByDebt := map[string]string{}
	for _, m := range scorecardsEntryRe.FindAllStringSubmatch(string(src), -1) {
		pyKeyByDebt[m[2]] = m[1]
	}
	if len(pyKeyByDebt) < 40 {
		t.Fatalf("parsed only %d SCORECARDS entries from %s — regex or file format drifted; "+
			"parity cannot be checked", len(pyKeyByDebt), pyPath)
	}

	// Go pane: debt key -> card key (in-process; Cards is this package's var).
	goKeyByDebt := map[string]string{}
	for _, c := range Cards {
		if prev, dup := goKeyByDebt[c.Debt]; dup {
			t.Errorf("Go Cards fold two cards under one debt key %q (%q and %q) — Fold sums by "+
				"debt key, so give each card a distinct debt key", c.Debt, prev, c.Key)
		}
		goKeyByDebt[c.Debt] = c.Key
	}

	// Go-pane-only membership divergence.
	for debt, goKey := range goKeyByDebt {
		if _, inPy := pyKeyByDebt[debt]; inPy {
			continue
		}
		if _, allowed := knownPaneDivergence[debt]; allowed {
			continue
		}
		t.Errorf("Go-pane-only card %q (debt %q): folded in internal/scorecardpane.Cards but NOT "+
			"in tools/scorecard_control_pane.py SCORECARDS. The two panes must fold the same "+
			"portfolio — wire it into the Python SCORECARDS, or add %q to knownPaneDivergence "+
			"with a reason.", goKey, debt, debt)
	}

	// Python-pane-only membership divergence.
	for debt, pyKey := range pyKeyByDebt {
		if _, inGo := goKeyByDebt[debt]; inGo {
			continue
		}
		if _, allowed := knownPaneDivergence[debt]; allowed {
			continue
		}
		t.Errorf("Python-pane-only card %q (debt %q): folded in tools/scorecard_control_pane.py "+
			"SCORECARDS but NOT in internal/scorecardpane.Cards — wire it into the Go Cards, or "+
			"add %q to knownPaneDivergence with a reason.", pyKey, debt, debt)
	}

	// Card-key spelling drift for debt keys folded in BOTH panes.
	for debt, goKey := range goKeyByDebt {
		pyKey, inPy := pyKeyByDebt[debt]
		if !inPy || pyKey == goKey {
			continue
		}
		if _, allowed := knownKeyDrift[debt]; allowed {
			continue
		}
		t.Errorf("card-key drift for debt %q: Go Cards uses key %q, Python SCORECARDS uses key "+
			"%q. The per-key baseline keys on this spelling, so they must match — reconcile, or "+
			"add %q to knownKeyDrift with a reason.", debt, goKey, pyKey, debt)
	}

	// Stale-allow-list guard: a listed divergence that no longer exists must be
	// removed, so the allow-list stays a live worklist rather than fossilizing.
	for debt := range knownPaneDivergence {
		_, inGo := goKeyByDebt[debt]
		_, inPy := pyKeyByDebt[debt]
		switch {
		case inGo && inPy:
			t.Errorf("knownPaneDivergence lists %q but it is now folded in BOTH panes — the drift "+
				"is reconciled; delete the entry.", debt)
		case !inGo && !inPy:
			t.Errorf("knownPaneDivergence lists %q but it is in NEITHER pane — stale; delete the "+
				"entry.", debt)
		}
	}
	for debt := range knownKeyDrift {
		goKey, inGo := goKeyByDebt[debt]
		pyKey, inPy := pyKeyByDebt[debt]
		if inGo && inPy && goKey == pyKey {
			t.Errorf("knownKeyDrift lists %q but the keys now match (%q) — reconciled; delete the "+
				"entry.", debt, goKey)
		}
	}
}

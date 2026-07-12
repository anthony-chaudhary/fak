package scorecardpane

// incremental.go — the shift-left (--since) fold of the portfolio control-pane.
//
// The full control-pane runs all ~40 cards as subprocesses every time — a fixed,
// heavy cost an at-origin agent pays just to learn "did my edit move the number?".
// Most edits touch a handful of corpora, so most cards' debt provably cannot have
// moved. CollectSince exploits that: a card whose declared Corpus is disjoint from
// the diff since <ref> is CARRIED from the pinned baseline (its exact (debt, weight)
// contribution reproduced) instead of re-measured; only cards whose corpus the diff
// touched are run. The fold is otherwise unchanged, so the numbers a carried card
// contributes are identical to its last full measurement — never fabricated.
//
// Fail-safe by construction: carrying is opt-in per card (an empty Corpus always
// measures), requires a pinned baseline that pins BOTH the card's debt and its grade
// weight, and any card whose corpus is touched — or whose baseline entry is missing —
// is measured. The only way carrying can mislead is an under-broad Corpus glob, which
// is why Corpus must be a documented SUPERSET of what a card reads.

import (
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scdiff"
)

// IncrementalInfo annotates a control-pane payload folded under --since. Its presence
// marks the payload as an incremental read — some cards were carried from the baseline,
// NOT freshly measured — so a downstream sink must not pin or post it as a new floor.
type IncrementalInfo struct {
	Since          string   `json:"since"`
	BaselineCommit string   `json:"baseline_commit"`
	ChangedFiles   int      `json:"changed_files"`
	Measured       int      `json:"measured_cards"`
	Carried        int      `json:"carried_cards"`
	CarriedKeys    []string `json:"carried_keys"`
}

// CanCarry reports whether a card may be CARRIED from the baseline instead of
// re-measured, given the changed-path set. All must hold: the card declares a Corpus,
// none of the changed paths intersects that Corpus, and the baseline pins BOTH a debt
// and a grade weight for the card (so the carried metric reproduces the baseline's
// exact contribution). Any card failing these is measured — carrying never guesses.
func CanCarry(card Card, changed []string, baseline *Baseline) bool {
	if len(card.Corpus) == 0 || baseline == nil {
		return false
	}
	if len(scdiff.Filter(changed, card.Corpus)) > 0 {
		return false // corpus touched -> must re-measure
	}
	if _, ok := baseline.Metrics[card.Key]; !ok {
		return false
	}
	if _, ok := baseline.GradeWeights[card.Key]; !ok {
		return false
	}
	// Only carry when the pinned weight maps back to a known letter grade, so the
	// synthesized metric reproduces the baseline weight exactly through the fold.
	return weightLetter(baseline.GradeWeights[card.Key]) != "?"
}

// carriedMetric synthesizes the Metric for a carried card from the pinned baseline:
// the baseline debt, and the grade LETTER whose severity weight equals the baseline's
// pinned weight — so the unchanged Fold reproduces the card's exact baseline (debt,
// grade_weight) contribution. The caller must have gated on CanCarry.
func carriedMetric(card Card, baseline *Baseline) Metric {
	debt := baseline.Metrics[card.Key]
	letter := weightLetter(baseline.GradeWeights[card.Key])
	return Metric{
		Key: card.Key, Label: card.Label, DebtKey: card.Debt,
		Debt: &debt, Grade: &letter, OK: true, Verdict: "CARRIED", Carried: true,
	}
}

// CollectSince is the incremental shell of the control-pane: it walks the cards in
// canonical order and, for each, either CARRIES it from the baseline (Corpus untouched
// by the diff since `since`) or MEASURES it as a subprocess. It returns the per-card
// metrics (canonical order preserved, so the fold and --json shape are unchanged) plus
// the IncrementalInfo describing what was carried vs measured. `changed` is the diff
// set from scdiff.ChangedPaths; an empty `changed` (nothing changed since `since`)
// carries every card that CanCarry, so the fold reproduces the baseline exactly.
func CollectSince(root, python string, timeout time.Duration, since string, changed []string, baseline *Baseline) ([]Metric, IncrementalInfo) {
	if python == "" {
		python = defaultPython()
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	metrics := make([]Metric, 0, len(Cards))
	carriedKeys := make([]string, 0, len(Cards))
	measured := 0
	for _, card := range Cards {
		if CanCarry(card, changed, baseline) {
			metrics = append(metrics, carriedMetric(card, baseline))
			carriedKeys = append(carriedKeys, card.Key)
			continue
		}
		payload, errMsg := RunScorecard(root, card, python, timeout)
		metrics = append(metrics, MetricFromPayload(card, payload, errMsg))
		measured++
	}
	sort.Strings(carriedKeys)
	info := IncrementalInfo{
		Since: since, ChangedFiles: len(changed),
		Measured: measured, Carried: len(carriedKeys), CarriedKeys: carriedKeys,
	}
	if baseline != nil {
		info.BaselineCommit = baseline.Commit
	}
	return metrics, info
}

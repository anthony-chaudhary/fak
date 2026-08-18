package issuefanout

// scorecard.go gives the issue-fanout planner its observability scorecard fold: a
// deterministic grade over the planner's health, scored from WITNESSES only — the
// adoption report (git-shipped leaves + gh-filed marker keys) plus the planner's own
// emitted plan — never a self-report that the planner is "still good". This is the
// obs-scorecard follow-on (#2520): "is it still good" is a command, not an audit.
//
// Three KPIs fold the issue's named health axes (usage / failure rate / drift):
//   - adoption_floor (usage): fraction of shipped leaves that cleared the fan-out
//     floor (>= MinFanout follow-ons filed). One defect per below-floor leaf.
//   - marker_integrity (failure_rate): orphan fan-out markers — keys filed against a
//     leaf not in the shipped set — are bookkeeping failures. One defect per orphan.
//   - taxonomy_drift (drift): the planner's own Build, on a canonical spine, must
//     still emit a follow-on for every doctrine area. One defect per missing area.
//
// Pure over (shippedLeaves, markerKeys) like the rest of the leaf: the caller gathers
// the witnesses (git for leaves, gh for marker keys); the scorecard only grades.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// ScorecardSchema is the control-pane schema id for the issue-fanout scorecard.
	ScorecardSchema = "fak-issue-fanout-scorecard/1"
	// DebtKey is the headline integer the control pane folds (corpus.issuefanout_debt).
	DebtKey = "issuefanout_debt"
)

// DoctrineAreas is the canonical follow-on area set the spine-first default demands a
// shipped spine fan out into. The drift KPI grades the planner's emitted plan against
// THIS anchor, not against the taxonomy's current shape, so a template silently deleted
// from the taxonomy surfaces as drift rather than disappearing unnoticed. A deliberate
// doctrine change updates both the taxonomy AND this list (the guard test pins them
// equal today, so a one-sided edit reds the build on purpose).
var DoctrineAreas = []string{"qa", "dogfood", "product", "observability", "integration", "docs", "release"}

// Scorecard grades the issue-fanout planner's health from witnesses. shippedLeaves are
// the git-gathered shipped leaves; markerKeys are the gh-gathered fanout-<leaf>-<slug>
// marker keys filed against issues. Both are the SAME witnesses Adoption folds; the
// scorecard never reads a self-report. The returned Payload is the shared control-pane
// envelope every scorecard in the family emits, so it folds into the scorecard control
// pane and a --json/--compare CLI identically.
func Scorecard(shippedLeaves, markerKeys []string) scorecard.Payload {
	rep := Adoption(shippedLeaves, markerKeys)

	adoption := kpiAdoption(rep)
	integrity := kpiIntegrity(rep)
	drift := kpiDrift()

	debt := len(adoption.Defects) + len(integrity.Defects) + len(drift.Defects)
	finding := fmt.Sprintf("issuefanout_debt %d = %d adoption gap(s) + %d orphan marker(s) + %d taxonomy drift(s)",
		debt, len(adoption.Defects), len(integrity.Defects), len(drift.Defects))
	return scorecard.Fold(ScorecardSchema, []scorecard.KPI{adoption, integrity, drift}, DebtKey, nil, scorecard.Messages{
		Finding:         finding,
		Reason:          finding,
		FindingClean:    fmt.Sprintf("planner healthy: %d/%d shipped leaf/leaves clear the fan-out floor, 0 orphan markers, taxonomy covers all %d doctrine areas", rep.ClearedLeaves, rep.ShippedLeaves, len(DoctrineAreas)),
		NextAction:      "retire worst-first: file the missing follow-ons for below-floor leaves, prune/re-file orphan markers, restore any dropped doctrine-area template (see defects)",
		NextActionClean: "hold the line: re-run on every spine ship or fan-out filing; the planner is healthy when debt is 0",
		ExtraCorpus: map[string]any{
			"shipped_leaves":  rep.ShippedLeaves,
			"cleared_leaves":  rep.ClearedLeaves,
			"gap_leaves":      rep.GapLeaves,
			"orphan_markers":  rep.OrphanMarkers,
			"doctrine_areas":  DoctrineAreas,
			"adoption_score":  adoption.Score,
			"integrity_score": integrity.Score,
			"drift_score":     drift.Score,
		},
	})
}

// kpiAdoption grades the usage axis: a shipped spine is healthy when its >= MinFanout
// follow-ons are filed. Score is the cleared fraction; 100 when nothing is shipped
// (the scorecard family's "no work, perfect" convention).
func kpiAdoption(r AdoptionReport) scorecard.KPI {
	var defects []string
	for _, l := range r.Leaves {
		if l.ClearsFloor {
			continue
		}
		defects = append(defects, fmt.Sprintf("leaf %s: %d/%d follow-on(s) filed - below the fan-out floor", l.Leaf, l.FanoutFiled, r.MinFanout))
	}
	return scorecard.KPI{
		Key:     "adoption_floor",
		Group:   "usage",
		Score:   scorecard.CompletionPercent(r.ClearedLeaves, r.ShippedLeaves),
		Detail:  fmt.Sprintf("%d/%d shipped leaf/leaves clear the >= %d fan-out floor", r.ClearedLeaves, r.ShippedLeaves, r.MinFanout),
		Defects: defects,
	}
}

// kpiIntegrity grades the failure-rate axis: a fan-out marker keyed against a leaf not in
// the shipped set is a bookkeeping failure (a renamed/removed leaf, or a typo). Score is
// the clean-marker fraction over all filed fan-out markers; 100 when nothing is filed.
func kpiIntegrity(r AdoptionReport) scorecard.KPI {
	clean := 0 // markers credited to a shipped leaf (the non-orphan count)
	for _, l := range r.Leaves {
		clean += l.FanoutFiled
	}
	var defects []string
	for _, o := range r.OrphanMarkers {
		defects = append(defects, fmt.Sprintf("orphan marker %s: fan-out filed against a leaf not in the shipped set", o))
	}
	return scorecard.KPI{
		Key:     "marker_integrity",
		Group:   "failure_rate",
		Score:   scorecard.CompletionPercent(clean, clean+len(r.OrphanMarkers)),
		Detail:  fmt.Sprintf("%d credited marker(s), %d orphan(s)", clean, len(r.OrphanMarkers)),
		Defects: defects,
	}
}

// kpiDrift grades the drift axis: the planner's own Build, on a canonical spine, must
// still emit a follow-on for every doctrine area. The witness is the emitted plan, not a
// self-report - if a template is dropped from the taxonomy, this KPI fires.
func kpiDrift() scorecard.KPI {
	plan, err := Build(Input{
		Title:    "issue fanout planner",
		Leaf:     "issuefanout",
		SpineRef: "canonical (scorecard drift witness)",
	})
	emitted := map[string]bool{}
	if err == nil {
		for _, c := range plan.Candidates {
			if len(c.Labels) >= 2 {
				emitted[c.Labels[1]] = true
			}
		}
	}
	var defects []string
	for _, area := range DoctrineAreas {
		if !emitted[area] {
			defects = append(defects, fmt.Sprintf("doctrine area %s: no follow-on emitted - the planner taxonomy has drifted from the doctrine", area))
		}
	}
	present := len(DoctrineAreas) - len(defects)
	return scorecard.KPI{
		Key:     "taxonomy_drift",
		Group:   "drift",
		Score:   scorecard.CompletionPercent(present, len(DoctrineAreas)),
		Detail:  fmt.Sprintf("%d/%d doctrine area(s) covered by the emitted plan", present, len(DoctrineAreas)),
		Defects: defects,
	}
}

// ratio is the 0..100 cleared fraction (100 when total == 0), rounded to one decimal -

// RenderScorecard prints the scorecard for a terminal: the shared renderer over the
// headline debt key, so a `fak issue fanout-health` style surface reads identically to
// every other card in the family.
func RenderScorecard(p scorecard.Payload) string {
	return scorecard.Render(p, DebtKey)
}

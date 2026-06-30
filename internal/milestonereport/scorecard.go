package milestonereport

// scorecard.go gives the milestone report its RSI-scorecard surface: a
// deterministic milestone_debt integer + a worst-first retire worklist, folded
// into the shared pkg/scorecard control-pane Payload the rest of the scorecard
// family emits. This is what lets the RSI loop retire/harden milestones like any
// other surface (issue #1444, epic #1436).
//
// The design tension (#1444): the CLIMB is ALREADY partly graded by
// internal/supportmaturityscore (support_maturity_debt = declared-target
// shortfall). milestone_debt does NOT re-derive a parallel maturity scale; it
// COMPOSES the milestone report's OWN two dimensions — the distance-from-MATURED
// (M4) climb shortfall this report already folds (the maturedRung floor) PLUS the
// roadmap's un-progressed tracked-epic gaps — into one retire-able number. The
// climb half is the report's own matured-floor framing (not the fenced
// declared-target one), so the two scorecards stay distinct: support-maturity
// fences each cell to its regime ceiling, milestone scores raw distance-to-matured
// across the grid as the project's headline climb, alongside the roadmap.

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// ScorecardSchema is the control-pane schema id for the milestone scorecard.
	ScorecardSchema = "fak-milestone-scorecard/1"
	// DebtKey is the headline integer the control pane folds (corpus.milestone_debt).
	DebtKey = "milestone_debt"
)

// WorklistItem is one worst-first retire step: which cell to climb or which epic
// to advance next, and by how much (the per-item debt severity). The list is
// ordered most-severe-first so a loop drains it top-down.
type WorklistItem struct {
	Kind     string `json:"kind"`           // "climb" | "roadmap"
	Ref      string `json:"ref"`            // the cell ("family x backend") or epic ("#N title")
	Severity int    `json:"severity"`       // this item's debt weight (rung shortfall, or open children)
	Need     string `json:"need"`           // the next checkable step that retires it
	Rung     string `json:"rung,omitempty"` // the cell's current rung (climb items)
}

// BuildScorecard folds the two ALREADY-MEASURED milestone dimensions into the
// shared scorecard Payload. It is pure over (Maturity, Epics) so the debt is
// deterministic for a fixed grid + roadmap — the live runners stay in collect.go.
//
// milestone_debt = climbDebt + roadmapDebt, where:
//   - climbDebt = sum over cells below the MATURED floor (M4Correct) of
//     (maturedRung - current) rung-steps — the distance-from-matured the report
//     already names via Matured/maturedRung. One defect per missing rung-step, so
//     the kernel's len(Defects) count fold equals the rung-weighted shortfall.
//   - roadmapDebt = sum over measured DISCRETE epics of open children
//     (Total - Closed) — the un-progressed roadmap gap. Ongoing PROGRAMS have no
//     100% (worktype.Class.Ongoing), so they never contribute roadmap debt; an
//     UNREADABLE epic is not counted as debt (you cannot retire what you cannot
//     measure — that is the report's ACTION verdict's job, not the scorecard's).
//
// The worst-first worklist orders climb cells by rung shortfall (lowest rung,
// biggest gap, first) ahead of roadmap gaps by open-child count — the climb is
// the deeper debt, so it retires first.
func BuildScorecard(m Maturity, e Epics) scorecard.Payload {
	climbDefects, climbItems, climbDebt := climbShortfall(m)
	roadDefects, roadItems, roadDebt := roadmapShortfall(e)

	worklist := append([]WorklistItem{}, climbItems...)
	worklist = append(worklist, roadItems...)
	sortWorklist(worklist)

	climbKPI := scorecard.KPI{
		Key:   "climb_distance_to_matured",
		Group: "climb",
		Score: kpiScore(m.Matured, m.Cells),
		Detail: fmt.Sprintf("%d/%d cell(s) matured (M4+); climb shortfall sum(M4-current) = %d rung(s)",
			m.Matured, m.Cells, climbDebt),
		Defects: climbDefects,
	}
	roadKPI := scorecard.KPI{
		Key:   "roadmap_open_children",
		Group: "roadmap",
		Score: kpiScore(e.Closed, e.Total),
		Detail: fmt.Sprintf("%d/%d discrete child(ren) closed; roadmap shortfall = %d open child(ren) over %d measured discrete epic(s)",
			e.Closed, e.Total, roadDebt, e.Discrete),
		Defects: roadDefects,
	}

	debt := climbDebt + roadDebt
	finding := fmt.Sprintf("milestone_debt %d = %d climb rung(s) to MATURED + %d open roadmap child(ren)",
		debt, climbDebt, roadDebt)
	return scorecard.Fold(ScorecardSchema, []scorecard.KPI{climbKPI, roadKPI}, DebtKey, nil, scorecard.Messages{
		Finding: finding,
		// The per-step defects are intentionally duplicated so the kernel's len(Defects)
		// fold equals the weighted debt; surface the compact finding as the reason rather
		// than the (long, repeated) joined defect list.
		Reason: finding,
		FindingClean: fmt.Sprintf("every cell is MATURED (M4+) and every measured discrete epic is complete across %d cell(s) + %d epic(s)",
			m.Cells, e.Measured),
		NextAction:      "retire worst-first: climb the lowest-rung cell to MATURED, then close the most-open discrete epic (see milestone_worklist)",
		NextActionClean: "hold the line: re-run `fak milestone-scorecard` on every model/backend or epic change; the worklist is empty",
		ExtraCorpus: map[string]any{
			"climb_debt":         climbDebt,
			"roadmap_debt":       roadDebt,
			"cells":              m.Cells,
			"matured":            m.Matured,
			"matured_floor":      maturedRung.String(),
			"epics_tracked":      e.Tracked,
			"epics_measured":     e.Measured,
			"discrete_epics":     e.Discrete,
			"ongoing_programs":   e.Programs,
			"milestone_worklist": worklist,
		},
	})
}

// climbShortfall folds the distance-to-MATURED debt. Every cell at rung r < M4 owes
// (M4 - r) rung-steps; summed over the M0..M3 distribution buckets that IS the
// climb shortfall. It emits ONE DEFECT PER RUNG-STEP (n*steps of them per bucket)
// so the kernel's count fold (scorecard.Fold sums len(Defects)) equals the
// rung-weighted shortfall — the same contract internal/supportmaturityscore uses.
// The worklist carries one ITEM per rung bucket (severity = the bucket's weighted
// debt), keyed by rung so the order is deterministic.
func climbShortfall(m Maturity) (defects []string, items []WorklistItem, debt int) {
	for _, r := range supportmaturity.Rungs {
		if !r.Less(maturedRung) {
			break // M4..M7 are at/above the matured floor: zero climb debt
		}
		n := m.Dist[r.String()]
		if n == 0 {
			continue
		}
		steps := int(maturedRung) - int(r) // rung-steps each such cell owes
		cellDebt := n * steps
		debt += cellDebt
		need := fmt.Sprintf("climb %d cell(s) from %s(%s) up %d rung(s) to %s(%s)",
			n, r.String(), r.Label(), steps, maturedRung.String(), maturedRung.Label())
		// One defect per owed rung-step, so len(Defects) == the weighted shortfall.
		for i := 0; i < cellDebt; i++ {
			defects = append(defects, need)
		}
		items = append(items, WorklistItem{
			Kind:     "climb",
			Ref:      fmt.Sprintf("%d cell(s) @ %s", n, r.String()),
			Severity: cellDebt,
			Need:     need,
			Rung:     r.String(),
		})
	}
	return defects, items, debt
}

// roadmapShortfall folds the un-progressed roadmap debt: open children summed over
// the MEASURED DISCRETE epics. An ongoing program has no 100% so it never owes
// roadmap debt; an unreadable epic is excluded (you cannot retire an unmeasured
// gap). One defect + one worklist item per epic with open children.
func roadmapShortfall(e Epics) (defects []string, items []WorklistItem, debt int) {
	for _, row := range e.Rows {
		if row.Err != "" || row.Ongoing() {
			continue
		}
		open := row.Total - row.Closed
		if open <= 0 {
			continue
		}
		debt += open
		need := fmt.Sprintf("close %d open child(ren) of #%d %s (%d/%d done)",
			open, row.Number, row.Title, row.Closed, row.Total)
		// One defect per open child, so len(Defects) == the open-child shortfall.
		for i := 0; i < open; i++ {
			defects = append(defects, need)
		}
		items = append(items, WorklistItem{
			Kind:     "roadmap",
			Ref:      fmt.Sprintf("#%d %s", row.Number, row.Title),
			Severity: open,
			Need:     need,
		})
	}
	return defects, items, debt
}

// sortWorklist orders the retire steps worst-first: climb items (the deeper,
// structural debt) ahead of roadmap items, each by descending severity, with a
// stable tiebreak on Ref so the order is fully deterministic.
func sortWorklist(items []WorklistItem) {
	rank := func(kind string) int {
		if kind == "climb" {
			return 0
		}
		return 1
	}
	sort.SliceStable(items, func(i, j int) bool {
		ri, rj := rank(items[i].Kind), rank(items[j].Kind)
		if ri != rj {
			return ri < rj
		}
		if items[i].Severity != items[j].Severity {
			return items[i].Severity > items[j].Severity
		}
		return items[i].Ref < items[j].Ref
	})
}

// kpiScore renders a 0..100 completion score for a KPI (100 when nothing is owed,
// i.e. total == 0, matching the scorecard family's "no work, perfect" convention).
func kpiScore(done, total int) float64 {
	if total == 0 {
		return 100
	}
	return scorecard.Round1(100 * float64(done) / float64(total))
}

package qaprocessscore

import "github.com/anthony-chaudhary/fak/pkg/scorecard"

// Schema is the control-pane schema id the qa-process card emits; consumers key on it.
const Schema = "fak-qa-process-scorecard/1"

// DebtKey is the control-pane debt integer for the qa-process card: Σ len(kpi.Defects)
// across its KPIs -- the count of concrete, in-tree-mendable QA-process gaps (a revert with
// no regression test, a skip with no ticket, ...). Written into corpus[DebtKey] by Fold.
const DebtKey = "qa_process_debt"

// Compose folds the qa-process KPIs into the shared control-pane Payload. The card gates on
// HARD debt (a below-the-line QA-process leak is a real, fixable gap), so ok = debt==0.
// KPIs are equal-weight (weights nil); the card is a union of independent process signals,
// none privileged over another.
//
// Today the card wires exactly the KPIs that live in this package (regression_catch, #3841);
// coverage_discipline (#3834) and mutation_efficacy (#3845) append here as they land, and the
// command picks them up without touching the fold.
func Compose(kpis []scorecard.KPI) scorecard.Payload {
	return scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Finding:         "qa-process debt: tests exist and land, but the process leaks -- work the worst-first defect list.",
		FindingClean:    "qa-process signals clean across every wired KPI.",
		NextAction:      "Close each defect: add the missing regression test / ticket the skip, then re-run.",
		NextActionClean: "Hold the line -- keep the ratchets green.",
		Grade:           scorecard.GradeStd,
	})
}

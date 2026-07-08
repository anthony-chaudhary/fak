// spend.go — the cross-account spend rollup shape + its labeling gate (#2903).
//
// Hermes (agent/credential_pool.py, credits_tracker.py) pools spend across accounts
// but relays the provider's own number, which is easy to conflate with what the
// harness itself did. fak's discipline (the conflation scorecard) is that every
// reported figure names WHO AUTHORED it: a value fak counted itself is WITNESSED,
// a value relayed from an external party is OBSERVED — and arithmetic never
// upgrades provenance. This file owns the pure shape: the figure, the rollup, the
// weakest-provenance fold for totals, and the gate that fails any spend figure
// missing its valuation basis or provenance label. The roster fold that produces
// figures lives with the CLI (cmd/fak); this package stays pure.
package metrics

import "fmt"

// SpendProvenance names who authored a spend figure. The set is closed: a figure
// carrying any other token is unlabeled as far as the gate is concerned.
type SpendProvenance string

const (
	// SpendWitnessed marks a figure fak authored itself (e.g. a session count from
	// fak's own watchdog registry). Only a fak-authored number may claim it.
	SpendWitnessed SpendProvenance = "WITNESSED"
	// SpendObserved marks a figure relayed from an external party (e.g. a
	// provider-reported usage-limit state). An OBSERVED value is never a fak claim,
	// and a provider-side charge must never be attributed to a fak action.
	SpendObserved SpendProvenance = "OBSERVED"
)

// SpendRollupSchema stamps the JSON envelope so a reader can bind the shape.
const SpendRollupSchema = "fak-spend-rollup/1"

// SpendFigure is one labeled number in the cross-account rollup. Every field is
// load-bearing for the gate: a figure with an empty ValuationBasis or a
// Provenance outside the closed set fails the rollup.
type SpendFigure struct {
	// Account is the roster tag the figure belongs to; "" means a fleet total.
	Account string `json:"account,omitempty"`
	// Metric names the figure (e.g. "live_sessions", "usage_throttled").
	Metric string  `json:"metric"`
	Value  float64 `json:"value"`
	Unit   string  `json:"unit"`
	// ValuationBasis names what the number is denominated in and where it came
	// from, so a session count can never masquerade as a billed dollar amount.
	ValuationBasis string `json:"valuation_basis"`
	// Provenance is the authorship label: WITNESSED (fak authored) or OBSERVED
	// (external party's number, relayed).
	Provenance SpendProvenance `json:"provenance"`
}

// SpendRollup is the cross-account rollup envelope `fak spend` emits.
type SpendRollup struct {
	Schema  string        `json:"schema"`
	Figures []SpendFigure `json:"figures"`
	// DollarNote states the rollup's dollar honesty in one line (e.g. that no
	// per-account billed-charge source is wired, so no figure is a USD amount).
	DollarNote string `json:"dollar_note,omitempty"`
}

// WeakestSpendProvenance folds the provenance of several inputs into the label
// their aggregate may honestly carry: arithmetic never upgrades authorship, so
// one OBSERVED input makes the total OBSERVED. An empty input set — or any input
// outside the closed set — yields "" (unlabeled), which the gate then refuses;
// a total can never launder an unlabeled term.
func WeakestSpendProvenance(inputs ...SpendProvenance) SpendProvenance {
	if len(inputs) == 0 {
		return ""
	}
	out := SpendWitnessed
	for _, p := range inputs {
		switch p {
		case SpendWitnessed:
		case SpendObserved:
			out = SpendObserved
		default:
			return ""
		}
	}
	return out
}

// GateSpendLabeled is the unlabeled-figure gate (#2903's first slice): it returns
// one defect per spend figure missing its valuation basis or carrying a
// provenance outside the closed WITNESSED/OBSERVED set. An empty result is the
// only pass; callers treat any defect as a hard failure.
func GateSpendLabeled(r SpendRollup) []string {
	var defects []string
	for i, f := range r.Figures {
		id := fmt.Sprintf("figure %d (%s/%s)", i, f.Account, f.Metric)
		if f.Account == "" {
			id = fmt.Sprintf("figure %d (total/%s)", i, f.Metric)
		}
		if f.ValuationBasis == "" {
			defects = append(defects, id+": spend figure carries no valuation basis")
		}
		switch f.Provenance {
		case SpendWitnessed, SpendObserved:
		default:
			defects = append(defects, id+": spend figure is unlabeled — provenance "+
				fmt.Sprintf("%q", string(f.Provenance))+" is not WITNESSED or OBSERVED")
		}
	}
	return defects
}

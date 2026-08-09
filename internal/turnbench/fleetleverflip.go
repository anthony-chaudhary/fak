package turnbench

import (
	"context"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// RungMarginalRow is the corpus-level marginal value of removing one registered
// adjudicator rung. Positive InjectionsAdmitted means harmful sinks that the full
// chain refused became admitted without the rung. DeniesDelta is the candidate
// (rung removed) deny count minus the full-chain deny count.
type RungMarginalRow struct {
	Rung                  string `json:"rung"`
	InjectionsAdmitted    int    `json:"injections_admitted"`
	DeniesDelta           int64  `json:"denies_delta"`
	Cells                 int    `json:"cells"`
	ExactCells            int    `json:"exact_cells"`
	BoundedCells          int    `json:"bounded_cells"`
	NeedsLiveRevalidation bool   `json:"needs_live_revalidation"`
}

// RungMarginalReport prices each registered adjudicator rung over a frozen
// corpus using replay only. A bounded cell credits harmful-sink admissions only
// at or before the first observed divergence; later outcomes are counterfactual
// fiction and require live revalidation.
type RungMarginalReport struct {
	Schema          string            `json:"schema"`
	CorpusSize      int               `json:"corpus_size"`
	ModelCallsSpent int               `json:"model_calls_spent"`
	Rows            []RungMarginalRow `json:"rows"`
}

// RunFleetLeverFlip computes per-rung marginal value over corpus without model
// calls. It uses the configured production adjudicator registry.
func RunFleetLeverFlip(ctx context.Context, corpus []DivHistInput) (*RungMarginalReport, error) {
	agent.Configure()
	return runFleetLeverFlipWithChain(ctx, corpus, abi.Adjudicators())
}

func runFleetLeverFlipWithChain(ctx context.Context, corpus []DivHistInput, chain []abi.Adjudicator) (*RungMarginalReport, error) {
	names := abi.RungNames(chain)
	if len(names) == 0 {
		return nil, fmt.Errorf("turnbench: no addressable adjudicator rungs")
	}
	seen := make(map[string]bool, len(names))
	rows := make([]RungMarginalRow, 0, len(names))
	for _, name := range names {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		rows = append(rows, RungMarginalRow{Rung: name})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Rung < rows[j].Rung })

	for _, in := range corpus {
		if in.Trace == nil {
			return nil, fmt.Errorf("turnbench: nil trace in corpus")
		}
		var policy adjudicator.Policy
		matchedReference := false
		for _, arm := range in.Arms {
			if arm.Name == in.RefName {
				policy = arm.Policy
				matchedReference = true
				break
			}
		}
		if !matchedReference {
			return nil, fmt.Errorf("turnbench: trace %q missing reference arm %q", in.Trace.SliceID, in.RefName)
		}
		baseDisp, baseCounters, err := replayDispositionsCounters(ctx, in.Trace, policy, chain)
		if err != nil {
			return nil, fmt.Errorf("turnbench: baseline replay %q: %w", in.Trace.SliceID, err)
		}
		for i := range rows {
			maskedChain, found := abi.WithoutRung(chain, rows[i].Rung)
			if found == 0 {
				return nil, fmt.Errorf("turnbench: rung %q is not independently maskable", rows[i].Rung)
			}
			candDisp, candCounters, err := replayDispositionsCounters(ctx, in.Trace, policy, maskedChain)
			if err != nil {
				return nil, fmt.Errorf("turnbench: replay %q without %s: %w", in.Trace.SliceID, rows[i].Rung, err)
			}
			frontier := firstObservedDivergence(baseDisp, candDisp)
			rows[i].Cells++
			if frontier < 0 {
				rows[i].ExactCells++
			} else {
				rows[i].BoundedCells++
				rows[i].NeedsLiveRevalidation = true
			}
			rows[i].DeniesDelta += candCounters.Denies - baseCounters.Denies
			for callIdx, call := range in.Trace.Calls {
				if !isHarmfulSink(call) || callIdx >= len(baseDisp) || callIdx >= len(candDisp) {
					continue
				}
				if frontier >= 0 && callIdx > frontier {
					continue
				}
				if dispositionRefuses(baseDisp[callIdx]) && !dispositionRefuses(candDisp[callIdx]) {
					rows[i].InjectionsAdmitted++
				}
			}
		}
	}

	return &RungMarginalReport{
		Schema: "fak.turnbench.rung-marginal.v1", CorpusSize: len(corpus),
		ModelCallsSpent: 0, Rows: rows,
	}, nil
}

func dispositionRefuses(d CallDisposition) bool {
	return d.Class == "deny" || d.Class == "quarantine"
}

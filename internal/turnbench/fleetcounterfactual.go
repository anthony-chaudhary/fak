// fleetcounterfactual.go — the FLEET COUNTERFACTUAL REPLAY: re-adjudicate a recorded
// CORPUS (many traces) against candidate policies at $0 model spend, and report, from
// real k.Syscall counters, the SECURITY-FLOOR coverage across the whole population
// (issue #502, the headline coverage/$ artifact).
//
// WHY THIS IS THE SECURITY-FLOOR ANALOG OF THE TURN-TAX REPLAY. The turn-tax replay
// (policyreplay.go / divhist.go) collapses a K-policy efficiency comparison from a
// product (K full agent+model runs) to a sum (1 recording + K model-free replays). That
// story lives in the engine-OWNER regime: the savings are vDSO/grammar elisions only a
// kernel that intercepts the model's syscalls can book. This file scores the OTHER axis —
// the SECURITY FLOOR (quarantines, denies, transforms) — which is ENGINE-AGNOSTIC: the
// floor is the same whether the model is your SGLang deploy or a frontier API you only
// consume. So this report works in the API-CONSUMER regime where the turn-tax story does
// not, and it answers the policy-development question directly: given a recorded corpus +
// a candidate policy, how many injections would it have quarantined, how many calls would
// it have denied/transformed, ACROSS the population — with zero model calls.
//
// It turns policy development from "deploy and pray" into "replay and prove": a candidate
// policy's floor coverage is a deterministic function of (the recorded calls, the policy),
// so it is re-derivable from a frozen corpus without spending a single model turn.
//
// THE HONESTY SPLIT (the load-bearing half — carried verbatim from policyreplay.go's
// gate). The floor counters (Quarantines / Denies / Transforms) are MEASURED — they are
// real kernel events for EVERY recorded call, so they are reported for every cell. But a
// candidate policy that denies/quarantines a call the recorded run served (or serves one
// it denied) would have made the live model OBSERVE a different result and BRANCH; every
// recorded call after that divergence is counterfactual. So:
//
//   - VERDICT / FLOOR COVERAGE is reported for ALL cells (exact and bounded alike).
//   - RESOLVE-RATE (task completion past a divergence) is reported ONLY for EXACT cells.
//     A bounded cell's resolve-rate is fiction the frozen trace cannot produce; it is
//     FLAGGED "needs live re-run from frontier", never aggregated into a headline number.
//
// $0 MODEL. The whole report is built on RunPolicyReplay, which is model-free replay end
// to end (real k.Syscall, no engine decode). This driver adds NO model call — it only
// re-aggregates the per-arm counters and divergence witness RunPolicyReplay already
// produces across the corpus population. The replay path's model-free-ness is asserted in
// the tests (zero engine-decode is the existing replay's invariant, reused here).
//
// MECHANISM. For each corpus trace, score its candidate arms through RunPolicyReplay (a
// scoreWorld-style model-free replay on a fresh kernel.New per arm). Roll the per-arm
// floor counters up BY POLICY (arm name) across the population, and label each
// (trace, arm) cell exact|bounded@i from the per-arm Replayability witness. The result is
// a per-policy floor-coverage row plus a per-cell label table, all at $0 model spend.
//
// SCOPE / FENCES. This is ADDITIVE: it builds on the EXPORTED RunPolicyReplay,
// PolicyArm, PolicyArmResult, Trace, KernelCounters, DivHistInput — no fork of the kernel
// replay path. The reference arm of each trace replays itself (exact by construction) and
// is EXCLUDED from the cell population, exactly as divhist.go excludes it, so the coverage
// fraction reflects only the CANDIDATE comparisons. Like RunPolicyReplay this is a
// corpus-serial loop over per-trace concurrent fan-outs; it is NOT safe to interleave with
// another replay in the same process.
package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// FloorCounters is the security-floor slice of the kernel counters: the three verdict
// classes that are the engine-agnostic floor (a quarantined injection, a denied call, a
// transformed/redacted call). These are MEASURED — real k.Syscall events for every
// recorded call — so they are reported for every cell regardless of divergence.
type FloorCounters struct {
	Quarantines int64 `json:"quarantines"` // poisoned results held out of context (injections caught)
	Denies      int64 `json:"denies"`      // calls refused by the capability/IFC floor
	Transforms  int64 `json:"transforms"`  // calls rewritten before dispatch (redact, grammar)
}

// add folds another cell's floor counters into this running total.
func (f *FloorCounters) add(o FloorCounters) {
	f.Quarantines += o.Quarantines
	f.Denies += o.Denies
	f.Transforms += o.Transforms
}

// floorOf projects a cell's full kernel counters onto the security-floor slice.
func floorOf(kc KernelCounters) FloorCounters {
	return FloorCounters{Quarantines: kc.Quarantines, Denies: kc.Denies, Transforms: kc.Transforms}
}

// FloorCell is one (trace, candidate arm) cell of the counterfactual table: the floor
// counters that candidate produced on that recorded trace, and the divergence label.
// Floor is MEASURED for every cell; Replayability tells the reader whether a resolve-rate
// past the divergence frontier would be sound (exact) or counterfactual (bounded@i).
type FloorCell struct {
	Trace         string        `json:"trace"`          // the recorded trace's slice id
	Policy        string        `json:"policy"`         // the candidate arm name
	Floor         FloorCounters `json:"floor_counters"` // MEASURED: real kernel events on this trace
	Replayability string        `json:"replayability"`  // "exact" | "bounded@<i>"
	Exact         bool          `json:"exact"`          // true iff resolve-rate would replay soundly
	// ResolveRateReportable is the honesty flag: true ONLY for an exact cell, where a
	// resolve-rate / completion number past the recorded frontier replays soundly. A
	// bounded cell sets this false and carries ResolveRateNote — its resolve-rate is
	// counterfactual and is NEVER aggregated into a headline.
	ResolveRateReportable bool   `json:"resolve_rate_reportable"`
	ResolveRateNote       string `json:"resolve_rate_note,omitempty"`
}

// PolicyFloorCoverage is one candidate policy's floor coverage rolled up across the whole
// corpus population: the summed floor counters over EVERY cell the policy was scored on,
// the exact/bounded cell split, and the honesty fence on resolve-rate.
type PolicyFloorCoverage struct {
	Policy string `json:"policy"` // the candidate arm name

	// Floor is the population total of the security-floor counters — MEASURED across ALL
	// cells (exact and bounded), because every counter is a real kernel event on a
	// recorded call. This is the honest "across the corpus, this policy would have
	// quarantined N injections / denied M calls / transformed P calls" headline.
	Floor FloorCounters `json:"floor_counters"`

	Cells        int `json:"cells"`         // (trace, this policy) cells scored
	ExactCells   int `json:"exact_cells"`   // cells that replayed exactly (resolve-rate sound)
	BoundedCells int `json:"bounded_cells"` // cells that diverged (resolve-rate counterfactual)

	// ResolveRateReportable is true iff EVERY cell for this policy is exact — only then
	// would a population resolve-rate be sound. When any cell is bounded it is false and
	// ResolveRateNote names how many cells need a live re-run. The floor coverage above
	// stands regardless; this fence governs the resolve-rate axis ALONE.
	ResolveRateReportable bool   `json:"resolve_rate_reportable"`
	ResolveRateNote       string `json:"resolve_rate_note,omitempty"`
}

// FleetCounterfactualReport is the issue-#502 artifact: per-candidate-policy security-floor
// coverage across a recorded corpus, computed at $0 model spend (pure replay). The
// per-policy rows carry the MEASURED floor totals (reported for all cells); the per-cell
// table carries the exact|bounded label and the resolve-rate honesty flag for every
// (trace, candidate) cell. Deterministic for a fixed corpus — a regenerable artifact, not
// a sample.
type FleetCounterfactualReport struct {
	Provenance Provenance `json:"provenance"`
	Cost       CostModel  `json:"cost_model"`

	Traces   int `json:"traces"`   // corpus traces scored
	Policies int `json:"policies"` // distinct candidate policies (non-reference arm names)
	Cells    int `json:"cells"`    // total (trace, candidate) cells across the population

	ExactCells    int     `json:"exact_cells"`    // cells whose resolve-rate replays soundly
	BoundedCells  int     `json:"bounded_cells"`  // cells whose resolve-rate is counterfactual
	ExactFraction float64 `json:"exact_fraction"` // ExactCells / Cells (0 cells => 1.0)

	// $0 model accounting. A naive per-policy floor audit would re-run the whole
	// agent+model once per (trace, policy) cell; this report runs each trace's recording
	// ONCE and scores every candidate as a model-free kernel replay, so ModelCallsSpent
	// is ZERO — the coverage/$ headline.
	ModelCallsSpent int64 `json:"model_calls_spent"` // 0 — the whole point (replay is model-free)
	ReplayWallNs    int64 `json:"replay_wall_ns"`    // MEASURED: summed per-trace replay wall spans

	// PopulationFloor is the corpus-wide floor total summed over EVERY cell — the
	// one-line "across the whole population, the candidate policies would have caught
	// this much floor" number. MEASURED.
	PopulationFloor FloorCounters `json:"population_floor"`

	// Coverage is the per-policy roll-up (sorted by policy name for a stable artifact).
	Coverage []PolicyFloorCoverage `json:"coverage"`

	// Cells is the per-(trace, candidate) label table (sorted by trace then policy).
	CellTable []FloorCell `json:"cell_table"`
}

// JSON renders the report (stable indentation, trailing newline) for an artifact file —
// mirrors PolicyReplayReport.JSON().
func (r *FleetCounterfactualReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// boundedResolveNote is the standard flag a bounded cell carries on the resolve-rate axis:
// the floor counters are real, but completion past the divergence frontier is fiction the
// frozen trace cannot produce and must be re-run live.
func boundedResolveNote(replayability string) string {
	return fmt.Sprintf(
		"resolve-rate NOT reportable: this cell is %s — a live run would branch at the "+
			"divergence frontier, so task-completion past that call needs a live re-run "+
			"from the frontier; the floor counters above are real kernel events and stand",
		replayability)
}

// RunFleetCounterfactual scores a recorded corpus against its candidate policies as
// model-free replay and reports the per-policy SECURITY-FLOOR coverage across the whole
// population (issue #502). Each corpus entry is a (trace, arms, refName) tuple — the same
// DivHistInput shape divhist.go uses — so a corpus is simply a set of recorded traces
// (e.g. from internal/tracesink) each paired with the candidate policies to score.
//
// For each trace it calls RunPolicyReplay (the blessed model-free replay path — a fresh
// kernel.New per arm, real k.Syscall, no engine decode), then:
//
//   - rolls the per-arm floor counters (Quarantines/Denies/Transforms) up BY POLICY
//     across every trace — MEASURED, reported for ALL cells; and
//   - labels each (trace, candidate) cell exact|bounded@i from the per-arm divergence
//     witness, and reports resolve-rate as reportable ONLY for exact cells.
//
// The reference arm of each trace is EXCLUDED from the cell population (it replays itself,
// exact by construction), exactly as divhist.go excludes it, so the coverage reflects only
// the CANDIDATE comparisons. The report is deterministic for a fixed corpus.
//
// Like RunPolicyReplay / RunDivergenceHistogram this is NOT safe to interleave with
// another replay in the same process.
func RunFleetCounterfactual(ctx context.Context, corpus []DivHistInput, cm CostModel) (*FleetCounterfactualReport, error) {
	if len(corpus) == 0 {
		return nil, fmt.Errorf("turnbench: RunFleetCounterfactual needs a non-empty corpus")
	}
	cm = withCostModelVersion(cm)

	// Per-policy accumulators keyed by candidate arm name.
	type acc struct {
		floor             FloorCounters
		cells, exact, bnd int
	}
	byPolicy := map[string]*acc{}
	var cells []FloorCell
	var population FloorCounters
	var totalExact, totalBounded int
	var wallNs int64

	for ci, in := range corpus {
		if in.Trace == nil || len(in.Trace.Calls) == 0 {
			return nil, fmt.Errorf("turnbench: corpus entry %d has an empty trace", ci)
		}
		rep, err := RunPolicyReplay(ctx, in.Trace, in.Arms, in.RefName, cm)
		if err != nil {
			return nil, fmt.Errorf("turnbench: corpus entry %d (%s): %w", ci, in.Trace.SliceID, err)
		}
		wallNs += rep.ReplayWallNs

		ref := rep.Reference
		for _, a := range rep.Arms {
			// The reference arm replays itself (exact by construction); exclude it from
			// the cell population so coverage reflects only candidate policies.
			if a.Name == ref {
				continue
			}
			fc := floorOf(a.Counters)
			population.add(fc)

			ac := byPolicy[a.Name]
			if ac == nil {
				ac = &acc{}
				byPolicy[a.Name] = ac
			}
			ac.floor.add(fc)
			ac.cells++

			cell := FloorCell{
				Trace:         in.Trace.SliceID,
				Policy:        a.Name,
				Floor:         fc,
				Replayability: a.Replayability,
				Exact:         a.FirstDivergence < 0,
			}
			if cell.Exact {
				ac.exact++
				totalExact++
				cell.ResolveRateReportable = true
			} else {
				ac.bnd++
				totalBounded++
				cell.ResolveRateReportable = false
				cell.ResolveRateNote = boundedResolveNote(a.Replayability)
			}
			cells = append(cells, cell)
		}
	}

	// Roll the per-policy accumulators into sorted coverage rows. Resolve-rate is
	// reportable for a policy ONLY when EVERY one of its cells is exact.
	coverage := make([]PolicyFloorCoverage, 0, len(byPolicy))
	for name, ac := range byPolicy {
		row := PolicyFloorCoverage{
			Policy:                name,
			Floor:                 ac.floor,
			Cells:                 ac.cells,
			ExactCells:            ac.exact,
			BoundedCells:          ac.bnd,
			ResolveRateReportable: ac.bnd == 0,
		}
		if ac.bnd > 0 {
			row.ResolveRateNote = fmt.Sprintf(
				"resolve-rate NOT reportable for this policy: %d of %d cells are bounded "+
					"(diverged from the recorded trajectory) and need a live re-run from the "+
					"frontier; the floor counters above are MEASURED across all %d cells and stand",
				ac.bnd, ac.cells, ac.cells)
		}
		coverage = append(coverage, row)
	}
	sort.Slice(coverage, func(i, j int) bool { return coverage[i].Policy < coverage[j].Policy })

	// Stable cell table ordering: trace then policy.
	sort.Slice(cells, func(i, j int) bool {
		if cells[i].Trace != cells[j].Trace {
			return cells[i].Trace < cells[j].Trace
		}
		return cells[i].Policy < cells[j].Policy
	})

	totalCells := totalExact + totalBounded
	frac := 1.0
	if totalCells > 0 {
		frac = float64(totalExact) / float64(totalCells)
	}

	return &FleetCounterfactualReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunFleetCounterfactual",
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (fleet counterfactual replay)",
		},
		Cost:            cm,
		Traces:          len(corpus),
		Policies:        len(byPolicy),
		Cells:           totalCells,
		ExactCells:      totalExact,
		BoundedCells:    totalBounded,
		ExactFraction:   frac,
		ModelCallsSpent: 0, // the whole point: replay is model-free, $0 model spend
		ReplayWallNs:    wallNs,
		PopulationFloor: population,
		Coverage:        coverage,
		CellTable:       cells,
	}, nil
}

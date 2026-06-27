package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
)

// marshalArtifact renders v as stable-indented JSON with a trailing newline — the
// canonical artifact encoding shared by every report/sweep JSON() method in this
// package (FanoutSweep, FleetSweep, ParityReport, TopologySearchReport,
// DivergenceHistogramReport, …). The MarshalIndent error is intentionally dropped:
// these report structs are always JSON-encodable by construction.
func marshalArtifact(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return append(b, '\n')
}

// deriveSeeds derives a fixed stream of trials per-trial seeds from a single root
// seed. Each caller composes its own root seed (folding in the cell coordinates it
// varies over) and then draws the per-trial seeds here, so the trial order is fixed
// and independent of how many draws each trial body consumes.
func deriveSeeds(rootSeed int64, trials int) []int64 {
	root := rand.New(rand.NewSource(rootSeed))
	seeds := make([]int64, trials)
	for i := range seeds {
		seeds[i] = root.Int63()
	}
	return seeds
}

// paretoFrontierBy returns the Pareto-non-dominated members of cands: a candidate is on
// the frontier iff no OTHER candidate dominates it. `dominates(a, b)` reports whether a
// strictly dominates b (>=/<= on every axis, with strict inequality on at least one);
// `less` is the stable sort applied to the frontier for a regenerable artifact. The
// shared frontier skeleton behind paretoFrontier (injections-vs-denies) and
// topoParetoFrontier (savings-vs-collision) — only the per-axis dominance test and the
// sort key differ between the two, threaded here as the two closures.
func paretoFrontierBy[C any](cands []C, dominates func(a, b C) bool, less func(a, b C) bool) []C {
	var front []C
	for i := range cands {
		dominated := false
		for j := range cands {
			if i == j {
				continue
			}
			if dominates(cands[j], cands[i]) {
				dominated = true
				break
			}
		}
		if !dominated {
			front = append(front, cands[i])
		}
	}
	sort.Slice(front, func(i, j int) bool { return less(front[i], front[j]) })
	return front
}

// stampFrontier marks every candidate in cands whose name appears in `frontier` by
// flipping its on-frontier flag via setOnFrontier. The shared OnFrontier-stamping step of
// the search reports (policysearch, toposearch): the frontier copies are re-derived AFTER
// this so they carry the stamped flag too.
func stampFrontier[C any](cands []C, frontier []C, name func(C) string, setOnFrontier func(*C, bool)) {
	frontierNames := map[string]bool{}
	for _, fc := range frontier {
		frontierNames[name(fc)] = true
	}
	for i := range cands {
		setOnFrontier(&cands[i], frontierNames[name(cands[i])])
	}
}

// bestCandidate returns the single best member of cands under `better` (a candidate
// strictly beats the incumbent, ties already broken inside `better`). cands must be
// non-empty. The shared best-pick loop of the search reports.
func bestCandidate[C any](cands []C, better func(c, incumbent C) bool) C {
	best := cands[0]
	for _, c := range cands[1:] {
		if better(c, best) {
			best = c
		}
	}
	return best
}

// candidateByName returns the candidate in cands whose name equals want, and whether one
// was found. The shared baseline-refresh lookup of the search reports (re-reading the
// "baseline" copy out of the OnFrontier-stamped slice so its flag is set).
func candidateByName[C any](cands []C, name func(C) string, want string) (C, bool) {
	for i := range cands {
		if name(cands[i]) == want {
			return cands[i], true
		}
	}
	var zero C
	return zero, false
}

// topKNeedsRevalidation returns the names of the first topK frontier candidates flagged
// NeedsLiveRevalidation. topK is clamped to len(frontier) (and treated as the full
// frontier when non-positive). The shared top-k live-revalidation flagging of the search
// reports — a FLAG, never an executed model run.
func topKNeedsRevalidation[C any](frontier []C, topK int, name func(C) string, needsReval func(C) bool) []string {
	if topK <= 0 || topK > len(frontier) {
		topK = len(frontier)
	}
	var flagged []string
	for i := 0; i < topK; i++ {
		if needsReval(frontier[i]) {
			flagged = append(flagged, name(frontier[i]))
		}
	}
	return flagged
}

// aliasConvertArgs builds an aliased convert_currency call's args: it picks one of the
// {from/to, source/target} alias spellings deterministically from rng, draws an amount in
// [50, 949], and marshals {alias_a:"USD", alias_b:"EUR", amount}. It returns the chosen
// alias pair too (callers stamp it into a human note). The shared grammar-alias arg
// builder behind the fleet and stochastic call generators.
func aliasConvertArgs(rng *rand.Rand) (aliasPair, json.RawMessage) {
	ap := convertAliasPairs[rng.Intn(len(convertAliasPairs))]
	amt := 50 + rng.Intn(900)
	args, _ := json.Marshal(map[string]any{ap.a: "USD", ap.b: "EUR", "amount": amt})
	return ap, json.RawMessage(args)
}

// convertAliasPairs is the shared deterministic table of from/to alias spellings used by
// aliasConvertArgs (the grammar-equivalence lever both call generators vary over).
var convertAliasPairs = []aliasPair{{"from", "to"}, {"source", "target"}}

// replayCorpusEntry validates corpus entry ci and scores it through RunPolicyReplay,
// wrapping any error with the entry index (and slice id) — the shared per-entry
// preamble of the corpus-loop callers (RunDivergenceHistogram, RunFleetCounterfactual).
func replayCorpusEntry(ctx context.Context, ci int, in DivHistInput, cm CostModel) (*PolicyReplayReport, error) {
	if in.Trace == nil || len(in.Trace.Calls) == 0 {
		return nil, fmt.Errorf("turnbench: corpus entry %d has an empty trace", ci)
	}
	rep, err := RunPolicyReplay(ctx, in.Trace, in.Arms, in.RefName, cm)
	if err != nil {
		return nil, fmt.Errorf("turnbench: corpus entry %d (%s): %w", ci, in.Trace.SliceID, err)
	}
	return rep, nil
}

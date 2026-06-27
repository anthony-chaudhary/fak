package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
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

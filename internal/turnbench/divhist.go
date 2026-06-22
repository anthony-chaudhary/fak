// divhist.go — the DIVERGENCE-RATE HISTOGRAM over a corpus of traces × candidate
// policies (issue #501(b)). This is the report that sets the TRUE exact-cell fraction:
// no headline "K-policy comparison for the price of one" multiple is honestly quotable
// until you know, over a real corpus, what fraction of (trace, policy) cells actually
// replay EXACTLY (resolve-rate sound) vs come out BOUNDED (verdict-only). This file
// rolls the per-arm divergence witness RunPolicyReplay already produces up into:
//
//   - the DISTRIBUTION of first_divergence indices across every non-reference cell, and
//   - the EXACT-CELL FRACTION (exact cells / total cells).
//
// It is pure model-free replay — cheap and deterministic, so the report is a regenerable
// ARTIFACT, not a sample: the same corpus yields byte-identical numbers every run. The
// load-bearing control is the no-divergence corpus (every candidate == the reference),
// which MUST read 100% exact — the witness can never manufacture a divergence where the
// policy did not change (the analogue of the turn-tax happy-path=0 control).
//
// A "cell" is one (trace, non-reference arm) pair: the reference arm replays itself and
// is excluded from the rate (it is exact by construction and would only inflate the
// fraction). Each cell is EXACT (FirstDivergence < 0) or BOUNDED@i (its first_divergence
// index is binned into the histogram).
package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// DivHistInput is one corpus entry: a frozen trace plus the candidate policy arms to
// score against it and the reference arm name (the recorded trajectory's policy). The
// reference arm itself is not counted as a cell.
type DivHistInput struct {
	Trace   *Trace
	Arms    []PolicyArm
	RefName string
}

// DivHistBin is one bucket of the first_divergence distribution: how many bounded cells
// first diverged at this call index.
type DivHistBin struct {
	Index int `json:"first_divergence_index"`
	Cells int `json:"cells"`
}

// DivergenceHistogramReport is the regenerable artifact: over Traces traces and Cells
// scored (trace, non-reference arm) pairs, the count of exact vs bounded cells, the
// exact-cell fraction, and the distribution of first_divergence indices for the bounded
// cells. Deterministic for a fixed corpus.
type DivergenceHistogramReport struct {
	Provenance Provenance `json:"provenance"`

	Traces int `json:"traces"`         // number of corpus traces scored
	Arms   int `json:"candidate_arms"` // non-reference arms summed across traces (== Cells)
	Cells  int `json:"cells"`          // total (trace, non-reference arm) cells

	ExactCells    int     `json:"exact_cells"`    // cells that replayed EXACTLY (resolve-rate sound)
	BoundedCells  int     `json:"bounded_cells"`  // cells that diverged (verdict-only past the frontier)
	ExactFraction float64 `json:"exact_fraction"` // ExactCells / Cells (0 cells => 1.0, vacuously exact)

	// FirstDivergence is the distribution of first_divergence call indices over the
	// BOUNDED cells, sorted by index ascending. Exact cells (no divergence) are not
	// binned here — they are counted in ExactCells. The bins sum to BoundedCells.
	FirstDivergence []DivHistBin `json:"first_divergence_histogram"`
}

// JSON renders the report (stable indentation, trailing newline) for an artifact file.
func (r *DivergenceHistogramReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// RunDivergenceHistogram scores each corpus trace through RunPolicyReplay and rolls the
// per-arm divergence witness up into the first_divergence distribution and the exact-cell
// fraction over the whole corpus. It is model-free replay end to end, so the report is a
// deterministic, regenerable artifact: the same corpus yields identical numbers.
//
// The reference arm of each trace is excluded from the cell count (it is exact against
// itself by construction). A no-divergence corpus — every candidate arm identical to the
// reference — therefore reads exactly 100% exact, the anti-inflation control.
//
// Like RunPolicyReplay, this is NOT safe to call concurrently with another replay in the
// same process: it swaps the process-global monitor policy per arm.
func RunDivergenceHistogram(ctx context.Context, corpus []DivHistInput, cm CostModel) (*DivergenceHistogramReport, error) {
	if len(corpus) == 0 {
		return nil, fmt.Errorf("turnbench: RunDivergenceHistogram needs a non-empty corpus")
	}
	cm = withCostModelVersion(cm)

	bins := map[int]int{}
	var cells, exact, bounded int

	for ci, in := range corpus {
		if in.Trace == nil || len(in.Trace.Calls) == 0 {
			return nil, fmt.Errorf("turnbench: corpus entry %d has an empty trace", ci)
		}
		rep, err := RunPolicyReplay(ctx, in.Trace, in.Arms, in.RefName, cm)
		if err != nil {
			return nil, fmt.Errorf("turnbench: corpus entry %d (%s): %w", ci, in.Trace.SliceID, err)
		}
		// The reference arm replays itself (exact by construction); exclude it from the
		// cell count so the fraction reflects only the CANDIDATE comparisons.
		ref := rep.Reference
		for _, a := range rep.Arms {
			if a.Name == ref {
				continue
			}
			cells++
			if a.FirstDivergence < 0 {
				exact++
				continue
			}
			bounded++
			bins[a.FirstDivergence]++
		}
	}

	hist := make([]DivHistBin, 0, len(bins))
	for idx, n := range bins {
		hist = append(hist, DivHistBin{Index: idx, Cells: n})
	}
	sort.Slice(hist, func(i, j int) bool { return hist[i].Index < hist[j].Index })

	// 0 cells is vacuously 100% exact (no candidate broke the trajectory).
	frac := 1.0
	if cells > 0 {
		frac = float64(exact) / float64(cells)
	}

	return &DivergenceHistogramReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunDivergenceHistogram",
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (divergence-rate histogram)",
		},
		Traces:          len(corpus),
		Arms:            cells,
		Cells:           cells,
		ExactCells:      exact,
		BoundedCells:    bounded,
		ExactFraction:   frac,
		FirstDivergence: hist,
	}, nil
}

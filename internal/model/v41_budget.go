package model

import (
	"errors"
	"fmt"
)

// Published, measured DeepSeek-V4.1-Flash checkpoint facts. These are cited
// byte counts, not derived estimates. Sources:
//   - Official 48 weight shards total 510,296,708,312 bytes, recorded in
//     docs/deepseek/v4-flash-native.md for DeepSeek-V4.1-Flash at the pinned
//     revision DeepSeekV41FlashRevision (dba1be0a40aa45a94ad051997016db3960a90277).
//   - Engram (the two embedding tables for layer IDs {1,14}) measured footprint
//     202,758,032,400 bytes, recorded in the issue body.
const (
	// DeepSeekV41CheckpointBytes is the published total size of the official
	// DeepSeek-V4.1-Flash weight shards (48 files).
	DeepSeekV41CheckpointBytes int64 = 510296708312

	// DeepSeekV41EngramBytes is the published total footprint of the two Engram
	// embedding tables.
	DeepSeekV41EngramBytes int64 = 202758032400

	// DeepSeekV41DefaultResidentFraction is the documented streamed-residency
	// fraction used for the published per-node example. 1.0 means full resident;
	// 0.25 means a quarter of the checkpoint is held resident and the remainder
	// is streamed from disk.
	DeepSeekV41DefaultResidentFraction = 0.25
)

// ErrV41BudgetExceeded is returned when a checkpoint would not fit under a
// per-node admission budget. It is a placement/admission refusal, not a claim
// about distributed execution or throughput.
var ErrV41BudgetExceeded = errors.New("model: DeepSeek V4.1 checkpoint exceeds per-node budget")

// DeepSeekV41NodeBudget is a per-node placement budget: how much of the
// checkpoint a single node must be able to hold resident under a given
// resident fraction. It is an admission/planning budget, not a
// distributed-execution witness.
//
// Backbone and Engram are accounted separately so a caller can see the
// embedding share explicitly: EngramResidentBytes + BackboneResidentBytes always
// equals ResidentBytes.
type DeepSeekV41NodeBudget struct {
	Nodes                 int
	CheckpointBytes       int64
	EngramBytes           int64
	BackboneBytes         int64   // checkpoint minus Engram
	ResidentFraction      float64 // 1.0 = full resident, e.g. 0.25 = stream 75%
	ResidentBytes         int64   // per-node resident bytes for the given fraction
	EngramResidentBytes   int64
	BackboneResidentBytes int64
	Fits                  bool
}

// DeepSeekV41BudgetForNodes returns the per-node placement budget for a
// checkpoint replicated across `nodes` nodes at the given resident fraction.
//
// Invalid inputs (nodes < 1, residentFraction <= 0, residentFraction > 1, a
// non-positive backbone) return the zero-value budget with Fits=false; this
// function never panics.
//
// For residentFraction == 1.0 the full checkpoint is split across nodes:
// ResidentBytes = ceil(CheckpointBytes/nodes), EngramResidentBytes =
// ceil(EngramBytes/nodes), and BackboneResidentBytes = Resident - Engram.
// For 0 < residentFraction < 1.0 only that fraction is budgeted resident and
// the remainder is assumed streamed from disk:
// ResidentBytes = ceil(CheckpointBytes * residentFraction / nodes), with the
// Engram/backbone split scaled the same way so the two shares still sum to
// ResidentBytes.
func DeepSeekV41BudgetForNodes(nodes int, residentFraction float64) DeepSeekV41NodeBudget {
	if nodes < 1 {
		return DeepSeekV41NodeBudget{}
	}
	if !(residentFraction > 0) || residentFraction > 1 {
		return DeepSeekV41NodeBudget{}
	}
	backbone := DeepSeekV41CheckpointBytes - DeepSeekV41EngramBytes
	if backbone <= 0 {
		return DeepSeekV41NodeBudget{}
	}
	b := DeepSeekV41NodeBudget{
		Nodes:            nodes,
		CheckpointBytes:  DeepSeekV41CheckpointBytes,
		EngramBytes:      DeepSeekV41EngramBytes,
		BackboneBytes:    backbone,
		ResidentFraction: residentFraction,
	}
	b.ResidentBytes = ceilDivScaled(DeepSeekV41CheckpointBytes, residentFraction, nodes)
	b.EngramResidentBytes = ceilDivScaled(DeepSeekV41EngramBytes, residentFraction, nodes)
	b.BackboneResidentBytes = b.ResidentBytes - b.EngramResidentBytes
	b.Fits = b.ResidentBytes > 0 && b.BackboneResidentBytes >= 0
	return b
}

// ceilDivScaled returns ceil(total * fraction / nodes) using integer arithmetic
// so the published byte counts stay exact and deterministic. fraction is
// assumed to be in (0,1]; it is expressed as an exact rational so 0.25 does not
// introduce binary floating-point drift.
func ceilDivScaled(total int64, fraction float64, nodes int) int64 {
	num, den := fractionRatio(fraction)
	div := int64(nodes) * den
	return (total*num + div - 1) / div
}

// fractionRatio converts a resident fraction in (0,1] to an exact positive
// numerator/denominator pair. The documented fractions (1.0, 0.25) and any
// fraction with a small exact binary expansion resolve exactly; other values
// fall back to a fine-denominator rational that is provably not smaller than
// the requested fraction, keeping the budget conservative (it never under-states
// the resident bytes).
func fractionRatio(fraction float64) (int64, int64) {
	for _, den := range []int64{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} {
		num := int64(fraction * float64(den))
		if num > 0 && float64(num)/float64(den) >= fraction {
			return num, den
		}
	}
	// Conservative fallback: round up to the nearest 1/1024.
	num := int64(fraction * 1024)
	if float64(num)/1024 < fraction {
		num++
	}
	if num < 1 {
		num = 1
	}
	return num, 1024
}

// Describe renders the budget readably for placement logs and test receipts.
func (b DeepSeekV41NodeBudget) Describe() string {
	return fmt.Sprintf(
		"V4.1 per-node budget: nodes=%d fraction=%.4g resident=%d engram_resident=%d backbone_resident=%d checkpoint=%d engram=%d backbone=%d fits=%t",
		b.Nodes, b.ResidentFraction, b.ResidentBytes, b.EngramResidentBytes, b.BackboneResidentBytes,
		b.CheckpointBytes, b.EngramBytes, b.BackboneBytes, b.Fits)
}

// AdmitDeepSeekV41Checkpoint reports whether a checkpoint of checkpointBytes can
// be admitted under a per-node budget. It refuses before any large allocation.
func AdmitDeepSeekV41Checkpoint(checkpointBytes int64, budget DeepSeekV41NodeBudget) error {
	if !budget.Fits || budget.ResidentBytes <= 0 {
		return fmt.Errorf("%w: invalid budget (fits=%t resident=%d)", ErrV41BudgetExceeded, budget.Fits, budget.ResidentBytes)
	}
	if checkpointBytes > budget.ResidentBytes {
		return fmt.Errorf("%w: checkpoint=%d exceeds per-node resident=%d for nodes=%d fraction=%.4g",
			ErrV41BudgetExceeded, checkpointBytes, budget.ResidentBytes, budget.Nodes, budget.ResidentFraction)
	}
	return nil
}

// Per-scheme rounding-floor receipts for KV-cache quantization quality (#14257).
//
// The budget evaluator answers "does this candidate stay within the requested
// drift budget?". This file answers the prior question: how much error does the
// scheme itself pay on real data? A layer that compresses its own state (window
// KV, compressed latent, indexer) diverges from an unquantized baseline by the
// scheme's intrinsic round-trip floor even when nothing is wrong, so
// port-vs-reference divergence is only meaningful relative to that floor.
//
// Method (clean-room extraction, cite-and-reimplement): capture the true KV
// vectors from the reference graph, apply the reference's own activation
// quantizers as a round trip, and measure correlation / relative error /
// max|d| against the original. The floor is a property of the scheme applied to
// real data, not of any implementation: a faithful port pays at least this
// much, and a port that keeps the state in f16 pays none of it.
//
// Source pin: JigSawPT/deepseek-v41-flash-on-5090 (MIT),
// tools/kv_floor.py:63-81@21d29fdb275191bbd5d288c3729324c7b16d2599, re-verified
// unchanged at its registry pin on 2026-11-08. No upstream code is vendored or
// executed; the three quoted floors are upstream-measured priors labeled
// upstream_quoted_unverified, never appliance measurements.
package kvquantquality

import (
	"errors"
	"fmt"
	"strings"
)

// FloorContractVersion stamps per-scheme rounding-floor receipts. It is
// distinct from ContractVersion: a floor receipt records what a scheme owes,
// while a v1 report adjudicates a candidate against a requested budget.
const FloorContractVersion = "fak.kvquant-quality.floor/v1"

// FloorRole names which state a scheme compresses. Floors are per-role because
// a layer that compresses its own state pays more than one that does not: the
// same precision applied to window KV, a compressed latent, and an indexer q/k
// cache measures different floors on the same checkpoint.
type FloorRole string

const (
	// FloorRoleWindowKV is the rolling window KV cache (e.g. fp8 block-32, ue8m0 scales).
	FloorRoleWindowKV FloorRole = "window_kv"
	// FloorRoleLatent is a compressed latent state (e.g. fp4 block-16, e4m3 scales).
	FloorRoleLatent FloorRole = "latent"
	// FloorRoleIndexer is the indexer q/k cache (e.g. fp4 block-32).
	FloorRoleIndexer FloorRole = "indexer"
)

// knownFloorRoles is the closed role vocabulary; anything else fails closed.
var knownFloorRoles = map[FloorRole]bool{
	FloorRoleWindowKV: true,
	FloorRoleLatent:   true,
	FloorRoleIndexer:  true,
}

// FloorEvidenceKind designates where a floor's statistics came from.
type FloorEvidenceKind string

const (
	// FloorEvidenceUpstreamQuoted marks fixture data quoted from an upstream
	// study. It is modeled evidence: the numbers are priors, not measurements
	// on this appliance, and must never be cited as hardware results.
	FloorEvidenceUpstreamQuoted FloorEvidenceKind = "upstream_quoted_unverified"
	// FloorEvidenceObservedHardware marks floors measured by round-tripping
	// captured vectors through the scheme on identified hardware.
	FloorEvidenceObservedHardware FloorEvidenceKind = "observed_hardware"
)

// FloorScheme describes the quantization scheme whose intrinsic round-trip
// error floor a receipt records: precision, block size, scale format, and the
// role the scheme compresses.
type FloorScheme struct {
	Precision   string    `json:"precision"`
	BlockSize   int       `json:"block_size"`
	ScaleFormat string    `json:"scale_format,omitempty"`
	Role        FloorRole `json:"role"`
}

// RoundTripStats holds the measured statistics of round-tripping captured
// vectors through the scheme itself. RelativeError is the headline floor
// fraction (for example 0.0280 is ~2.80 %); Correlation and MaxAbsDeviation
// are optional axes a measurement may or may not report, so they are pointers
// whose nil means "not reported" rather than a fabricated zero.
type RoundTripStats struct {
	RelativeError   float64  `json:"relative_error"`
	Correlation     *float64 `json:"correlation,omitempty"`
	MaxAbsDeviation *float64 `json:"max_abs_deviation,omitempty"`
}

// FloorReceipt is the versioned record of one scheme's intrinsic error floor:
// the scheme descriptor, the round-trip statistics, and the provenance that
// says whether the numbers are quoted priors or observed measurements.
type FloorReceipt struct {
	Contract   string            `json:"contract"`
	Scheme     FloorScheme       `json:"scheme"`
	Stats      RoundTripStats    `json:"stats"`
	Evidence   FloorEvidenceKind `json:"evidence"`
	SourcePin  string            `json:"source_pin,omitempty"`
	License    string            `json:"license,omitempty"`
	Provenance string            `json:"provenance,omitempty"`
}

// NewFloorReceipt builds a versioned floor receipt, failing closed on
// malformed input: an unknown role or evidence kind, a non-positive block
// size, a blank precision, or a non-finite relative error. A zero relative
// error is admitted: a lossless scheme genuinely owes nothing.
func NewFloorReceipt(scheme FloorScheme, stats RoundTripStats, evidence FloorEvidenceKind, sourcePin, license, provenance string) (FloorReceipt, error) {
	if strings.TrimSpace(scheme.Precision) == "" {
		return FloorReceipt{}, errors.New("floor scheme requires a precision")
	}
	if scheme.BlockSize <= 0 {
		return FloorReceipt{}, fmt.Errorf("floor scheme block size %d must be positive", scheme.BlockSize)
	}
	if !knownFloorRoles[scheme.Role] {
		return FloorReceipt{}, fmt.Errorf("floor role %q is not one of window_kv, latent, indexer", scheme.Role)
	}
	if evidence != FloorEvidenceUpstreamQuoted && evidence != FloorEvidenceObservedHardware {
		return FloorReceipt{}, fmt.Errorf("floor evidence %q is not classified", evidence)
	}
	if !finite(stats.RelativeError) || stats.RelativeError < 0 {
		return FloorReceipt{}, fmt.Errorf("floor relative error %v must be finite and non-negative", stats.RelativeError)
	}
	for _, opt := range []*float64{stats.Correlation, stats.MaxAbsDeviation} {
		if opt != nil && !finite(*opt) {
			return FloorReceipt{}, errors.New("optional floor statistics must be finite when present")
		}
	}
	return FloorReceipt{
		Contract:   FloorContractVersion,
		Scheme:     scheme,
		Stats:      stats,
		Evidence:   evidence,
		SourcePin:  sourcePin,
		License:    license,
		Provenance: provenance,
	}, nil
}

// FloorVerdict is the floor-relative outcome of a port-vs-reference comparison.
type FloorVerdict string

const (
	// FloorWithinFloor: the observed drift sits at or below the scheme's
	// intrinsic floor. The divergence is what the scheme itself pays on real
	// data, so it is not a defect of the port.
	FloorWithinFloor FloorVerdict = "within_floor"
	// FloorAboveFloor: the observed drift exceeds the scheme's intrinsic
	// floor. The excess is a genuine regression signal that the budget
	// evaluation should see, not tolerated scheme error.
	FloorAboveFloor FloorVerdict = "above_floor"
)

// AdjudicateAgainstFloor compares an observed port-vs-reference relative error
// against the scheme's intrinsic floor on the headline axis: observed drift at
// or below the floor is within floor (not a defect); drift above it is a
// regression signal. The comparison is fail-closed: a mismatched contract, a
// non-finite floor statistic, or a non-finite observed value is refused rather
// than guessed.
func AdjudicateAgainstFloor(receipt FloorReceipt, observedRelativeError float64) (FloorVerdict, error) {
	if receipt.Contract != FloorContractVersion {
		return "", fmt.Errorf("floor receipt contract %q is not %s", receipt.Contract, FloorContractVersion)
	}
	if !finite(receipt.Stats.RelativeError) || receipt.Stats.RelativeError < 0 {
		return "", errors.New("floor receipt carries a non-finite or negative relative error")
	}
	if !finite(observedRelativeError) || observedRelativeError < 0 {
		return "", fmt.Errorf("observed relative error %v must be finite and non-negative", observedRelativeError)
	}
	if observedRelativeError <= receipt.Stats.RelativeError {
		return FloorWithinFloor, nil
	}
	return FloorAboveFloor, nil
}

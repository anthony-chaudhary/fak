// Read-path cost decomposition with a falsifiable additive model.
//
// The problem it solves. internal/roofline today prices a moe-streaming token
// with an *analytical* roofline (DRAM bandwidth, the MALL boundary sweep, WMMA
// ceilings, knee points) and a digest-verified empirical receipt. Neither knows
// where an individual token's time actually goes across tiers — compute vs the
// RAM-served miss path vs the disk-served miss path — so engine work that moves
// the expert read path cannot be priced from real tier measurements. This file
// adds the missing seam: a pure, bounded cost-model helper that takes a
// fixed-cache tier sweep plus a *measured* resident rate and returns the split.
//
// Method note (source pin, license, clean-room provenance).
//
//	Source:   JigSawPT/deepseek-v41-flash-on-5090 (MIT), pinned at
//	          path:line@sha `tools/stall_decomposition.py:22-31,43-92` and
//	          `results/tier_direct_20260923/README.md:17-22`
//	          @ dbbdb887211fc8a9581c29b8ddcce1a6cad74061.
//	License:  MIT. Disposition: cite-and-reimplement at the *method* level only.
//	          No upstream source is vendored, translated, or copied; nothing
//	          above is mechanically derived from its code. The anchors are the
//	          citation record a reviewer can re-read.
//	Method:   stall_per_call = base + misses x [ f*ram_cost + (1-f)*disk_cost ],
//	          with `base` read from the *measured resident case* and never a
//	          fitted intercept. This file reimplements that algebra clean-room as
//	          a bounded least-squares solve, and — the load-bearing part of the
//	          source finding — returns a *falsified* verdict rather than coercing
//	          a fit when a fitted per-miss cost is negative or the sweep does not
//	          actually vary the RAM/disk split (the hit-rate-invariance
//	          precondition). A negative fitted per-miss cost is the signal that
//	          the additive assumption is wrong for that read path, exactly as the
//	          source observed on its O_DIRECT direct path.
//
// Honesty contract. `base` (and therefore the compute fraction) comes only from
// the caller's measured resident tokens/s, never from a fitted intercept. A
// decomposition that cannot be identified, or that fits a physically impossible
// per-miss cost, is returned as Falsified with a reason — never silently coerced
// into a split. Source-only: this is a pure model over caller-supplied
// measurements and makes no hardware claim of its own.
package roofline

import (
	"errors"
	"fmt"
	"math"
)

// ReadPathVerdict is the outcome of a read-path decomposition. A model that
// cannot be identified or that fits an impossible per-miss cost is Falsified,
// not coerced.
type ReadPathVerdict string

const (
	// ReadPathFits means the tier sweep identified a non-negative, correctly
	// ordered per-miss RAM/disk cost pair; the split and ceilings are valid.
	ReadPathFits ReadPathVerdict = "fits"
	// ReadPathFalsified means the additive model does not hold for this read
	// path — a negative or mis-ordered fitted cost, or a sweep that does not
	// vary the RAM/disk split. This is a first-class returned outcome.
	ReadPathFalsified ReadPathVerdict = "falsified"
)

// ReadPathTierPoint is one measured point of a fixed-cache tier sweep. The sweep
// varies the cache footprint (L2SizeBytes) so that both the expected number of
// L2 misses per call and the fraction of those misses served by the RAM tier
// move; that co-variation is what makes the two per-miss costs separable.
type ReadPathTierPoint struct {
	// L2SizeBytes is the swept cache footprint at this point.
	L2SizeBytes int64 `json:"l2_size_bytes"`
	// MeanStallMillis is the measured mean stall per call at this point.
	MeanStallMillis float64 `json:"mean_stall_millis"`
	// MissesPerCall is the expected number of L2 misses per call at this point.
	MissesPerCall float64 `json:"misses_per_call"`
	// RAMFraction is the fraction of those misses served by the RAM tier; the
	// remainder (1-RAMFraction) is served by the disk tier. 0 <= RAMFraction<=1.
	RAMFraction float64 `json:"ram_fraction"`
}

// ReadPathCostModel is the decomposition result. The fitted costs are always
// reported, even when the verdict is Falsified (an informative negative cost is
// more useful than a withheld one); the split and ceilings are only valid when
// Verdict == ReadPathFits.
type ReadPathCostModel struct {
	// Identifiability is the normal-equations determinant of the solve, scaled
	// to the point magnitudes. A value at or below zero means the sweep points
	// do not separate the RAM and disk per-miss costs.
	Identifiability float64 `json:"identifiability"`

	// BaseStallMillis is the resident-case stall per call, derived from the
	// caller's measured resident rate — never a fitted intercept.
	BaseStallMillis float64 `json:"base_stall_millis"`
	// RAMMissCostMillis and DiskMissCostMillis are the fitted per-miss costs for
	// a miss served by the RAM tier and by the disk tier respectively. Negative
	// values are possible and are the falsification signal.
	RAMMissCostMillis  float64 `json:"ram_miss_cost_millis"`
	DiskMissCostMillis float64 `json:"disk_miss_cost_millis"`

	// ComputeFraction, RAMFraction and DiskFraction are the time split at the
	// sweep's mean operating point. Valid only when Verdict is ReadPathFits.
	ComputeFraction float64 `json:"compute_fraction"`
	RAMFraction     float64 `json:"ram_fraction"`
	DiskFraction    float64 `json:"disk_fraction"`

	// AllResidentTokensPerSec is the caller's measured resident rate (the
	// no-miss ceiling). NoDiskTokensPerSec is the ceiling with the disk tier
	// absent (every miss served by RAM). ObservedTokensPerSec is the inferred
	// rate at the sweep's mean stall. Valid only when Verdict is ReadPathFits.
	AllResidentTokensPerSec float64 `json:"all_resident_tokens_per_sec"`
	NoDiskTokensPerSec      float64 `json:"no_disk_tokens_per_sec"`
	ObservedTokensPerSec    float64 `json:"observed_tokens_per_sec"`

	Verdict ReadPathVerdict `json:"verdict"`
	Reason  string          `json:"reason,omitempty"`
}

// ErrReadPathCostModel is a typed refusal for an unusable input envelope (as
// opposed to a Falsified verdict, which is a valid model outcome over a usable
// envelope).
var ErrReadPathCostModel = errors.New("roofline: read-path cost model input refused")

// identityEpsilon bounds the relative identifiability below which the sweep is
// treated as not varying the RAM/disk split. It is deliberately loose: the cost
// pair is only meaningful when the points genuinely span different hit-rate
// regimes, and a near-singular system produces a wildly unstable fit.
const identityEpsilon = 1e-9

// DecomposeReadPathCost solves the additive read-path cost model from a
// fixed-cache tier sweep and a measured resident rate.
//
// Given points i with measured mean stall s_i, expected L2 misses per call m_i
// and RAM-served miss fraction f_i, and a resident rate R:
//
//	base   = 1000 / R                          (millis per call, measured)
//	y_i    = s_i - base
//	y_i    = (m_i*f_i)*cRAM + (m_i*(1-f_i))*cDisk
//
// The two per-miss costs are recovered by least squares over the sweep. The
// result is Falsified — not coerced — when the sweep does not identify the pair
// (miss-fraction-invariance precondition) or when a fitted cost is negative or
// the RAM tier costs more than the disk tier it fronts.
//
// It returns ErrReadPathCostModel (wrapped) only when the input envelope itself
// is unusable: fewer than two points, a non-positive resident rate, or an
// out-of-range field. A usable envelope over inconsistent measurements yields a
// Falsified model with a nil error, so callers can distinguish "bad inputs" from
// "the model does not hold".
func DecomposeReadPathCost(points []ReadPathTierPoint, residentTokensPerSec float64) (ReadPathCostModel, error) {
	if !(residentTokensPerSec > 0) || math.IsInf(residentTokensPerSec, 0) || math.IsNaN(residentTokensPerSec) {
		return ReadPathCostModel{}, fmt.Errorf("%w: resident tokens/s must be finite and positive, got %v", ErrReadPathCostModel, residentTokensPerSec)
	}
	if len(points) < 2 {
		return ReadPathCostModel{}, fmt.Errorf("%w: need at least two sweep points, got %d", ErrReadPathCostModel, len(points))
	}
	for i, p := range points {
		if p.L2SizeBytes <= 0 {
			return ReadPathCostModel{}, fmt.Errorf("%w: point %d: L2SizeBytes must be positive, got %d", ErrReadPathCostModel, i, p.L2SizeBytes)
		}
		if math.IsNaN(p.MeanStallMillis) || p.MeanStallMillis < 0 {
			return ReadPathCostModel{}, fmt.Errorf("%w: point %d: MeanStallMillis must be non-negative, got %v", ErrReadPathCostModel, i, p.MeanStallMillis)
		}
		if math.IsNaN(p.MissesPerCall) || p.MissesPerCall < 0 {
			return ReadPathCostModel{}, fmt.Errorf("%w: point %d: MissesPerCall must be non-negative, got %v", ErrReadPathCostModel, i, p.MissesPerCall)
		}
		if math.IsNaN(p.RAMFraction) || p.RAMFraction < 0 || p.RAMFraction > 1 {
			return ReadPathCostModel{}, fmt.Errorf("%w: point %d: RAMFraction must be in [0,1], got %v", ErrReadPathCostModel, i, p.RAMFraction)
		}
	}

	base := 1000.0 / residentTokensPerSec

	// Build the normal equations for y_i = a_i*cRAM + b_i*cDisk, with
	// a_i = m_i*f_i (RAM-tier misses per call) and b_i = m_i*(1-f_i).
	var saa, sab, sbb, say, sby float64
	for _, p := range points {
		a := p.MissesPerCall * p.RAMFraction
		b := p.MissesPerCall * (1 - p.RAMFraction)
		y := p.MeanStallMillis - base
		saa += a * a
		sab += a * b
		sbb += b * b
		say += a * y
		sby += b * y
	}

	det := saa*sbb - sab*sab
	scale := saa*sbb + sab*sab + 1
	identifiability := det / scale

	m := ReadPathCostModel{
		Identifiability:         identifiability,
		BaseStallMillis:         base,
		AllResidentTokensPerSec: residentTokensPerSec,
	}

	// The hit-rate-invariance precondition: if every point shares the same
	// RAM/disk split ratio, the two per-miss costs are not separable and any
	// fitted pair is an artifact of noise. Return Falsified, not a split.
	if math.Abs(det) <= identityEpsilon*scale {
		m.Verdict = ReadPathFalsified
		m.Reason = "hit-rate invariance failed: the sweep does not vary the RAM/disk split, so the per-miss costs are not identifiable"
		return m, nil
	}

	m.RAMMissCostMillis = (sbb*say - sab*sby) / det
	m.DiskMissCostMillis = (saa*sby - sab*say) / det

	switch {
	case m.RAMMissCostMillis < 0 || m.DiskMissCostMillis < 0:
		m.Verdict = ReadPathFalsified
		m.Reason = fmt.Sprintf("negative fitted per-miss cost (ram=%.6g ms, disk=%.6g ms): the additive model does not hold for this read path", m.RAMMissCostMillis, m.DiskMissCostMillis)
		return m, nil
	case m.RAMMissCostMillis > m.DiskMissCostMillis:
		m.Verdict = ReadPathFalsified
		m.Reason = fmt.Sprintf("ordering violated: the RAM tier fronts the disk tier but fits a higher per-miss cost (ram=%.6g ms > disk=%.6g ms)", m.RAMMissCostMillis, m.DiskMissCostMillis)
		return m, nil
	}

	// Split at the sweep's mean operating point.
	var meanRAMMiss, meanDiskMiss, meanStall float64
	for _, p := range points {
		meanRAMMiss += p.MissesPerCall * p.RAMFraction
		meanDiskMiss += p.MissesPerCall * (1 - p.RAMFraction)
		meanStall += p.MeanStallMillis
	}
	n := float64(len(points))
	meanRAMMiss /= n
	meanDiskMiss /= n
	meanStall /= n

	ramPart := meanRAMMiss * m.RAMMissCostMillis
	diskPart := meanDiskMiss * m.DiskMissCostMillis
	total := base + ramPart + diskPart
	if total <= 0 {
		m.Verdict = ReadPathFalsified
		m.Reason = fmt.Sprintf("non-positive total modeled stall (base=%.6g ms, ram=%.6g ms, disk=%.6g ms)", base, ramPart, diskPart)
		return m, nil
	}

	m.ComputeFraction = base / total
	m.RAMFraction = ramPart / total
	m.DiskFraction = diskPart / total
	m.NoDiskTokensPerSec = 1000.0 / (base + ramPart)
	if meanStall > 0 {
		m.ObservedTokensPerSec = 1000.0 / meanStall
	}
	m.Verdict = ReadPathFits
	return m, nil
}

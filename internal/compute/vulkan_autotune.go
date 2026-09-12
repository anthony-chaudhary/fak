package compute

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"
)

// VulkanAutotuneKey is the exact evidence/cache key. No architecture aliases,
// rounded dimensions, driver normalization, or shader substitutions are allowed.
// The source/binary/model/workload/power envelope is checked separately on reuse.
type VulkanAutotuneKey struct {
	DeviceUUID, DriverVersion, QuantFormat string
	M, N, K                                int
	SPIRVDigest                            string
}

// VulkanAutotuneEnvelope identifies the common experiment, including both shader
// variants in the same source and binary. PowerEnvelope identifies a fixed power
// configuration; it is not a measured wattage that may vary from sample to sample.
type VulkanAutotuneEnvelope struct {
	SourceRevision, BinaryDigest, ModelDigest, WorkloadDigest, PowerEnvelope string
}

// VulkanRowtileVariant names a precompiled runtime variant. ResourceCost is a
// positive, caller-declared comparable cost (for example LDS bytes); a campaign
// must use one cost definition for its entire registry. Evidence cannot change it.
type VulkanRowtileVariant struct {
	ID           string
	Key          VulkanAutotuneKey
	ResourceCost uint64
}

// VulkanAutotuneObservation must come from synchronized fak-native execution of
// Variant. OraclePassed attests the campaign's exact/floored numerical oracle;
// it is not inferred from elapsed time or a successful dispatch. CompletedAt is
// the host completion timestamp, while Duration is the synchronized device time.
type VulkanAutotuneObservation struct {
	Variant       VulkanRowtileVariant
	Envelope      VulkanAutotuneEnvelope
	Engine        string
	Duration      time.Duration
	CompletedAt   time.Time
	OraclePassed  bool
	FallbackCount uint64
}

// VulkanAutotunePair records one control/candidate comparison. Even-indexed pairs
// run control then candidate; odd-indexed pairs reverse that order. Completion
// timestamps must establish that exact chronological transcript. Each reported
// duration must fit after the preceding completion (no overlapping device runs).
type VulkanAutotunePair struct {
	Control, Candidate VulkanAutotuneObservation
}

// VulkanAutotuneRequest pins the registry and admission policy before sampling.
// Now and MaxAge make cache expiry explicit and deterministic. Pairs is fixed
// before the sweep: stopping early when a winner appears is forbidden.
type VulkanAutotuneRequest struct {
	GFX        string
	Control    VulkanRowtileVariant
	Candidates []VulkanRowtileVariant
	Envelope   VulkanAutotuneEnvelope
	Now        time.Time
	MaxAge     time.Duration
	Pairs      int
	MinSpeedup float64
}

// VulkanAutotuneSelection is an advisory runtime adapter result, not a global
// default mutation or a hardware performance claim. Runtime pipeline wiring must
// resolve the returned exact variant/key or retain Control.
type VulkanAutotuneSelection struct {
	Variant    VulkanRowtileVariant
	Promoted   bool
	LowerBound float64
	Reason     string
}

const (
	vulkanAutotuneMaxCandidates = 8
	vulkanAutotuneMaxPairs      = 32
)

// SelectVulkanRowtileCandidate applies the same gate to fresh and cached samples.
// Invariants/proof sketch: validate the fixed registry first; admit only exact
// identities, fresh oracle-passing zero-fallback interleaved samples; reject any
// series with sample CV >5%; require a strictly positive confidence-bound lift.
// Reduce eligible candidates by (descending bound, ascending declared cost,
// ascending unique ID). This total order is independent of registry/map order.
// If the eligible set is empty, the untouched static control is the result.
// Bounds are per candidate, not a family-wise claim about the winning sweep.
func SelectVulkanRowtileCandidate(r VulkanAutotuneRequest, profiles map[string][]VulkanAutotunePair) VulkanAutotuneSelection {
	best := VulkanAutotuneSelection{Variant: r.Control, Reason: "no eligible measured candidate"}
	if err := r.validate(); err != nil {
		best.Reason = err.Error()
		return best
	}
	threshold := math.Max(1, r.MinSpeedup)
	for _, v := range r.Candidates {
		bound, ok := vulkanAutotuneBound(r, v, profiles[v.ID])
		if !ok || bound <= threshold {
			continue
		}
		if !best.Promoted || bound > best.LowerBound ||
			(bound == best.LowerBound && (v.ResourceCost < best.Variant.ResourceCost ||
				(v.ResourceCost == best.Variant.ResourceCost && v.ID < best.Variant.ID))) {
			best = VulkanAutotuneSelection{Variant: v, Promoted: true, LowerBound: bound, Reason: "measured confidence bound exceeds threshold"}
		}
	}
	return best
}

// MeasureVulkanRowtileCandidates runs a bounded sweep through an issue-local
// runtime adapter. measure must honor ctx, execute the exact requested native
// pipeline, synchronize, and return actual identity and oracle evidence. The
// selector never manufactures identity or oracle fields for it. A deadline is
// mandatory; cancellation/error discards the entire partial sweep for selection.
// At most 8*32*2 calls occur, in lexicographic variant order and alternating pairs.
// Freshness is evaluated against the host clock after the sweep. Imported
// evidence instead uses the explicit Now passed to SelectVulkanRowtileCandidate.
func MeasureVulkanRowtileCandidates(ctx context.Context, r VulkanAutotuneRequest, measure func(context.Context, VulkanRowtileVariant) (VulkanAutotuneObservation, error)) (VulkanAutotuneSelection, map[string][]VulkanAutotunePair, error) {
	fallback := VulkanAutotuneSelection{Variant: r.Control, Reason: "incomplete measurement sweep"}
	if err := r.validate(); err != nil {
		return fallback, nil, err
	}
	if _, ok := ctx.Deadline(); !ok || measure == nil {
		return fallback, nil, fmt.Errorf("Vulkan autotune requires a deadline and measurement callback")
	}
	variants := append([]VulkanRowtileVariant(nil), r.Candidates...)
	sort.Slice(variants, func(i, j int) bool { return variants[i].ID < variants[j].ID })
	profiles := make(map[string][]VulkanAutotunePair, len(variants))
	for _, v := range variants {
		pairs := make([]VulkanAutotunePair, r.Pairs)
		for i := range pairs {
			order := []VulkanRowtileVariant{r.Control, v}
			out := []*VulkanAutotuneObservation{&pairs[i].Control, &pairs[i].Candidate}
			if i%2 != 0 {
				order[0], order[1] = order[1], order[0]
				out[0], out[1] = out[1], out[0]
			}
			for j, target := range order {
				if err := ctx.Err(); err != nil {
					return fallback, nil, err
				}
				observation, err := measure(ctx, target)
				if err != nil {
					return fallback, nil, err
				}
				*out[j] = observation
			}
		}
		profiles[v.ID] = pairs
	}
	if err := ctx.Err(); err != nil {
		return fallback, nil, err
	}
	r.Now = time.Now()
	return SelectVulkanRowtileCandidate(r, profiles), profiles, nil
}

func (r VulkanAutotuneRequest) validate() error {
	if r.GFX != "gfx1151" || r.Now.IsZero() || r.MaxAge <= 0 ||
		r.Pairs < 5 || r.Pairs > vulkanAutotuneMaxPairs ||
		len(r.Candidates) == 0 || len(r.Candidates) > vulkanAutotuneMaxCandidates ||
		math.IsNaN(r.MinSpeedup) || math.IsInf(r.MinSpeedup, 0) {
		return fmt.Errorf("invalid Vulkan autotune scope or bounded sampling policy")
	}
	e := r.Envelope
	if e.SourceRevision == "" || e.BinaryDigest == "" || e.ModelDigest == "" || e.WorkloadDigest == "" || e.PowerEnvelope == "" {
		return fmt.Errorf("incomplete Vulkan autotune experiment envelope")
	}
	seen := make(map[string]bool, len(r.Candidates)+1)
	for _, v := range append([]VulkanRowtileVariant{r.Control}, r.Candidates...) {
		k := v.Key
		if v.ID == "" || seen[v.ID] || v.ResourceCost == 0 || k.DeviceUUID == "" || k.DriverVersion == "" ||
			k.QuantFormat == "" || k.M <= 0 || k.N <= 0 || k.K <= 0 || k.SPIRVDigest == "" {
			return fmt.Errorf("invalid or ambiguous Vulkan autotune variant registry")
		}
		seen[v.ID] = true
		// Candidate shader identity differs, but device/driver/quant/shape cannot.
		k.SPIRVDigest = r.Control.Key.SPIRVDigest
		if k != r.Control.Key {
			return fmt.Errorf("incomparable Vulkan autotune candidate shape or device")
		}
	}
	return nil
}

func vulkanAutotuneBound(r VulkanAutotuneRequest, v VulkanRowtileVariant, pairs []VulkanAutotunePair) (float64, bool) {
	if len(pairs) != r.Pairs {
		return 0, false
	}
	controls := make([]float64, len(pairs))
	candidates := make([]float64, len(pairs))
	ratios := make([]float64, len(pairs))
	var previous time.Time
	for i, pair := range pairs {
		if !vulkanAutotuneObservationValid(r, r.Control, pair.Control) || !vulkanAutotuneObservationValid(r, v, pair.Candidate) {
			return 0, false
		}
		first, second := pair.Control, pair.Candidate
		if i%2 != 0 {
			first, second = second, first
		}
		if !first.CompletedAt.After(previous) || !second.CompletedAt.After(first.CompletedAt) ||
			(!previous.IsZero() && first.CompletedAt.Add(-first.Duration).Before(previous)) ||
			second.CompletedAt.Add(-second.Duration).Before(first.CompletedAt) {
			return 0, false
		}
		previous = second.CompletedAt
		controls[i], candidates[i] = float64(pair.Control.Duration), float64(pair.Candidate.Duration)
		ratios[i] = controls[i] / candidates[i]
	}
	for _, series := range [][]float64{controls, candidates, ratios} {
		mean, sd := vulkanAutotuneMoments(series)
		if sd/mean > 0.05 {
			return 0, false
		}
	}
	mean, sd := vulkanAutotuneMoments(ratios)
	// One-sided Student-t lower bound on mean paired speedup. The fixed critical
	// value rounds UP t(0.95,4), conservatively covering every admitted n>=5.
	// Assumptions: independent pairs, approximately normal ratio population.
	// CV alone does not establish those assumptions or replace held-out hardware
	// validation. https://www.itl.nist.gov/div898/handbook/prc/section2/prc221.htm
	lower := mean - 2.132847*sd/math.Sqrt(float64(len(ratios)))
	return lower, !math.IsNaN(lower) && !math.IsInf(lower, 0)
}

func vulkanAutotuneObservationValid(r VulkanAutotuneRequest, v VulkanRowtileVariant, o VulkanAutotuneObservation) bool {
	return o.Variant == v && o.Envelope == r.Envelope && o.Engine == "fak-native" && o.OraclePassed && o.FallbackCount == 0 &&
		o.Duration > 0 && !o.CompletedAt.IsZero() && !o.CompletedAt.After(r.Now) && r.Now.Sub(o.CompletedAt) <= r.MaxAge
}

func vulkanAutotuneMoments(values []float64) (mean, sd float64) {
	// A shifted mean preserves a constant series exactly, so floating-point summation
	// cannot manufacture lift when every ratio equals the campaign threshold.
	base := values[0]
	for _, x := range values {
		mean += x - base
	}
	mean = base + mean/float64(len(values))
	for _, x := range values {
		sd += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(sd / float64(len(values)-1))
}

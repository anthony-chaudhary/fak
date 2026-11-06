package model

import (
	"errors"
	"sort"
)

// dynamic_sd_schedule.go — the operator-driven batch-size -> K schedule for
// dynamic speculative decoding. It is an independent deterministic axis BESIDE
// AdaptiveDraftDepthController (the acceptance-hysteresis axis) and
// SelfSpecGovernor (the page-in-tax axis): the effective speculative depth for a
// live step is the min of the three, so no single axis can over-draft.
//
// ADAPT (not a verbatim port) of vLLM's dynamic speculative-decoding schedule:
//   - vllm/v1/spec_decode/dynamic/utils.py:7-148
//     (validate_and_normalize_dynamic_sd_schedule + build_dynamic_sd_schedule_lookup)
//   - vllm/config/speculative.py:517-519,2122-2123
//     (num_speculative_tokens_per_batch_size + uses_dynamic_speculative_decoding)
//   - vllm/v1/core/sched/scheduler.py:1658-1660 (per-step K = lookup[batch_size])
// at pin ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0).
//
// Mechanism: the operator supplies (batch_size, K) points; validation normalizes
// them into a DENSE monotonically-indexed table covering every batch size
// 1..MaxBatchSize (unspecified sizes carry forward the most recent point's K),
// rejects duplicates / out-of-range / non-monotonic entries, and the scheduler
// picks K per step from the live batch size instead of a global constant. The
// upstream expresses the schedule as inclusive (range_start, range_end, K)
// ranges; this adaptation re-expresses the same validate + build-dense-lookup +
// per-step-lookup idea as point pairs, which is the form the operator contract
// names and which the unit test pins.
//
// DEFAULT-OFF: the zero value (no table) imposes no operator bound, so
// EffectiveK falls back to min(adaptiveDepth, governorAllowance) — a kernel that
// never opts in a schedule speculates exactly as the other two axes decide.

// SDSchedulePoint is one operator-supplied (batch_size, K) pair: at runtime batch
// size BatchSize the speculative depth is K.
type SDSchedulePoint struct {
	BatchSize int
	K         int
}

// DynamicSDSchedule is a validated, dense batch_size -> K lookup. The table is
// 1-indexed (index 0 is unused) so a live batch size is a direct array index —
// the per-step O(1) lookup the scheduler wants instead of a range search.
type DynamicSDSchedule struct {
	table        []int
	maxBatchSize int
}

// Sentinel errors for the closed validation contract. Callers match these with
// errors.Is, never the prose.
var (
	// ErrEmptySchedule is returned when no (batch_size, K) points are supplied.
	ErrEmptySchedule = errors.New("model: dynamic SD schedule requires at least one point")
	// ErrNonPositiveMaxBatch is returned when maxBatchSize <= 0.
	ErrNonPositiveMaxBatch = errors.New("model: dynamic SD schedule max batch size must be positive")
	// ErrNegativeMaxK is returned when the configured K maximum is < 0.
	ErrNegativeMaxK = errors.New("model: dynamic SD schedule max K must be non-negative")
	// ErrNonPositiveBatch is returned when a point's batch size is <= 0.
	ErrNonPositiveBatch = errors.New("model: dynamic SD schedule batch size must be positive")
	// ErrFirstBatchNotOne is returned when the smallest point is not batch size 1,
	// which would leave the low batch sizes without a defined K.
	ErrFirstBatchNotOne = errors.New("model: dynamic SD schedule must start at batch size 1")
	// ErrNegativeK is returned when a point's K is < 0.
	ErrNegativeK = errors.New("model: dynamic SD schedule K must be non-negative")
	// ErrKOutOfRange is returned when a point's K exceeds the configured maximum.
	ErrKOutOfRange = errors.New("model: dynamic SD schedule K exceeds the configured maximum")
	// ErrDuplicateBatchSize is returned when two points share a batch size.
	ErrDuplicateBatchSize = errors.New("model: dynamic SD schedule has a duplicate batch size")
	// ErrNonMonotonicK is returned when K increases as batch size increases.
	ErrNonMonotonicK = errors.New("model: dynamic SD schedule K must be non-increasing with batch size")
)

// NewDynamicSDSchedule validates points and builds the dense 1..maxBatchSize
// table. maxBatchSize is the largest runtime batch size to cover; maxK is the
// global cap on K (a point's K above it is out-of-range and rejected).
//
// Validation (the closed contract):
//  1. maxBatchSize > 0, maxK >= 0, points non-empty.
//  2. every point has BatchSize > 0, 0 <= K <= maxK.
//  3. no duplicate batch size.
//  4. the smallest batch size is 1 (every low batch size has a defined K).
//  5. K is non-increasing as batch size increases (monotonicity enforced).
//
// The dense table is filled by carry-forward (step) interpolation: a gap between
// two points and the tail after the last point carry forward the most recent
// point's K, so batch sizes above the largest point clamp to the last entry.
func NewDynamicSDSchedule(points []SDSchedulePoint, maxBatchSize, maxK int) (DynamicSDSchedule, error) {
	if maxBatchSize <= 0 {
		return DynamicSDSchedule{}, ErrNonPositiveMaxBatch
	}
	if maxK < 0 {
		return DynamicSDSchedule{}, ErrNegativeMaxK
	}
	if len(points) == 0 {
		return DynamicSDSchedule{}, ErrEmptySchedule
	}

	sorted := make([]SDSchedulePoint, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].BatchSize < sorted[j].BatchSize })

	for i := 1; i < len(sorted); i++ {
		if sorted[i].BatchSize == sorted[i-1].BatchSize {
			return DynamicSDSchedule{}, ErrDuplicateBatchSize
		}
	}
	if sorted[0].BatchSize != 1 {
		return DynamicSDSchedule{}, ErrFirstBatchNotOne
	}

	prevK := 0
	for i, p := range sorted {
		if p.BatchSize <= 0 {
			return DynamicSDSchedule{}, ErrNonPositiveBatch
		}
		if p.K < 0 {
			return DynamicSDSchedule{}, ErrNegativeK
		}
		if p.K > maxK {
			return DynamicSDSchedule{}, ErrKOutOfRange
		}
		if i > 0 && p.K > prevK {
			return DynamicSDSchedule{}, ErrNonMonotonicK
		}
		prevK = p.K
	}

	table := make([]int, maxBatchSize+1) // index 0 unused
	next := 1
	lastK := 0
	for _, p := range sorted {
		if p.BatchSize > maxBatchSize {
			break // beyond the covered range; the tail below is already filled
		}
		for b := next; b < p.BatchSize; b++ {
			table[b] = lastK // carry forward the previous K across the gap
		}
		table[p.BatchSize] = p.K
		lastK = p.K
		next = p.BatchSize + 1
		if next > maxBatchSize {
			break
		}
	}
	for b := next; b <= maxBatchSize; b++ {
		table[b] = lastK // tail clamps to the last entry's K
	}

	return DynamicSDSchedule{table: table, maxBatchSize: maxBatchSize}, nil
}

// K returns the operator-schedule K for a live batch size. Batch sizes above the
// largest covered size clamp to the last entry; batch sizes below 1 clamp to 1.
// ok is false when the schedule is unconfigured (zero value), meaning the
// operator imposed no bound.
func (s DynamicSDSchedule) K(batchSize int) (int, bool) {
	if len(s.table) == 0 {
		return 0, false
	}
	if batchSize < 1 {
		batchSize = 1
	}
	if batchSize > s.maxBatchSize {
		batchSize = s.maxBatchSize
	}
	return s.table[batchSize], true
}

// EffectiveK composes the three independent deterministic axes into the
// effective speculative depth for a live batch size:
//
//	effective K = min(operator schedule K(batchSize), adaptiveDepth, governorAllowance)
//
// An unconfigured schedule (zero value) imposes no operator bound, so the result
// is min(adaptiveDepth, governorAllowance) — the default-off posture.
func (s DynamicSDSchedule) EffectiveK(batchSize, adaptiveDepth, governorAllowance int) int {
	k, ok := s.K(batchSize)
	if !ok {
		if adaptiveDepth < governorAllowance {
			return adaptiveDepth
		}
		return governorAllowance
	}
	m := k
	if adaptiveDepth < m {
		m = adaptiveDepth
	}
	if governorAllowance < m {
		m = governorAllowance
	}
	return m
}

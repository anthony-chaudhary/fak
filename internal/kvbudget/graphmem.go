// This file adapts vLLM's CUDA-graph memory profiling into the KV-budget
// derivation: a probe pass over the captured decode-bucket ladder records the
// free-device-bytes delta around each bucket capture, the full-ladder pool
// footprint is extrapolated from that sample, and the derived max-blocks count
// charges the estimate against the utilized budget alongside the non-KV bytes
// before dividing by per-block bytes.
//
// # Source of truth (ADAPT verdict — re-expressed in idiomatic Go, no copied
// code blocks)
//
//	vllm-project/vllm @ ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0)
//	  - vllm/v1/worker/gpu/cudagraph_utils.py:922-1037
//	    profile_cudagraph_memory + _extrapolate_full_graph_memory:
//	    first capture pays the pool baseline, later buckets mostly reuse it, so
//	    the second sample is taken as the per-bucket cost (floored at
//	    _MIN_PER_GRAPH_BYTES) and the total is first + (total-1) × per-bucket;
//	    a NONE-mode / zero-sample probe reserves 0.
//	  - vllm/v1/worker/gpu_worker.py:705-750
//	    determine_available_memory: available_kv_cache_memory_bytes =
//	    requested - non_kv_cache_memory - cudagraph_estimate_applied.
package kvbudget

import "math"

// MinPerGraphBytes is the floor for the extrapolated per-bucket graph cost:
// even when the second probe sample measures less, every captured bucket pays
// at least this much driver/pool overhead (vLLM's _MIN_PER_GRAPH_BYTES).
const MinPerGraphBytes = uint64(1 << 20) // 1 MiB

// GraphMemProbe is the device-allocation evidence a probe pass records over a
// captured decode-bucket ladder: the free-device-bytes delta observed around
// each bucket's capture, in capture order (largest buckets first). Deltas[0]
// is the first capture, which pays the graph-pool baseline; Deltas[1:] are the
// incremental costs of the later buckets. A probe that captured nothing (the
// NONE-mode case) carries an empty Deltas slice and reserves 0.
type GraphMemProbe struct {
	// Deltas is the free-device-bytes delta recorded around each captured
	// bucket, in capture order.
	Deltas []uint64
}

// ExtrapolatedPoolFootprint estimates the total device bytes the full
// decode-bucket ladder occupies once all totalBuckets are captured,
// extrapolated from the probe sample: the first capture allocates the pool
// baseline, later buckets mostly reuse it, so the second sample is taken as
// the per-bucket cost (floored at MinPerGraphBytes) and the total is
// first + (totalBuckets-1) × per-bucket. Samples beyond the second are
// ignored. A zero-bucket or NONE-mode probe (no deltas, or totalBuckets ≤ 0)
// reserves 0.
func (p GraphMemProbe) ExtrapolatedPoolFootprint(totalBuckets int) uint64 {
	if len(p.Deltas) == 0 || totalBuckets <= 0 {
		return 0
	}
	first := p.Deltas[0]
	var perBucket uint64
	if len(p.Deltas) > 1 {
		perBucket = p.Deltas[1]
		if perBucket < MinPerGraphBytes {
			perBucket = MinPerGraphBytes
		}
	}
	return first + uint64(totalBuckets-1)*perBucket
}

// DeriveMaxBlocksWithGraphReserve sizes a paged KV pool from measured memory
// while reserving the CUDA-graph pool footprint: the graph estimate is charged
// against the utilized budget alongside the non-KV bytes, so
// available = floor(freeBytes×utilization) − nonKVBytes − graphReserveBytes,
// blocks = floor(available / bytesPerBlock) — vLLM's
// available_kv_cache_memory_bytes = requested − non_kv −
// cudagraph_estimate_applied (gpu_worker.py:744-750@ae065918640a198a56f6f17eb45e251c3229fe41).
// It fails closed with ErrNonKVExceedsBudget when the combined charge exceeds
// the utilized budget — never a negative block count. A zero reserve reduces
// to DeriveMaxBlocks exactly.
func DeriveMaxBlocksWithGraphReserve(freeBytes, nonKVBytes, graphReserveBytes uint64, utilization float64, bytesPerBlock uint64) (int, error) {
	if !(utilization > 0 && utilization <= 1) {
		return 0, ErrUtilizationOutOfRange
	}
	if bytesPerBlock == 0 {
		return 0, ErrZeroBlockSize
	}
	utilized := uint64(math.Floor(float64(freeBytes) * utilization))
	if utilization == 1 {
		utilized = freeBytes
	}
	nonKVTotal := nonKVBytes
	if graphReserveBytes > math.MaxUint64-nonKVBytes {
		// Saturate instead of wrapping: an absurd reserve must refuse, not
		// wrap past the budget.
		nonKVTotal = math.MaxUint64
	} else {
		nonKVTotal += graphReserveBytes
	}
	if nonKVTotal > utilized {
		return 0, ErrNonKVExceedsBudget
	}
	blocks := (utilized - nonKVTotal) / bytesPerBlock
	if blocks > math.MaxInt {
		return math.MaxInt, nil
	}
	return int(blocks), nil
}

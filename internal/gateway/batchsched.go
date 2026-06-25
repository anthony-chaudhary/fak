package gateway

// batchsched.go — DYNAMIC, PADDING-AWARE batch composition for multi-request prefill
// (issue #272 / B-008, "Dynamic Batching Optimization"). This is the gateway-side
// *scheduling policy* that decides HOW to group concurrently-pending requests into the
// rectangular panels the in-kernel batched prefill (internal/model) can actually
// accelerate — the two scope items "Dynamic batch size selection" and "Padding
// minimization", expressed as a pure, deterministic, host-tractable function.
//
// WHY THIS EXISTS — the concrete tension it resolves. The kernel's batched-decode lane
// (model.StepBatch / StepBatchActive) is already shipped and bit-exact: it amortises the
// memory-bandwidth-bound weight stream across B users, and the ragged StepBatchActive
// already removes idle-lane padding from DECODE (witnessed by model.LastStepMACs, #520).
// The remaining lever B-008 names lives at PREFILL admission. The fast batched prefill
// (model.prefillEachRectF32 / prefillEachRectQ) only fires when EVERY prompt in the panel
// has the SAME length P (model.rectangularPrefillLen); mixed-length prompts fall back to
// per-session prefill and win nothing. To force a mixed batch rectangular you must pad
// every prompt up to the longest, wasting (max·B − Σlenᵢ) token-rows of prefill compute.
// So a scheduler that batches blindly either (a) gets no batching (lengths differ) or
// (b) pays unbounded padding (pad-to-max). This file is the policy that gets batching
// WHILE bounding that padding.
//
// THE GUARANTEE (what the witness proves). ComposeBatches sorts pending requests by
// prompt length and greedily packs each rectangular batch only while its padding overhead
// stays ≤ Policy.MaxPadOverhead (default 0.10) AND its size stays ≤ the dynamic cap. Because
// every emitted batch individually satisfies padᵦ ≤ MaxPadOverhead·rowsᵦ, the AGGREGATE
// padding overhead Σpadᵦ / Σrowsᵦ is ≤ MaxPadOverhead for ANY input distribution — the exact
// "Padding overhead ≤ 10%" acceptance criterion, as a closed-form invariant rather than a
// wall-clock measurement (TestBatchPaddingOverheadInvariant). The composition is index-
// preserving, allocation-light, and depends only on the lengths, so it is byte-deterministic
// across machines — the repo's house form for a metric that must not drift with hardware.
//
// HONEST FENCE — what this is NOT. This is the admission/composition POLICY only: it moves
// no KV, runs no model, and is not yet wired into the live serve request path (the
// production continuous-batching scheduler — paged KV, preemption, fairness, p99 SLA
// admission — is the deferred sibling work tracked by the serving issues and the Track-B
// status note docs/notes/track-b-performance-parity-tracking-306.md). The two remaining
// B-008 acceptance bars — near-linear throughput scaling (1.8× on 2× requests) and p99
// within 1.5× of single-request — are wall-clock measurements that require that scheduler
// on real serving hardware, so they stay deferred; this file closes only the padding-
// minimisation + dynamic-batch-size scope, witnessed deterministically.

import "sort"

// rectPrefillTokenCeiling mirrors the in-kernel rectangular-prefill token ceiling
// (internal/model.batchRectPrefillMaxTokens = 512): a prompt longer than this never takes
// the batched rectangular prefill lane, so the policy never pays padding to drag it into a
// panel — it is scheduled as a serial singleton instead. Kept as a named default so the
// gateway policy stays in step with the kernel without importing an unexported constant.
const rectPrefillTokenCeiling = 512

// BatchPolicy holds the dynamic-batching knobs. The zero value is not meaningful; build one
// with DefaultBatchPolicy and override fields as needed.
type BatchPolicy struct {
	// MaxBatch is the upper bound on requests per rectangular batch (the static ceiling the
	// dynamic size selection clamps against). Mirrors the B the kernel benchmarks exercise.
	MaxBatch int
	// MaxPadOverhead is the per-batch padding-overhead ceiling in [0,1): a batch is never
	// extended past this. The aggregate overhead is then ≤ this by construction. Default 0.10
	// is the B-008 acceptance bar.
	MaxPadOverhead float64
	// MaxPromptLen is the longest prompt eligible for batched rectangular prefill; longer
	// prompts are emitted as serial singletons. Defaults to the kernel ceiling.
	MaxPromptLen int
}

// DefaultBatchPolicy returns the shipping defaults: batches of up to 32 requests, padding
// overhead capped at the 10% acceptance bar, prompts up to the kernel's rectangular ceiling.
func DefaultBatchPolicy() BatchPolicy {
	return BatchPolicy{
		MaxBatch:       32,
		MaxPadOverhead: 0.10,
		MaxPromptLen:   rectPrefillTokenCeiling,
	}
}

// DynamicBatchSize selects the batch size to admit for the current queue depth: it adapts to
// how many requests are actually pending, clamped to [1, MaxBatch]. This is the "dynamic
// batch size selection" lever — a shallow queue admits a small batch (no waiting for a full
// panel that will not arrive), a deep queue fills up to the cap. Composition then refines the
// admitted set by length so the chosen size is also padding-bounded.
func (p BatchPolicy) DynamicBatchSize(pending int) int {
	limit := p.MaxBatch
	if limit < 1 {
		limit = 1
	}
	if pending < 1 {
		return 1
	}
	if pending < limit {
		return pending
	}
	return limit
}

// PaddingOverhead returns the exact fraction of a rectangular [len(lengths), max] prefill
// panel's token-rows that are padding: (max·B − Σlenᵢ) / (max·B). It is 0 for an empty set,
// a single request, or any set of equal lengths, and approaches 1 as the lengths spread.
func PaddingOverhead(lengths []int) float64 {
	if len(lengths) == 0 {
		return 0
	}
	max, sum := 0, 0
	for _, l := range lengths {
		if l > max {
			max = l
		}
		sum += l
	}
	rows := max * len(lengths)
	if rows <= 0 {
		return 0
	}
	return float64(rows-sum) / float64(rows)
}

// ComposeBatches groups the pending requests (given by their prompt lengths) into rectangular
// batches under the policy and returns, for each batch, the ORIGINAL indices of its members.
// The union of the returned groups is exactly {0..len(lengths)-1} (every request is scheduled
// exactly once). Algorithm: oversized prompts (> MaxPromptLen) and non-positive lengths become
// serial singletons; the rest are sorted by length and greedily packed while padding overhead
// stays ≤ MaxPadOverhead and size stays ≤ the dynamic cap. Greedy-over-sorted is optimal for
// this bound because padding only grows as the batch's max grows. Deterministic for a given
// input (stable sort on (length, index)).
func (p BatchPolicy) ComposeBatches(lengths []int) [][]int {
	if len(lengths) == 0 {
		return nil
	}
	limit := p.DynamicBatchSize(len(lengths))
	maxLen := p.MaxPromptLen
	if maxLen < 1 {
		maxLen = rectPrefillTokenCeiling
	}

	var batches [][]int
	// Eligible (batchable) indices, sorted by (length, original index) for determinism.
	eligible := make([]int, 0, len(lengths))
	for i, l := range lengths {
		if l <= 0 || l > maxLen {
			batches = append(batches, []int{i}) // serial singleton: no padding by definition
			continue
		}
		eligible = append(eligible, i)
	}
	sort.SliceStable(eligible, func(a, b int) bool {
		if lengths[eligible[a]] != lengths[eligible[b]] {
			return lengths[eligible[a]] < lengths[eligible[b]]
		}
		return eligible[a] < eligible[b]
	})

	for i := 0; i < len(eligible); {
		// Open a batch at i; its panel max is the largest member's length. Because eligible is
		// sorted ascending, extending the batch can only raise the max, so the running overhead
		// is monotone in the batch's right edge — a clean greedy stop.
		batch := []int{eligible[i]}
		sum := lengths[eligible[i]]
		j := i + 1
		for j < len(eligible) && len(batch) < limit {
			cand := eligible[j]
			newMax := lengths[cand] // sorted: this is the new panel max
			newSum := sum + lengths[cand]
			newRows := newMax * (len(batch) + 1)
			over := float64(newRows-newSum) / float64(newRows)
			if over > p.MaxPadOverhead {
				break
			}
			batch = append(batch, cand)
			sum = newSum
			j++
		}
		batches = append(batches, batch)
		i = j
	}
	return batches
}

// BatchPlanStats summarises a composition for observability and for the witness.
type BatchPlanStats struct {
	NumRequests          int     // total requests scheduled
	NumBatches           int     // number of rectangular batches emitted (incl. singletons)
	NumBatched           int     // requests that landed in a batch of size ≥ 2 (got the batching win)
	AggregatePadOverhead float64 // Σ padding rows / Σ panel rows across all batches
	WorstPadOverhead     float64 // the largest single-batch padding overhead
}

// BatchedFraction is the share of requests that landed in a batch of size ≥ 2 — the requests
// that actually get the shared-weight-stream throughput win. A composition that bounds padding
// only by emitting all singletons would score 0 here; a healthy policy keeps it high.
func (s BatchPlanStats) BatchedFraction() float64 {
	if s.NumRequests == 0 {
		return 0
	}
	return float64(s.NumBatched) / float64(s.NumRequests)
}

// StatsFor computes the BatchPlanStats for a composition over the given lengths.
func StatsFor(lengths []int, batches [][]int) BatchPlanStats {
	st := BatchPlanStats{NumRequests: len(lengths), NumBatches: len(batches)}
	var totalRows, totalPad int
	for _, b := range batches {
		if len(b) >= 2 {
			st.NumBatched += len(b)
		}
		ls := make([]int, len(b))
		for k, idx := range b {
			ls[k] = lengths[idx]
		}
		over := PaddingOverhead(ls)
		if over > st.WorstPadOverhead {
			st.WorstPadOverhead = over
		}
		max := 0
		sum := 0
		for _, l := range ls {
			if l > max {
				max = l
			}
			sum += l
		}
		if rows := max * len(ls); rows > 0 {
			totalRows += rows
			totalPad += rows - sum
		}
	}
	if totalRows > 0 {
		st.AggregatePadOverhead = float64(totalPad) / float64(totalRows)
	}
	return st
}

// PlanBatches is the one-call entry point: compose padding-bounded batches for the pending
// prompt lengths and return both the index groups and their summary stats.
func (p BatchPolicy) PlanBatches(lengths []int) ([][]int, BatchPlanStats) {
	batches := p.ComposeBatches(lengths)
	return batches, StatsFor(lengths, batches)
}

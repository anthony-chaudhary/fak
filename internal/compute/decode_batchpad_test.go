package compute

import (
	"math"
	"testing"
)

// decode_batchpad_test.go — the host-runnable rungs for issue #5852. None of them need the GGUF
// model, the L4 node, or a wall clock: the padding side of #5852 is an exact count, and the two
// measured walls are read from the already-executed #5819 witness rather than re-run.
//
// The chain is: (1) the static policy REPRODUCES the committed witness's 94/144 slot-token
// accounting from the fixture's own generator, which is what makes the model's baseline
// trustworthy rather than asserted; (2) both mechanisms #5852 names really do cut padding, and
// compaction dominates sub-bucketing on queue tax at equal padding; (3) the measured 0.539209x
// factors exactly into a scheduling term and a kernel term; (4) the zero-padding CEILING is
// 0.826x, so the whole family of padding-only fixes is refused against tuned sequential — the
// null result #5852's witness clause admits, decided before a scheduler is written.

// witnessLaneLengths rebuilds the #5819 fixture's decode lanes from the generator
// cmd/microcontextdemo/batch_execution.go uses: 24 jobs, job i carries 2+i%5 tokens and lands in
// compatibility class i%3 (three model/tool/bucket classes the planner keeps isolated). Rebuilding
// it from the generator rather than hardcoding 94/144 is the point — the witness's headline
// numbers have to FALL OUT of the model, or the model is not describing the witness.
func witnessLaneLengths() [][]int {
	classes := make([][]int, 3)
	for i := 0; i < 24; i++ {
		classes[i%3] = append(classes[i%3], 2+i%5)
	}
	return classes
}

// Measured constants from experiments/microcontext/s4e-gcp-inkernel-compat-batch-pass-2026-08-07.json
// (schema fak-microcontext-compat-batch-execution/1, verdict PASS, Qwen2.5-0.5B-Instruct Q8_0 on
// the GCP-sanctioned controlled node, CPU-reference path). These are the ONLY numbers here that a
// host cannot derive; everything else is arithmetic over them.
const (
	witnessBatchWallMS      = 29756.985165
	witnessSequentialWallMS = 16045.241721
	witnessRatio            = 0.5392092522824632
	witnessPaddingTax       = 0.3472222222222222
	witnessUsefulSteps      = 94
	witnessAllocatedSteps   = 144
	witnessBatchWidth       = 8
)

// TestStaticSlotScheduleReproducesWitness is the load-bearing rung. The static full-width policy,
// fed only the fixture's generator and the physical batch width, must land on the witness's
// executed 94 useful / 144 allocated / 34.7222% padding. If it does not, the model is not
// describing the run that produced the 0.539209x, and every downstream projection is void.
func TestStaticSlotScheduleReproducesWitness(t *testing.T) {
	s := StaticSlotSchedule(witnessLaneLengths(), witnessBatchWidth)
	if s.Lanes != 24 || s.Batches != 3 {
		t.Fatalf("#5852: schedule shape = %d lanes / %d batches, want 24 / 3", s.Lanes, s.Batches)
	}
	if s.UsefulSteps != witnessUsefulSteps || s.AllocatedSteps != witnessAllocatedSteps {
		t.Fatalf("#5852: static schedule = %d useful / %d allocated, want witness %d / %d",
			s.UsefulSteps, s.AllocatedSteps, witnessUsefulSteps, witnessAllocatedSteps)
	}
	if math.Abs(s.PaddingFrac-witnessPaddingTax) > 1e-12 {
		t.Fatalf("#5852: static padding %.16f != witness real_padding_tax %.16f", s.PaddingFrac, witnessPaddingTax)
	}
	t.Logf("#5852 baseline (static width %d): %d useful / %d allocated, padding %.6f, %d batches, queue tax %d slot-steps",
		witnessBatchWidth, s.UsefulSteps, s.AllocatedSteps, s.PaddingFrac, s.Batches, s.HeadOfLineSlotSteps)
}

// TestBothMechanismsCutPadding pins #5852's actual ask: length-aware sub-bucketing and
// active-lane compaction each reduce padding against the baseline, neither invents or drops
// useful work, and ideal compaction hits the 0-padding floor by construction.
func TestBothMechanismsCutPadding(t *testing.T) {
	classes := witnessLaneLengths()
	base := StaticSlotSchedule(classes, witnessBatchWidth)
	for _, tc := range []struct {
		name      string
		got       DecodeSlotSchedule
		wantAlloc int
	}{
		{"sub-bucket 4", LengthBucketedSlotSchedule(classes, witnessBatchWidth, 4), 116},
		{"sub-bucket 2", LengthBucketedSlotSchedule(classes, witnessBatchWidth, 2), 102},
		{"compaction gran 4", CompactedSlotSchedule(classes, witnessBatchWidth, 4), 116},
		{"compaction ideal", CompactedSlotSchedule(classes, witnessBatchWidth, 1), 94},
	} {
		if tc.got.UsefulSteps != base.UsefulSteps || tc.got.Lanes != base.Lanes {
			t.Fatalf("#5852 %s: changed the work (%d useful / %d lanes, want %d / %d)",
				tc.name, tc.got.UsefulSteps, tc.got.Lanes, base.UsefulSteps, base.Lanes)
		}
		if tc.got.AllocatedSteps != tc.wantAlloc {
			t.Fatalf("#5852 %s: allocated %d slot-steps, want %d", tc.name, tc.got.AllocatedSteps, tc.wantAlloc)
		}
		if tc.got.PaddingFrac >= base.PaddingFrac {
			t.Fatalf("#5852 %s: padding %.6f did not improve on baseline %.6f", tc.name, tc.got.PaddingFrac, base.PaddingFrac)
		}
		t.Logf("#5852 %s: %d allocated, padding %.6f, %d batches, queue tax %d slot-steps",
			tc.name, tc.got.AllocatedSteps, tc.got.PaddingFrac, tc.got.Batches, tc.got.HeadOfLineSlotSteps)
	}
	if got := CompactedSlotSchedule(classes, witnessBatchWidth, 1).PaddingFrac; got != 0 {
		t.Fatalf("#5852: ideal compaction padding = %v, want the 0 floor", got)
	}
}

// TestSubBucketingBuysPaddingWithQueueTax is the trade #5852 asks to see reported separately.
// Sub-bucketing only cuts padding by splitting a class into more, narrower batches, and under the
// serial executor the witness ran, every extra batch is head-of-line wait. Compaction reaches the
// SAME 116 slot-steps as sub-bucket-4 without splitting a single batch, so it strictly dominates
// on queue tax — the concrete reason #5852's two candidate mechanisms are not interchangeable.
func TestSubBucketingBuysPaddingWithQueueTax(t *testing.T) {
	classes := witnessLaneLengths()
	base := StaticSlotSchedule(classes, witnessBatchWidth)
	b4 := LengthBucketedSlotSchedule(classes, witnessBatchWidth, 4)
	b2 := LengthBucketedSlotSchedule(classes, witnessBatchWidth, 2)
	c4 := CompactedSlotSchedule(classes, witnessBatchWidth, 4)
	if !(b4.HeadOfLineSlotSteps > base.HeadOfLineSlotSteps && b2.HeadOfLineSlotSteps > b4.HeadOfLineSlotSteps) {
		t.Fatalf("#5852: queue tax should rise as sub-buckets narrow: base=%d sub4=%d sub2=%d",
			base.HeadOfLineSlotSteps, b4.HeadOfLineSlotSteps, b2.HeadOfLineSlotSteps)
	}
	if !(b4.Batches > base.Batches && b2.Batches > b4.Batches) {
		t.Fatalf("#5852: sub-bucketing should split batches: base=%d sub4=%d sub2=%d", base.Batches, b4.Batches, b2.Batches)
	}
	if c4.AllocatedSteps != b4.AllocatedSteps {
		t.Fatalf("#5852: compare compaction and sub-bucketing at equal padding, got %d vs %d", c4.AllocatedSteps, b4.AllocatedSteps)
	}
	if c4.Batches != base.Batches || c4.HeadOfLineSlotSteps >= b4.HeadOfLineSlotSteps {
		t.Fatalf("#5852: compaction should hit sub4's padding without its queue tax: batches=%d (base %d), tax=%d (sub4 %d)",
			c4.Batches, base.Batches, c4.HeadOfLineSlotSteps, b4.HeadOfLineSlotSteps)
	}
	t.Logf("#5852 queue tax at equal padding (%d allocated): compaction %d slot-steps in %d batches vs sub-bucketing %d in %d",
		c4.AllocatedSteps, c4.HeadOfLineSlotSteps, c4.Batches, b4.HeadOfLineSlotSteps, b4.Batches)
}

// TestWitnessRatioFactorsExactly pins the identity the whole verdict rests on: the measured
// 0.539209x is exactly 1/(PaddingFactor x SlotCostFactor). The factoring is what separates the
// term #5852 can move (scheduling, 1.5319x) from the term it cannot (the batched kernel's own
// per-slot cost, 1.2106x) — and it recovers the witness's published ratio to float precision, so
// it is a decomposition of the real run and not a parallel model.
func TestWitnessRatioFactorsExactly(t *testing.T) {
	b := DecomposeBatchVsSequential(witnessUsefulSteps, witnessAllocatedSteps, witnessBatchWallMS, witnessSequentialWallMS)
	if math.Abs(b.Ratio-witnessRatio) > 1e-12 {
		t.Fatalf("#5852: decomposed ratio %.16f != witness %.16f", b.Ratio, witnessRatio)
	}
	if got := 1 / (b.PaddingFactor * b.SlotCostFactor); math.Abs(got-b.Ratio) > 1e-12 {
		t.Fatalf("#5852: 1/(padding %.6f x slotcost %.6f) = %.16f, want ratio %.16f",
			b.PaddingFactor, b.SlotCostFactor, got, b.Ratio)
	}
	if b.SlotCostFactor <= 1 {
		t.Fatalf("#5852: slot-cost factor %.6f <= 1 would mean the kernel amortizes and padding is the whole story", b.SlotCostFactor)
	}
	t.Logf("#5852 factoring: padding %.6fx x slot-cost %.6fx -> %.6fx (slot %.3f ms vs tuned sequential token %.3f ms)",
		b.PaddingFactor, b.SlotCostFactor, b.Ratio, b.MSPerSlotStep, b.MSPerSequentialToken)
}

// TestPaddingOnlyFixCannotReachParity is #5852's "keep the change only if fak claim-check grades a
// net gain" gate, decided on the host. Even PERFECT packing — zero padding, every allocated slot
// useful — leaves the CPU-reference batch at 0.826x tuned sequential, because 21% of the deficit
// is per-slot kernel cost that no scheduler touches. Every candidate policy is bounded by that
// ceiling. This is the null result, and it is the reason no sub-bucketing scheduler is landed
// against this witness rather than an untested claim of one.
func TestPaddingOnlyFixCannotReachParity(t *testing.T) {
	classes := witnessLaneLengths()
	b := DecomposeBatchVsSequential(witnessUsefulSteps, witnessAllocatedSteps, witnessBatchWallMS, witnessSequentialWallMS)
	if b.PaddingFixCanReachParity() {
		t.Fatalf("#5852: zero-padding ceiling %.6fx claims parity; re-derive before landing a scheduler", b.ZeroPaddingRatio)
	}
	if math.Abs(b.ZeroPaddingRatio-0.8260226843476034) > 1e-9 {
		t.Fatalf("#5852: zero-padding ceiling %.16f drifted from the witness-derived 0.8260226843476034", b.ZeroPaddingRatio)
	}
	for _, tc := range []struct {
		name string
		s    DecodeSlotSchedule
	}{
		{"static (witness)", StaticSlotSchedule(classes, witnessBatchWidth)},
		{"sub-bucket 4", LengthBucketedSlotSchedule(classes, witnessBatchWidth, 4)},
		{"sub-bucket 2", LengthBucketedSlotSchedule(classes, witnessBatchWidth, 2)},
		{"compaction gran 4", CompactedSlotSchedule(classes, witnessBatchWidth, 4)},
		{"compaction ideal", CompactedSlotSchedule(classes, witnessBatchWidth, 1)},
	} {
		p := b.ProjectRatio(tc.s)
		if p <= 0 {
			t.Fatalf("#5852 %s: projection rejected (%v)", tc.name, p)
		}
		if p > b.ZeroPaddingRatio+1e-12 {
			t.Fatalf("#5852 %s: projection %.6f exceeds the zero-padding ceiling %.6f", tc.name, p, b.ZeroPaddingRatio)
		}
		if p >= 1 {
			t.Fatalf("#5852 %s: projection %.6f claims a net gain the ceiling forbids", tc.name, p)
		}
		t.Logf("#5852 projected %s: %d allocated -> %.6fx tuned sequential (optimistic; slot cost held at the width-%d measurement)",
			tc.name, tc.s.AllocatedSteps, p, witnessBatchWidth)
	}
	t.Logf("#5852 VERDICT: zero-padding ceiling %.6fx < 1.0 — no length-aware sub-bucketing or active-lane compaction policy reaches tuned sequential on this CPU-reference witness", b.ZeroPaddingRatio)
}

// TestSlotScheduleGuards pins the degenerate inputs so a caller cannot get a confident number out
// of nonsense: a sub-bucket wider than the physical batch is clamped instead of over-packing, and
// a projection onto a workload the measurement did not cover is refused rather than rescaled.
func TestSlotScheduleGuards(t *testing.T) {
	classes := witnessLaneLengths()
	wide := LengthBucketedSlotSchedule(classes, witnessBatchWidth, 99)
	if wide.Batches != 3 || wide.AllocatedSteps != witnessAllocatedSteps {
		t.Fatalf("#5852: sub >= width should clamp to the physical width, got %d batches / %d allocated",
			wide.Batches, wide.AllocatedSteps)
	}
	b := DecomposeBatchVsSequential(witnessUsefulSteps, witnessAllocatedSteps, witnessBatchWallMS, witnessSequentialWallMS)
	if got := b.ProjectRatio(StaticSlotSchedule([][]int{{1, 2, 3}}, 4)); got != 0 {
		t.Fatalf("#5852: projecting a foreign workload onto this wall returned %v, want a refusal", got)
	}
	if zero := DecomposeBatchVsSequential(0, 144, witnessBatchWallMS, witnessSequentialWallMS); zero.Ratio != 0 {
		t.Fatalf("#5852: degenerate decomposition returned ratio %v, want the zero value", zero.Ratio)
	}
	if got := CompactedSlotSchedule(nil, witnessBatchWidth, 1); got.Batches != 0 || got.AllocatedSteps != 0 {
		t.Fatalf("#5852: empty class set produced %d batches / %d allocated", got.Batches, got.AllocatedSteps)
	}
}

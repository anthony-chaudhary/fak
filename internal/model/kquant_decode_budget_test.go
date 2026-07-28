package model

import (
	"encoding/binary"
	"testing"
)

// These are the failure-first tests for the #4974 worker half on a q4_k_m MIXTURE.
//
// #4625 capped the Q4_K batch-1 decode GEMV to the witnessed knee (many-core amd64 →
// workers/4, ≤64) and wired it into q4kMatRowsInto. But a q4_k_m artifact is not pure Q4_K:
// the mix leaves ffn_down / lm_head (and mixed-quant routed experts) as Q5_K/Q6_K, and THOSE
// bytes are streamed every decode step by kQuantMatRowsInto — which took the global
// numWorkers. So the ordinary path only HALF-selected the witnessed regime: part of every
// token still ran in the uncapped all-workers regime the cap exists to dodge, and nothing
// failed to say so. These pin the whole artifact onto one knee.

// restoreKQuantSDOT snapshots the int8-gate override so a test can force either kernel branch
// and leave the package global as it found it.
func restoreKQuantSDOT(t *testing.T) {
	t.Helper()
	prev := kQuantSDOTForce
	t.Cleanup(func() { kQuantSDOTForce = prev })
}

// q6kTestTensor builds a Q6_K tensor whose decoded weights are finite: random super-block
// payloads with the trailing f16 `d` scale pinned to 1.0 (the established pattern in
// quant_kquant_int8_test.go), so bit-for-bit comparison is meaningful (no NaN).
func q6kTestTensor(out, in int, seed uint64) *kQuantTensor {
	nblk := in / qkK
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, seed)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[bb-2:], f16One)
		}
	}
	return quantizeKQuantFromRaw(raw, out, in, kindQ6K)
}

func testActivation(in int) []float32 {
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23) - 11
	}
	return x
}

// TestKQuantDecodeWorkersTakesTheWitnessedMixtureKnee is the core #4974 worker-half guarantee:
// the Q5_K/Q6_K decode GEMV takes the SAME bounded budget as the Q4_K decode GEMV, because the
// DA33 sweep witnessed ONE knee for the whole artifact (64 workers on a 256-thread / 8-NUMA
// host), not one knee per kernel. Before this, kQuantMatRowsInto ran at the global numWorkers.
func TestKQuantDecodeWorkersTakesTheWitnessedMixtureKnee(t *testing.T) {
	if got, want := KQuantDecodeWorkers(), Q4KDecodeWorkers(); got != want {
		t.Errorf("KQuantDecodeWorkers()=%d, want %d (=Q4KDecodeWorkers): the q4_k_m Q5_K/Q6_K "+
			"minority is streamed on the same decode step as the Q4_K majority, so it must take "+
			"the same witnessed knee — a second budget would re-open the uncapped regime for "+
			"part of every token", got, want)
	}
}

// TestKQuantDecodeBudgetDerivesFromHostShapeAndStaysOverrideable pins done-condition 2: the
// selected worker regime is DERIVED from runtime topology (worker width, GOOS/GOARCH) rather
// than hardcoded to DA33, and an explicit operator budget still wins. KQuantDecodeWorkers
// delegates to the same pure derivation, so the shape table is the k-quant contract too.
func TestKQuantDecodeBudgetDerivesFromHostShapeAndStaysOverrideable(t *testing.T) {
	shapes := []struct {
		workers    int
		goos       string
		goarch     string
		want       int
		wantCapped bool
	}{
		{12, "linux", "amd64", 12, false},   // small amd64 — untouched
		{63, "linux", "amd64", 63, false},   // just below the many-core threshold
		{256, "linux", "amd64", 64, true},   // the DA33 shape → the witnessed 64-worker knee
		{512, "linux", "amd64", 64, true},   // wider host, same ceiling
		{16, "darwin", "arm64", 6, true},    // a different topology picks a different regime
		{256, "linux", "arm64", 256, false}, // non-amd64 many-core is NOT assumed to share the knee
	}
	for _, s := range shapes {
		got, source := q4kDecodeWorkersFor(s.workers, defaultWorkerBudgetSource, s.goos, s.goarch)
		if got != s.want {
			t.Errorf("%s/%s workers=%d -> k-quant decode budget %d, want %d",
				s.goos, s.goarch, s.workers, got, s.want)
		}
		if capped := source != defaultWorkerBudgetSource; capped != s.wantCapped {
			t.Errorf("%s/%s workers=%d -> capped=%v (source=%q), want %v",
				s.goos, s.goarch, s.workers, capped, source, s.wantCapped)
		}
	}

	// Overrideable: an explicit FAK_WORKERS/FAK_BUDGET/-budget source bypasses the cap on the
	// exact host shape that would otherwise be capped, so an operator tuning DA33 by hand is
	// never silently overruled by the default regime.
	for _, source := range []string{"FAK_WORKERS=256", "FAK_BUDGET=1.0", "-budget=1.0"} {
		got, gotSource := q4kDecodeWorkersFor(256, source, "linux", "amd64")
		if got != 256 || gotSource != source {
			t.Errorf("source=%s -> k-quant decode budget %d (source=%q), want 256/%s",
				source, got, gotSource, source)
		}
	}
}

// TestKQuantMatRowsIntoWorkersIsBitIdenticalAcrossBudgets is the trunk-safety property that
// makes capping the decode GEMV a free change: the [lo,hi) row split is worker-count
// independent, and each output row is computed by exactly one worker in a fixed accumulation
// order, so narrowing the budget must not move a single bit of the answer. Both kernel
// branches (int8 reducer and f32 dequant-dot) are held to it.
func TestKQuantMatRowsIntoWorkersIsBitIdenticalAcrossBudgets(t *testing.T) {
	restoreKQuantSDOT(t)
	const out, in = 137, 512 // odd row count, 2 super-blocks per row: exercises the split tail
	qt := q6kTestTensor(out, in, 0x0fedcba987654321)
	x := testActivation(in)

	for _, branch := range []struct {
		name string
		int8 bool
	}{{"int8", true}, {"f32", false}} {
		t.Run(branch.name, func(t *testing.T) {
			setKQuantSDOTForTest(branch.int8)

			ref := make([]float32, out)
			kQuantMatRowsIntoWorkers(qt, x, ref, 1) // serial reference

			for _, workers := range []int{2, 3, 8, 64, 256} {
				got := make([]float32, out)
				kQuantMatRowsIntoWorkers(qt, x, got, workers)
				for o := 0; o < out; o++ {
					if got[o] != ref[o] {
						t.Fatalf("workers=%d row %d: %v, want %v (serial): capping the decode "+
							"budget must not change the answer", workers, o, got[o], ref[o])
					}
				}
			}

			// The ordinary decode entry point is the capped budget applied to that same body.
			got := make([]float32, out)
			kQuantMatRowsInto(qt, x, got)
			for o := 0; o < out; o++ {
				if got[o] != ref[o] {
					t.Fatalf("kQuantMatRowsInto row %d: %v, want %v (serial)", o, got[o], ref[o])
				}
			}
		})
	}
}

// TestKQuantMatRowsIntoBatchStaysBitIdenticalToPerTokenDecode guards the seam this change cut:
// the batched PREFILL fallback loops the per-token GEMV at the FULL numWorkers width (prefill
// is not the memory-bound batch-1 shape the decode cap targets), so it must still produce
// exactly what the decode path produces token by token.
func TestKQuantMatRowsIntoBatchStaysBitIdenticalToPerTokenDecode(t *testing.T) {
	restoreKQuantSDOT(t)
	setKQuantSDOTForTest(true) // force the int8 kind so the batch variant takes the fallback loop
	const out, in, P = 33, 256, 3
	qt := q6kTestTensor(out, in, 0x1234567890abcdef)

	X := make([]float32, P*in)
	for tok := 0; tok < P; tok++ {
		copy(X[tok*in:(tok+1)*in], testActivation(in))
		X[tok*in] += float32(tok) // make the token columns distinguishable
	}

	Y := make([]float32, P*out)
	kQuantMatRowsIntoBatch(qt, X, P, Y)

	for tok := 0; tok < P; tok++ {
		want := make([]float32, out)
		kQuantMatRowsIntoWorkers(qt, X[tok*in:(tok+1)*in], want, numWorkers)
		for o := 0; o < out; o++ {
			if Y[tok*out+o] != want[o] {
				t.Fatalf("token %d row %d: batch=%v, per-token=%v", tok, o, Y[tok*out+o], want[o])
			}
		}
	}
}

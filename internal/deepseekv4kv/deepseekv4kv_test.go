package deepseekv4kv

import (
	"math"
	"strings"
	"testing"
)

// The self-check must pass on the shipped constants; if it ever fails, the block
// accounting has drifted from the published V4 rates and every downstream figure is
// suspect — so this is the first thing the fixture asserts.
func TestValidateBlockAccountingPassesClosed(t *testing.T) {
	if err := ValidateBlockAccounting(); err != nil {
		t.Fatalf("block accounting must validate against published V4 constants: %v", err)
	}
}

// Block accounting: exact compressed footprints, per layer, in dense-entry units.
func TestSubCacheUnitsExact(t *testing.T) {
	const seq = 1 << 20 // 1M tokens
	cases := []struct {
		k    Kind
		want float64
	}{
		{KindCSA, float64(seq) / 4},   // rate-4
		{KindHCA, float64(seq) / 128}, // rate-128
		{KindTail, float64(seq)},      // dense
		{KindSWA, SWAWindow},          // window-bounded, NOT seq-scaled
	}
	for _, c := range cases {
		if got := SubCacheUnits(c.k, seq); got != c.want {
			t.Errorf("SubCacheUnits(%s, %d) = %g, want %g", c.k, seq, got, c.want)
		}
	}
}

// The load-bearing SWA invariant: storage is bounded by the window and does NOT grow
// with context length. Exact below the window, saturated at/above it.
func TestSWAWindowBounded(t *testing.T) {
	for _, seq := range []int{0, 1, 64, 127, 128, 129, Ctx128K, Ctx512K, Ctx1M} {
		got := SubCacheUnits(KindSWA, seq)
		want := float64(SWAWindow)
		if seq < SWAWindow {
			want = float64(seq)
		}
		if got != want {
			t.Errorf("SWA at seq=%d = %g, want %g (window %d)", seq, got, want, SWAWindow)
		}
	}
	// The whole point: SWA storage at 1M equals SWA storage at 128K.
	if SubCacheUnits(KindSWA, Ctx1M) != SubCacheUnits(KindSWA, Ctx128K) {
		t.Fatal("SWA storage must not grow with context length")
	}
}

// Amplification: the tradeoff must be monotonic and the storage figures must show the
// real finding — a 128-token SWA window makes storage barely move across policies, so
// the cost lives in write (full pays it) vs recompute (zero/periodic pay it).
func TestAmplificationTradeoff(t *testing.T) {
	const ckpt = 4096
	for _, seq := range ReportContexts {
		full := Amplify(seq, FullSWACache, ckpt)
		per := Amplify(seq, PeriodicCheckpoint, ckpt)
		zero := Amplify(seq, ZeroSWACache, ckpt)

		// Write amplification: full >= periodic >= zero, strictly at 1M scale.
		if !(full.WriteAmp > per.WriteAmp && per.WriteAmp > zero.WriteAmp) {
			t.Errorf("seq=%d write-amp not full>periodic>zero: %.4f %.4f %.4f",
				seq, full.WriteAmp, per.WriteAmp, zero.WriteAmp)
		}
		// Recompute: full pays none; zero recomputes the whole window; periodic is bounded by it.
		if full.RecomputeTokens != 0 {
			t.Errorf("seq=%d full-cache must not recompute, got %d", seq, full.RecomputeTokens)
		}
		if zero.RecomputeTokens != SWAWindow {
			t.Errorf("seq=%d zero-cache must recompute the window %d, got %d", seq, SWAWindow, zero.RecomputeTokens)
		}
		if per.RecomputeTokens > SWAWindow {
			t.Errorf("seq=%d periodic recompute must be window-bounded, got %d", seq, per.RecomputeTokens)
		}
		// Storage barely moves: SWA is bounded, so even full-cache storage is within 1%% of zero.
		if full.StorageAmp-zero.StorageAmp > 0.01 {
			t.Errorf("seq=%d storage-amp gap too large (SWA should be bounded): full=%.5f zero=%.5f",
				seq, full.StorageAmp, zero.StorageAmp)
		}
		if zero.StorageAmp != 1.0 || zero.WriteAmp != 1.0 || zero.ReadAmp != 1.0 {
			t.Errorf("seq=%d zero-cache must be the 1.0 baseline: %+v", seq, zero)
		}
	}
}

// An independently-derived golden: at 1M, base = 1048576/4 + 1048576/128 = 270336, and
// full-cache writes an extra `seq` units, so the write numerator is 270336 + 1048576.
func TestGolden1MFullWrite(t *testing.T) {
	const seq = Ctx1M
	const base = seq/4 + seq/128 // 270336, computed independently of the package
	a := Amplify(seq, FullSWACache, 4096)
	if base != 270336 {
		t.Fatalf("test arithmetic wrong: base=%d", base)
	}
	wantNumerator := float64(base + seq)
	if gotNumerator := a.WriteAmp * float64(base); math.Abs(gotNumerator-wantNumerator) > 1e-3 {
		t.Errorf("full write-amp numerator = %.3f, want %.3f", gotNumerator, wantNumerator)
	}
}

func TestReportShape(t *testing.T) {
	rows := Report(4096)
	if len(rows) != len(ReportContexts)*3 {
		t.Fatalf("report must be %d contexts × 3 policies = %d rows, got %d",
			len(ReportContexts), len(ReportContexts)*3, len(rows))
	}
	out := FormatReport(4096)
	for _, must := range []string{"128K", "512K", "1M", "full-swa-cache", "periodic-checkpoint", "zero-swa-cache", "storage×"} {
		if !strings.Contains(out, must) {
			t.Errorf("formatted report missing %q\n%s", must, out)
		}
	}
}

func TestCheckpointClampAndPeriodicMonotonicInN(t *testing.T) {
	// Coarser checkpointing (larger N) writes less; the clamp keeps N>=1 safe.
	a := Amplify(Ctx1M, PeriodicCheckpoint, 1024)
	b := Amplify(Ctx1M, PeriodicCheckpoint, 8192)
	if !(a.WriteAmp > b.WriteAmp) {
		t.Errorf("larger checkpoint interval must write less: N=1024 %.4f vs N=8192 %.4f", a.WriteAmp, b.WriteAmp)
	}
	if got := Amplify(Ctx1M, PeriodicCheckpoint, 0); got.WriteAmp <= 1.0 {
		t.Errorf("checkpointEvery=0 must clamp to 1 (max writes), got write-amp %.4f", got.WriteAmp)
	}
}

package compute

import "testing"

// f32 and q8 per-token costs for kvCfg768 (2 layers * 4 kv heads * 8 dims = 32 elems/row):
//
//	f32: 2 layers * 32 elems * 3 rows * 4 bytes                       = 768
//	q8 : 2 layers * (32*4 Kraw-f32 + 2 * (32 + 1*2) q8_0)  = 2 * 196  = 392
const (
	kvCfg768F32PerToken = 768
	kvCfg768Q8PerToken  = 392
)

// TestEstimateKVStoreBytesF32ParityUnchanged locks the F32 default to the exact bytes the
// hardcoded `3 rows * 4 bytes` produced before the precision tier existed, so adding the knob
// can never silently move an existing planner's arithmetic.
func TestEstimateKVStoreBytesF32ParityUnchanged(t *testing.T) {
	if got := EstimateKVStoreBytes(kvCfg768, 1); got != kvCfg768F32PerToken {
		t.Fatalf("default (f32) per-token KV bytes = %d, want %d (parity with the pre-tier 3x4 layout)", got, kvCfg768F32PerToken)
	}
	explicit := kvCfg768
	explicit.Precision = KVPrecisionF32
	if got := EstimateKVStoreBytes(explicit, 1); got != kvCfg768F32PerToken {
		t.Fatalf("explicit f32 per-token KV bytes = %d, want %d", got, kvCfg768F32PerToken)
	}
	// The zero value of KVPrecision must BE the f32 tier — a config that never set it.
	if (KVPrecision(0)) != KVPrecisionF32 {
		t.Fatal("KVPrecision zero value must be KVPrecisionF32 for backward compatibility")
	}
}

// TestQ8KVFitsMeasurablyLargerContext is the planner-measurable half of the issue's
// acceptance: the same model + box (a fixed KV budget) fits a measurably larger context with
// q8 KV than f32, because q8's per-token cost is strictly smaller.
func TestQ8KVFitsMeasurablyLargerContext(t *testing.T) {
	q8 := kvCfg768
	q8.Precision = KVPrecisionQ8
	if got := EstimateKVStoreBytes(q8, 1); got != kvCfg768Q8PerToken {
		t.Fatalf("q8 per-token KV bytes = %d, want %d", got, kvCfg768Q8PerToken)
	}

	// Same budget: q8 must fit strictly more tokens than f32.
	const budget = int64(kvCfg768F32PerToken) * 512 // room for exactly 512 f32 tokens
	f32Fit := budget / EstimateKVStoreBytes(kvCfg768, 1)
	q8Fit := budget / EstimateKVStoreBytes(q8, 1)
	if q8Fit <= f32Fit {
		t.Fatalf("q8 must fit a larger context than f32 on the same budget: q8=%d f32=%d", q8Fit, f32Fit)
	}
	// The win is ~2x (not the naive 4x of quantizing every row), because evict correctness
	// pins the pre-RoPE K row at f32. Assert it landed in a sane (1.5x, 2.5x) band.
	ratio := float64(q8Fit) / float64(f32Fit)
	if ratio < 1.5 || ratio > 2.5 {
		t.Fatalf("q8/f32 context ratio = %.2f, want ~2x in (1.5, 2.5)", ratio)
	}
}

// TestQ8KVPreservesExactEvictRow encodes the issue's correctness constraint: quantizing must
// preserve evict correctness, which fak does by keeping the pre-RoPE K row at f32. The q8
// per-token cost must therefore stay at or above that full f32 Kraw row — proving the tier did
// NOT quantize all three rows (which would be the naive 1/4 cost and break exact re-positioning).
func TestQ8KVPreservesExactEvictRow(t *testing.T) {
	q8 := kvCfg768
	q8.Precision = KVPrecisionQ8
	q8PerToken := EstimateKVStoreBytes(q8, 1)

	// The pre-RoPE K row alone, at f32: layers * elems * 4 bytes = 2 * 32 * 4 = 256.
	krawF32Floor := int64(kvCfg768.NumLayers) * int64(kvCfg768.NumKVHeads*kvCfg768.HeadDim) * 4
	if q8PerToken < krawF32Floor {
		t.Fatalf("q8 per-token (%d) dropped below the f32 pre-RoPE K floor (%d) — exact evict would be lost", q8PerToken, krawF32Floor)
	}
	// Naive all-rows-quantized would be ~1/4 of f32; q8 must be above it (it keeps a full f32 row).
	naiveAllQuantized := int64(kvCfg768F32PerToken) / 4
	if q8PerToken <= naiveAllQuantized {
		t.Fatalf("q8 per-token (%d) <= naive all-rows-q8 (%d) — the exact-evict f32 row was not retained", q8PerToken, naiveAllQuantized)
	}
}

// TestAutoSelectKVPrecisionKeepsF32WhenItFits: f32 already fits the window, so the auto-selector
// keeps the exact tier rather than needlessly going lossy.
func TestAutoSelectKVPrecisionKeepsF32WhenItFits(t *testing.T) {
	budget := int64(kvCfg768F32PerToken) * 1000 // room for 1000 f32 tokens
	got, reason := AutoSelectKVPrecision(kvCfg768, budget, 500)
	if got != KVPrecisionF32 {
		t.Fatalf("f32 fits the window, want KVPrecisionF32, got %s (%s)", got, reason)
	}
	if reason == "" {
		t.Fatal("auto-select must return a non-empty reason for the operator log")
	}
}

// TestAutoSelectKVPrecisionPicksQ8WhenF32TooTight: f32 would force a context below the desired
// window, q8 lifts it — the auto-selector steps down and names the tradeoff.
func TestAutoSelectKVPrecisionPicksQ8WhenF32TooTight(t *testing.T) {
	// Budget fits 408 f32 tokens but the window wants 1000.
	budget := int64(kvCfg768Q8PerToken) * 800
	got, reason := AutoSelectKVPrecision(kvCfg768, budget, 1000)
	if got != KVPrecisionQ8 {
		t.Fatalf("f32 too tight for the window, want KVPrecisionQ8, got %s (%s)", got, reason)
	}
	f32Fit := budget / EstimateKVStoreBytes(kvCfg768, 1)
	q8Fit := budget / kvCfg768Q8PerToken
	if !(f32Fit < 1000 && q8Fit > f32Fit) {
		t.Fatalf("test setup invalid: want f32Fit(%d) < 1000 and q8Fit(%d) > f32Fit", f32Fit, q8Fit)
	}
	if reason == "" {
		t.Fatal("auto-select must return a non-empty reason naming the tradeoff")
	}
}

// TestAutoSelectKVPrecisionFailsOpen: incomplete geometry, no budget, or no window all keep the
// exact f32 tier — the policy never invents a denser tier from missing evidence.
func TestAutoSelectKVPrecisionFailsOpen(t *testing.T) {
	bad := KVConfig{NumLayers: 2, HeadDim: 8} // NumKVHeads == 0 -> per-token 0
	if got, _ := AutoSelectKVPrecision(bad, 1<<40, 1000); got != KVPrecisionF32 {
		t.Fatalf("incomplete geometry must keep f32, got %s", got)
	}
	if got, _ := AutoSelectKVPrecision(kvCfg768, 0, 1000); got != KVPrecisionF32 {
		t.Fatalf("zero budget must keep f32, got %s", got)
	}
	if got, _ := AutoSelectKVPrecision(kvCfg768, 1<<40, 0); got != KVPrecisionF32 {
		t.Fatalf("zero want must keep f32, got %s", got)
	}
	// Budget too small for even one token of either tier: keep the exact tier.
	if got, _ := AutoSelectKVPrecision(kvCfg768, 100, 10); got != KVPrecisionF32 {
		t.Fatalf("a budget below one token of either tier must keep f32, got %s", got)
	}
}

// TestParseKVPrecision covers the serve-path selector tokens and the typo refusal.
func TestParseKVPrecision(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want KVPrecision
	}{
		{"", KVPrecisionF32},
		{"f32", KVPrecisionF32},
		{"q8", KVPrecisionQ8},
		{"q8_0", KVPrecisionQ8},
	} {
		got, err := ParseKVPrecision(tc.in)
		if err != nil || got != tc.want {
			t.Fatalf("ParseKVPrecision(%q) = (%s, %v), want (%s, nil)", tc.in, got, err, tc.want)
		}
	}
	if _, err := ParseKVPrecision("bogus"); err == nil {
		t.Fatal("ParseKVPrecision(\"bogus\") must error so a typo refuses rather than silently picking a tier")
	}
}

// TestKVPrecisionLabels locks the selector token and the MemoryPlan storage label per tier.
func TestKVPrecisionLabels(t *testing.T) {
	if KVPrecisionF32.String() != "f32" || KVPrecisionQ8.String() != "q8" {
		t.Fatalf("tier tokens = (%s, %s), want (f32, q8)", KVPrecisionF32, KVPrecisionQ8)
	}
	if KVPrecisionF32.StorageLabel() != "f32" {
		t.Fatalf("f32 storage label = %q, want f32", KVPrecisionF32.StorageLabel())
	}
	if KVPrecisionQ8.StorageLabel() != "mixed" {
		t.Fatalf("q8 storage label = %q, want mixed (f32 Kraw + q8_0 K/V)", KVPrecisionQ8.StorageLabel())
	}
}

// TestEstimateKVStoreMemoryPlanCarriesTierDType: the classed plan's DType reflects the tier, so
// a fit refusal / operator surface names the real storage, not always "f32".
func TestEstimateKVStoreMemoryPlanCarriesTierDType(t *testing.T) {
	f32Plan := EstimateKVStoreMemoryPlan(kvCfg768, 16)
	if len(f32Plan) != 1 || f32Plan[0].DType != "f32" {
		t.Fatalf("f32 plan DType = %+v, want a single f32 demand", f32Plan)
	}
	q8 := kvCfg768
	q8.Precision = KVPrecisionQ8
	q8Plan := EstimateKVStoreMemoryPlan(q8, 16)
	if len(q8Plan) != 1 || q8Plan[0].DType != "mixed" {
		t.Fatalf("q8 plan DType = %+v, want a single mixed demand", q8Plan)
	}
	if q8Plan[0].Bytes >= f32Plan[0].Bytes {
		t.Fatalf("q8 plan bytes (%d) must be smaller than f32 (%d)", q8Plan[0].Bytes, f32Plan[0].Bytes)
	}
}

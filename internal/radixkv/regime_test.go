package radixkv

import (
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func sampleRegime() Regime {
	return Regime{
		ModelID:     "meta-llama/Llama-3-8B",
		ModelSHA:    "a1b2c3d4e5",
		DType:       DTypeFP16,
		QuantPolicy: QuantPolicyNone,
		RoPE: RoPEConfig{
			Base:        10000.0,
			Scale:       1.0,
			ScalingType: "linear",
			Dim:         64,
		},
	}
}

func newTestKV(val float32) *model.KVCache {
	kv := model.NewKVCache(model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1})
	kv.K[0] = []float32{val}
	kv.V[0] = []float32{val}
	return kv
}

// TestRegime_KeyAndHash verifies that identical regimes produce identical keys and hashes,
// and that derivation is deterministic and independent of struct field construction order.
func TestRegime_KeyAndHash(t *testing.T) {
	r1 := sampleRegime()
	r2 := Regime{
		RoPE: RoPEConfig{
			Dim:         64,
			ScalingType: "linear",
			Scale:       1.0,
			Base:        10000.0,
		},
		QuantPolicy: QuantPolicyNone,
		DType:       DTypeFP16,
		ModelSHA:    "a1b2c3d4e5",
		ModelID:     "meta-llama/Llama-3-8B",
	}

	if r1.RegimeKey() != r2.RegimeKey() {
		t.Fatalf("identical regimes must produce identical RegimeKey: %q vs %q", r1.RegimeKey(), r2.RegimeKey())
	}
	if r1.Hash() != r2.Hash() {
		t.Fatalf("identical regimes must produce identical Hash: %x vs %x", r1.Hash(), r2.Hash())
	}
	if r1.HashHex() != r2.HashHex() {
		t.Fatalf("identical regimes must produce identical HashHex: %s vs %s", r1.HashHex(), r2.HashHex())
	}
	if ok, axis := r1.Match(r2); !ok || axis != AxisNone {
		t.Fatalf("identical regimes must match: ok=%v axis=%q", ok, axis)
	}
	if !r1.Reusable(r2) {
		t.Fatal("identical regimes must be reusable")
	}
}

// TestRegime_EveryAxisIsLoadBearing verifies that mutating ANY single axis of the regime
// (model ID, model SHA, dtype, quant policy, or any RoPE field) changes the RegimeKey, Hash,
// and causes Match to fail with the specific divergent axis.
func TestRegime_EveryAxisIsLoadBearing(t *testing.T) {
	base := sampleRegime()

	cases := []struct {
		name     string
		mutate   func(Regime) Regime
		wantAxis MismatchAxis
	}{
		{
			name:     "model_id",
			mutate:   func(r Regime) Regime { r.ModelID = "meta-llama/Llama-3-70B"; return r },
			wantAxis: AxisModel,
		},
		{
			name:     "model_sha",
			mutate:   func(r Regime) Regime { r.ModelSHA = "different-sha-999"; return r },
			wantAxis: AxisModel,
		},
		{
			name:     "dtype_fp8",
			mutate:   func(r Regime) Regime { r.DType = DTypeFP8; return r },
			wantAxis: AxisDtype,
		},
		{
			name:     "dtype_int4",
			mutate:   func(r Regime) Regime { r.DType = DTypeINT4; return r },
			wantAxis: AxisDtype,
		},
		{
			name:     "dtype_bf16",
			mutate:   func(r Regime) Regime { r.DType = DTypeBF16; return r },
			wantAxis: AxisDtype,
		},
		{
			name:     "quant_fp8",
			mutate:   func(r Regime) Regime { r.QuantPolicy = QuantPolicyFP8; return r },
			wantAxis: AxisQuant,
		},
		{
			name:     "quant_int4",
			mutate:   func(r Regime) Regime { r.QuantPolicy = QuantPolicyINT4; return r },
			wantAxis: AxisQuant,
		},
		{
			name:     "rope_base",
			mutate:   func(r Regime) Regime { r.RoPE.Base = 500000.0; return r },
			wantAxis: AxisRoPE,
		},
		{
			name:     "rope_scale",
			mutate:   func(r Regime) Regime { r.RoPE.Scale = 4.0; return r },
			wantAxis: AxisRoPE,
		},
		{
			name:     "rope_scaling_type",
			mutate:   func(r Regime) Regime { r.RoPE.ScalingType = "yarn"; return r },
			wantAxis: AxisRoPE,
		},
		{
			name:     "rope_dim",
			mutate:   func(r Regime) Regime { r.RoPE.Dim = 128; return r },
			wantAxis: AxisRoPE,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			mutated := tc.mutate(base)
			if base.RegimeKey() == mutated.RegimeKey() {
				t.Fatalf("axis %s: mutated regime must produce divergent key: %q", tc.name, base.RegimeKey())
			}
			if base.Hash() == mutated.Hash() {
				t.Fatalf("axis %s: mutated regime must produce divergent hash", tc.name)
			}
			ok, axis := base.Match(mutated)
			if ok {
				t.Fatalf("axis %s: mutated regime must not match", tc.name)
			}
			if axis != tc.wantAxis {
				t.Fatalf("axis %s: expected divergent axis %q, got %q", tc.name, tc.wantAxis, axis)
			}
			if base.Reusable(mutated) {
				t.Fatalf("axis %s: mutated regime must not be reusable", tc.name)
			}
			// Symmetric refusal
			if mutated.Reusable(base) {
				t.Fatalf("axis %s: refusal must be symmetric", tc.name)
			}
		})
	}
}

// TestRegime_CompleteAndFailsClosed witnesses that an incomplete regime cannot key
// or lookup any prefix cache, failing closed safely.
func TestRegime_CompleteAndFailsClosed(t *testing.T) {
	var zero Regime
	if zero.Complete() {
		t.Fatal("zero regime must not be complete")
	}

	complete := sampleRegime()
	if ok, axis := complete.Match(zero); ok || axis != AxisIncomplete {
		t.Fatalf("complete regime against zero request must fail with AxisIncomplete: ok=%v axis=%q", ok, axis)
	}
	if ok, axis := zero.Match(complete); ok || axis != AxisIncomplete {
		t.Fatalf("zero regime against complete request must fail with AxisIncomplete: ok=%v axis=%q", ok, axis)
	}

	tree := New(0)
	tokens := []int{101, 2023, 1037}

	// Lookup under incomplete regime fails closed (nil, 0)
	n, matched := tree.LookupRegime(zero, tokens)
	if n != nil || matched != 0 {
		t.Fatalf("LookupRegime under incomplete regime must fail closed: n=%v matched=%d", n, matched)
	}

	// ScopedTree under incomplete regime returns ErrRegimeIncomplete
	scoped := NewScoped(0)
	owner := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}
	if err := scoped.AdmitPrivateRegime(zero, owner, tokens, newTestKV(1.0), nil); err != ErrRegimeIncomplete {
		t.Fatalf("AdmitPrivateRegime under incomplete regime must return ErrRegimeIncomplete: got %v", err)
	}
	_, _, _, _, err := scoped.LookupRegime(zero, owner, tokens)
	if err != ErrRegimeIncomplete {
		t.Fatalf("LookupRegime under incomplete regime must return ErrRegimeIncomplete: got %v", err)
	}
}

// TestRegime_SameRegimeHitsCache witnesses that identical prefix under the identical regime
// hits the cache across Tree, RegimeTree, and ScopedTree.
func TestRegime_SameRegimeHitsCache(t *testing.T) {
	regime := sampleRegime()
	tokens := []int{101, 2023, 1037, 7099, 102}

	// 1. Direct Tree.LookupRegime / InsertRegime
	t.Run("Tree", func(t *testing.T) {
		tree := New(0)
		b, matched := tree.LookupRegime(regime, tokens)
		if matched != 0 {
			t.Fatalf("cold lookup should match 0 tokens, got %d", matched)
		}
		kv := newTestKV(42.0)
		leaf := tree.InsertRegime(regime, b, tokens[matched:], kv)
		tree.Done(leaf)

		// Second lookup with identical regime must HIT
		b2, matched2 := tree.LookupRegime(regime, tokens)
		defer tree.Done(b2)
		if matched2 != len(tokens) {
			t.Fatalf("warm lookup under same regime must hit all tokens: want %d, got %d", len(tokens), matched2)
		}
		if b2.KV() == nil || b2.KV().K[0][0] != 42.0 {
			t.Fatalf("reused KV cache payload corrupted: %v", b2.KV())
		}
		if b2.RegimeKey() != regime.RegimeKey() {
			t.Fatalf("node regime key mismatch: want %q, got %q", regime.RegimeKey(), b2.RegimeKey())
		}
	})

	// 2. Partitioned RegimeTree
	t.Run("RegimeTree", func(t *testing.T) {
		tree := New(0)
		rt := NewRegimeTree(tree, regime)
		b, matched := rt.Lookup(tokens)
		if matched != 0 {
			t.Fatalf("cold lookup matched %d tokens", matched)
		}
		leaf := rt.Insert(b, tokens[matched:], newTestKV(84.0))
		rt.Done(leaf)

		b2, matched2 := rt.Lookup(tokens)
		defer rt.Done(b2)
		if matched2 != len(tokens) {
			t.Fatalf("warm lookup on RegimeTree must hit: want %d, got %d", len(tokens), matched2)
		}
		if b2.KV().K[0][0] != 84.0 {
			t.Fatalf("reused KV mismatch: %v", b2.KV().K[0][0])
		}
	})

	// 3. ScopedTree with AdmitPrivateRegime and LookupRegime
	t.Run("ScopedTree", func(t *testing.T) {
		scoped := NewScoped(0)
		owner := CacheIdentity{Tenant: "tenant-x", Agent: "worker-1"}
		kv := newTestKV(126.0)
		logits := []float32{0.5, 0.5}

		if err := scoped.AdmitPrivateRegime(regime, owner, tokens, kv, logits); err != nil {
			t.Fatalf("AdmitPrivateRegime failed: %v", err)
		}

		gotKV, gotLogits, matched, scope, err := scoped.LookupRegime(regime, owner, tokens)
		if err != nil {
			t.Fatalf("LookupRegime failed: %v", err)
		}
		if matched != len(tokens) || scope != ScopeTenant {
			t.Fatalf("LookupRegime matched=%d scope=%d; want matched=%d scope=%d", matched, scope, len(tokens), ScopeTenant)
		}
		if gotKV.K[0][0] != 126.0 || !reflect.DeepEqual(gotLogits, logits) {
			t.Fatalf("LookupRegime payload mismatch: gotKV=%v logits=%v", gotKV, gotLogits)
		}
	})
}

// TestRegime_DifferentAxesMissCache witnesses that an identical prefix cached under regime A
// cleanly MISSES when demanded under a divergent regime (different RoPE / quant / dtype / modelID),
// structurally preventing silent attention corruption.
func TestRegime_DifferentAxesMissCache(t *testing.T) {
	baseRegime := sampleRegime()
	tokens := []int{101, 2023, 1037, 7099, 102}

	variants := []struct {
		name   string
		mutate func(Regime) Regime
	}{
		{
			name:   "different_model_id",
			mutate: func(r Regime) Regime { r.ModelID = "mistralai/Mistral-7B"; return r },
		},
		{
			name:   "different_model_sha",
			mutate: func(r Regime) Regime { r.ModelSHA = "checkpoint-rev2"; return r },
		},
		{
			name:   "different_dtype_fp8",
			mutate: func(r Regime) Regime { r.DType = DTypeFP8; return r },
		},
		{
			name:   "different_dtype_int4",
			mutate: func(r Regime) Regime { r.DType = DTypeINT4; return r },
		},
		{
			name:   "different_dtype_bf16",
			mutate: func(r Regime) Regime { r.DType = DTypeBF16; return r },
		},
		{
			name:   "different_quant_fp8",
			mutate: func(r Regime) Regime { r.QuantPolicy = QuantPolicyFP8; return r },
		},
		{
			name:   "different_quant_int4",
			mutate: func(r Regime) Regime { r.QuantPolicy = QuantPolicyINT4; return r },
		},
		{
			name:   "different_rope_base",
			mutate: func(r Regime) Regime { r.RoPE.Base = 500000.0; return r },
		},
		{
			name:   "different_rope_scale",
			mutate: func(r Regime) Regime { r.RoPE.Scale = 8.0; return r },
		},
		{
			name:   "different_rope_scaling_type",
			mutate: func(r Regime) Regime { r.RoPE.ScalingType = "llama3"; return r },
		},
		{
			name:   "different_rope_dim",
			mutate: func(r Regime) Regime { r.RoPE.Dim = 128; return r },
		},
	}

	for _, tc := range variants {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			divergentRegime := tc.mutate(baseRegime)

			// Setup shared Tree populated with baseRegime
			tree := New(0)
			b, m := tree.LookupRegime(baseRegime, tokens)
			leaf := tree.InsertRegime(baseRegime, b, tokens[m:], newTestKV(777.0))
			tree.Done(leaf)

			// 1. Tree lookup under divergent regime must cleanly MISS
			divBoundary, divMatched := tree.LookupRegime(divergentRegime, tokens)
			defer tree.Done(divBoundary)
			if divMatched != 0 {
				t.Fatalf("%s: expected 0 matched tokens under divergent regime, got %d", tc.name, divMatched)
			}
			if divBoundary.KV() != nil {
				t.Fatalf("%s: divergent regime must never receive KV bytes from base regime", tc.name)
			}

			// 2. RegimeTree lookup under divergent regime must cleanly MISS
			rtDiv := NewRegimeTree(tree, divergentRegime)
			rtBoundary, rtMatched := rtDiv.Lookup(tokens)
			defer rtDiv.Done(rtBoundary)
			if rtMatched != 0 {
				t.Fatalf("%s: RegimeTree expected 0 matched tokens under divergent regime, got %d", tc.name, rtMatched)
			}
			if rtBoundary.KV() != nil {
				t.Fatalf("%s: RegimeTree divergent regime received cached KV", tc.name)
			}

			// 3. ScopedTree lookup under divergent regime must cleanly MISS
			scoped := NewScoped(0)
			owner := CacheIdentity{Tenant: "tenant-iso", Agent: "worker-1"}
			if err := scoped.AdmitPrivateRegime(baseRegime, owner, tokens, newTestKV(888.0), nil); err != nil {
				t.Fatalf("AdmitPrivateRegime failed: %v", err)
			}
			gotKV, _, scopedMatched, _, err := scoped.LookupRegime(divergentRegime, owner, tokens)
			if err != nil {
				t.Fatalf("LookupRegime returned unexpected error: %v", err)
			}
			if scopedMatched != 0 || gotKV != nil {
				t.Fatalf("%s: ScopedTree expected 0 matched tokens under divergent regime, got %d (kv=%v)", tc.name, scopedMatched, gotKV)
			}
		})
	}
}

// TestRegime_MultiRegimeCoexistenceAndIsolation witnesses that two regimes sharing the exact
// same token path store completely independent KV tensors under the same global budget,
// and evicting a prefix in one regime leaves the other intact.
func TestRegime_MultiRegimeCoexistenceAndIsolation(t *testing.T) {
	tree := New(0)
	tokens := []int{1, 2, 3, 4, 5}

	regimeA := sampleRegime()
	regimeB := sampleRegime()
	regimeB.DType = DTypeFP8
	regimeB.QuantPolicy = QuantPolicyFP8

	// Populate regimeA with value 100
	bA, mA := tree.LookupRegime(regimeA, tokens)
	leafA := tree.InsertRegime(regimeA, bA, tokens[mA:], newTestKV(100.0))
	tree.Done(leafA)

	// Populate regimeB with value 200
	bB, mB := tree.LookupRegime(regimeB, tokens)
	leafB := tree.InsertRegime(regimeB, bB, tokens[mB:], newTestKV(200.0))
	tree.Done(leafB)

	// Lookup regimeA retrieves 100
	hitA, matchedA := tree.LookupRegime(regimeA, tokens)
	defer tree.Done(hitA)
	if matchedA != len(tokens) || hitA.KV().K[0][0] != 100.0 {
		t.Fatalf("regimeA lookup mismatch: matched=%d kv=%v", matchedA, hitA.KV())
	}

	// Lookup regimeB retrieves 200
	hitB, matchedB := tree.LookupRegime(regimeB, tokens)
	defer tree.Done(hitB)
	if matchedB != len(tokens) || hitB.KV().K[0][0] != 200.0 {
		t.Fatalf("regimeB lookup mismatch: matched=%d kv=%v", matchedB, hitB.KV())
	}

	// Evict regimeA's prefix
	freedA := tree.EvictPrefixRegime(regimeA, tokens)
	if freedA == 0 {
		t.Fatal("expected evicted tokens for regimeA")
	}

	// regimeA is now cold
	missA, matchedA2 := tree.LookupRegime(regimeA, tokens)
	defer tree.Done(missA)
	if matchedA2 != 0 {
		t.Fatalf("regimeA should be cold after eviction: matched=%d", matchedA2)
	}

	// regimeB remains fully resident and warm
	hitB2, matchedB2 := tree.LookupRegime(regimeB, tokens)
	defer tree.Done(hitB2)
	if matchedB2 != len(tokens) || hitB2.KV().K[0][0] != 200.0 {
		t.Fatalf("regimeB should remain resident after regimeA eviction: matched=%d kv=%v", matchedB2, hitB2.KV())
	}
}

// TestRegime_ConcurrentLookupsRace tests concurrent lookups, admissions, and matches
// across distinct decode regimes under -race to prove data-race freedom and isolation.
func TestRegime_ConcurrentLookupsRace(t *testing.T) {
	scoped := NewScoped(0)
	regimes := []Regime{
		sampleRegime(),
		func() Regime {
			r := sampleRegime()
			r.DType = DTypeFP8
			r.QuantPolicy = QuantPolicyFP8
			return r
		}(),
		func() Regime {
			r := sampleRegime()
			r.RoPE.Base = 500000.0
			r.RoPE.Scale = 4.0
			return r
		}(),
		func() Regime {
			r := sampleRegime()
			r.ModelID = "deepseek-ai/DeepSeek-V3"
			r.RoPE.ScalingType = "yarn"
			return r
		}(),
	}

	tokens := []int{42, 100, 200, 300, 400}
	owner := CacheIdentity{Tenant: "race-tenant", Agent: "agent-race"}

	// Seed all regimes concurrently
	var wg sync.WaitGroup
	for i, reg := range regimes {
		wg.Add(1)
		go func(idx int, r Regime) {
			defer wg.Done()
			val := float32(idx + 1)
			err := scoped.AdmitPrivateRegime(r, owner, tokens, newTestKV(val), []float32{val})
			if err != nil {
				t.Errorf("concurrent AdmitPrivateRegime failed for regime %d: %v", idx, err)
			}
		}(i, reg)
	}
	wg.Wait()

	// Concurrently query each regime repeatedly
	const concurrency = 16
	const iterations = 50

	for c := 0; c < concurrency; c++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			regIdx := workerID % len(regimes)
			targetRegime := regimes[regIdx]
			expectedVal := float32(regIdx + 1)

			for iter := 0; iter < iterations; iter++ {
				// Query target regime
				kv, logits, matched, scope, err := scoped.LookupRegime(targetRegime, owner, tokens)
				if err != nil {
					t.Errorf("worker %d iter %d LookupRegime failed: %v", workerID, iter, err)
					return
				}
				if matched != len(tokens) || scope != ScopeTenant {
					t.Errorf("worker %d iter %d matched=%d (want %d) scope=%d", workerID, iter, matched, len(tokens), scope)
					return
				}
				if kv == nil || kv.K[0][0] != expectedVal || logits[0] != expectedVal {
					t.Errorf("worker %d iter %d KV corruption: got %v logits=%v (expected %f)", workerID, iter, kv, logits, expectedVal)
					return
				}

				// Query different regime (must not cross-pollinate with targetRegime)
				otherIdx := (regIdx + 1) % len(regimes)
				otherRegime := regimes[otherIdx]
				otherExpected := float32(otherIdx + 1)

				okKV, _, otherMatched, _, err2 := scoped.LookupRegime(otherRegime, owner, tokens)
				if err2 != nil {
					t.Errorf("worker %d iter %d LookupRegime other failed: %v", workerID, iter, err2)
					return
				}
				if otherMatched != len(tokens) || okKV.K[0][0] != otherExpected {
					t.Errorf("worker %d iter %d cross-regime corruption: matched=%d kv=%v (expected %f)", workerID, iter, otherMatched, okKV, otherExpected)
					return
				}
			}
		}(c)
	}
	wg.Wait()
}

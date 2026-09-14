package compute

import (
	"reflect"
	"testing"
)

func tinyContextSizeConfig() ContextSizeConfig {
	return ContextSizeConfig{
		KV: KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 8, RopeTheta: 10000},
		Scratch: TransformerScratchConfig{
			HiddenSize: 32, IntermediateSize: 64, VocabSize: 320,
			NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IncludeLogits: true,
		},
		MaxContext: 4096,
	}
}

// A non-negative override is the explicit context-token count both call sites pass; it
// is used verbatim and the returned plan matches the shared per-context composition.
func TestAutoSizeContextPlanOverrideWinsVerbatim(t *testing.T) {
	cfg := tinyContextSizeConfig()
	for _, tok := range []int{0, 1, 128, 4096, 9000} {
		gotTokens, gotPlan := AutoSizeContextPlan(cfg, nil, FreeUnknown, tok)
		if gotTokens != tok {
			t.Fatalf("override %d: tokens = %d, want verbatim %d", tok, gotTokens, tok)
		}
		if want := cfg.PerContextMemoryPlan(tok); !reflect.DeepEqual(gotPlan, want) {
			t.Fatalf("override %d: plan = %#v, want %#v", tok, gotPlan, want)
		}
	}
}

// A negative override means "unset": fall back to the model's full declared window, and
// to zero context (scratch-only, never a panic) when no window is declared either.
func TestAutoSizeContextPlanUnsetOverrideFallsBackToFullWindow(t *testing.T) {
	cfg := tinyContextSizeConfig()
	gotTokens, gotPlan := AutoSizeContextPlan(cfg, nil, FreeUnknown, -1)
	if gotTokens != cfg.MaxContext {
		t.Fatalf("unset override: tokens = %d, want full window %d", gotTokens, cfg.MaxContext)
	}
	if want := cfg.PerContextMemoryPlan(cfg.MaxContext); !reflect.DeepEqual(gotPlan, want) {
		t.Fatalf("unset override: plan = %#v, want %#v", gotPlan, want)
	}

	cfg.MaxContext = 0
	zeroTokens, zeroPlan := AutoSizeContextPlan(cfg, nil, FreeUnknown, -1)
	if zeroTokens != 0 {
		t.Fatalf("no window + unset override: tokens = %d, want 0", zeroTokens)
	}
	if hasClass(zeroPlan, MemoryKVCache) {
		t.Fatalf("no window + unset override must omit the KV demand: %#v", zeroPlan)
	}
	if want := cfg.PerContextMemoryPlan(0); !reflect.DeepEqual(zeroPlan, want) {
		t.Fatalf("no window + unset override: plan = %#v, want %#v", zeroPlan, want)
	}
}

func hasClass(plan MemoryPlan, class MemoryClass) bool {
	for _, d := range plan {
		if d.Class == class && d.Bytes > 0 {
			return true
		}
	}
	return false
}

// qwen36_27BContextSizeConfig is the Qwen3.6-27B geometry the #1046 acceptance names: 64 layers ×
// 8 KV-heads × 128 head-dim, F32 KV (3 rows × 4 bytes) ⇒ 64*8*128*3*4 = 786432 bytes/token =
// 0.75 MiB/token (kvprecision.go), and a 262144-token declared context window.
func qwen36_27BContextSizeConfig() ContextSizeConfig {
	return ContextSizeConfig{
		KV: KVConfig{NumLayers: 64, NumKVHeads: 8, HeadDim: 128, RopeTheta: 1000000},
		Scratch: TransformerScratchConfig{
			HiddenSize: 5120, IntermediateSize: 17408, VocabSize: 152064,
			NumLayers: 64, NumHeads: 40, NumKVHeads: 8, HeadDim: 128, IncludeLogits: true,
		},
		MaxContext: 262144,
	}
}

// #1046: with no override and a known memory ceiling, AutoSizeContextPlan derives the LARGEST
// context whose KV + scratch fits the budget alongside the fixed weights — NOT the model's full
// declared window. Anchored to the M3-Pro (36 GB unified) / Qwen3.6-27B case the issue names: the
// 262144-token window must auto-size DOWN to the ~16k that fits, never the full window that OOMs.
func TestAutoSizeContextPlanDerivesLargestFittingContext(t *testing.T) {
	cfg := qwen36_27BContextSizeConfig()
	const perToken = int64(786432) // 0.75 MiB/token
	if got := EstimateKVStoreBytes(cfg.KV, 1); got != perToken {
		t.Fatalf("kv/token = %d, want 0.75 MiB (%d) for the Qwen3.6-27B geometry", got, perToken)
	}
	scratch := EstimateHALTransientMemoryPlan(cfg.Scratch).Total()

	// Construct a box where — after a 20 GiB device-resident weight footprint and the per-token
	// scratch — exactly 12 GiB (16384 × 0.75 MiB) of KV budget remains, so the derivation lands on
	// 16384 tokens, far below the 262144-token full window.
	const weightBytes = int64(20) << 30
	const wantTokens = 16384
	weights := MemoryPlan{{Class: MemoryWeights, Bytes: weightBytes, Scope: MemoryScopeDevice}}
	avail := int64(wantTokens)*perToken + weightBytes + scratch

	gotTokens, gotPlan := AutoSizeContextPlan(cfg, weights, avail, -1)
	if gotTokens != wantTokens {
		t.Fatalf("derived tokens = %d, want %d (largest fitting); full window is %d", gotTokens, wantTokens, cfg.MaxContext)
	}
	if gotTokens >= cfg.MaxContext {
		t.Fatalf("auto-fit must size DOWN from the %d-token full window, got %d", cfg.MaxContext, gotTokens)
	}
	if want := cfg.PerContextMemoryPlan(wantTokens); !reflect.DeepEqual(gotPlan, want) {
		t.Fatalf("derived plan = %#v, want per-context plan for %d tokens", gotPlan, wantTokens)
	}
}

func TestAutoSizeContextPlanChargesFixedSessionState(t *testing.T) {
	cfg := tinyContextSizeConfig()
	const fixedBytes = int64(4096)
	cfg.SessionState = MemoryPlan{{
		Class: MemoryKVCache, Bytes: fixedBytes, Detail: "test-recurrent-state", DType: F32.String(),
	}}
	const wantTokens = 1024
	perToken := EstimateKVStoreBytes(cfg.KV, 1)
	scratch := EstimateHALTransientMemoryPlan(cfg.Scratch).Total()
	weights := MemoryPlan{{Class: MemoryWeights, Bytes: 1 << 20}}
	avail := weights.DeviceTotal() + fixedBytes + scratch + int64(wantTokens)*perToken

	gotTokens, plan := AutoSizeContextPlan(cfg, weights, avail, -1)
	if gotTokens != wantTokens {
		t.Fatalf("derived tokens = %d, want %d after subtracting fixed session state", gotTokens, wantTokens)
	}
	var fixedSeen int64
	for _, d := range plan {
		if d.Detail == "test-recurrent-state" {
			fixedSeen += d.Bytes
		}
	}
	if fixedSeen != fixedBytes {
		t.Fatalf("fixed session demand = %d, want %d in per-context plan: %#v", fixedSeen, fixedBytes, plan)
	}
}

// #1046: a cpu-offload serve pins its routed experts in HOST RAM, so they must NOT be charged
// against the DEVICE budget the KV cache competes for. The derivation subtracts only the
// device-scoped weights — host-scoped offload bytes (even when far larger than the device) leave
// the device-side KV fit unchanged.
func TestAutoSizeContextPlanIgnoresHostScopedOffloadWeights(t *testing.T) {
	cfg := qwen36_27BContextSizeConfig()
	const perToken = int64(786432)
	scratch := EstimateHALTransientMemoryPlan(cfg.Scratch).Total()
	const denseBytes = int64(20) << 30
	const wantTokens = 16384
	avail := int64(wantTokens)*perToken + denseBytes + scratch

	deviceOnly := MemoryPlan{{Class: MemoryWeights, Bytes: denseBytes, Scope: MemoryScopeDevice}}
	withExperts := MemoryPlan{
		{Class: MemoryWeights, Bytes: denseBytes, Scope: MemoryScopeDevice},
		{Class: MemoryOffload, Bytes: int64(400) << 30, Scope: MemoryScopeHost}, // 400 GiB host experts
	}
	deviceTokens, _ := AutoSizeContextPlan(cfg, deviceOnly, avail, -1)
	expertTokens, _ := AutoSizeContextPlan(cfg, withExperts, avail, -1)
	if deviceTokens != wantTokens || expertTokens != wantTokens {
		t.Fatalf("host experts must not shrink the device KV fit: device-only=%d, with-experts=%d, want %d", deviceTokens, expertTokens, wantTokens)
	}
}

// #1046/#13036: a known ceiling too small to hold even the weights clamps to the floor
// (MinAutoContextTokens), not 0 — so the plan keeps a small KV demand and the LOAD-TIME fit check,
// not this sizer, refuses a genuinely-too-small box. An explicit override is clamped the SAME way
// (to the floor here, since the derived bound is the floor when the weights over-subscribe) — the
// sizer never emits a KV plan larger than the largest context that fits the known ceiling.
func TestAutoSizeContextPlanFloorAndOverridePolicy(t *testing.T) {
	cfg := tinyContextSizeConfig() // MaxContext 4096, well above the floor
	weights := MemoryPlan{{Class: MemoryWeights, Bytes: 1 << 30, Scope: MemoryScopeDevice}}

	floorTokens, _ := AutoSizeContextPlan(cfg, weights, 1<<20 /* 1 MiB << 1 GiB weights */, -1)
	if floorTokens != MinAutoContextTokens {
		t.Fatalf("weights-overflow derived = %d, want floor %d", floorTokens, MinAutoContextTokens)
	}
	// #13036 clamps the explicit 1234 to the fitted bound too: with the weights alone over-subscribed
	// the derived fitting bound IS the floor, so the plan never carries an over-maximal KV demand that
	// the load-time fit check would have to refuse anyway.
	overrideTokens, _ := AutoSizeContextPlan(cfg, weights, 1<<20, 1234)
	if overrideTokens != MinAutoContextTokens {
		t.Fatalf("explicit override = %d, want clamp to the fitted floor %d when the weights alone overflow the known ceiling", overrideTokens, MinAutoContextTokens)
	}
}

// #13036: an explicit override is consulted against the weights-first auto-fit instead of being
// taken verbatim-and-fatal. (a) An over-maximal 256K override on a box where the weights fit but the
// KV never could is clamped DOWN to the largest fitting context, and the emitted plan's KV bytes stay
// within the residual budget. (b) An override that fits stays verbatim. (c) An unknown ceiling
// (avail <= 0) fails open and keeps the override verbatim.
func TestContextOverrideClamp(t *testing.T) {
	cfg := qwen36_27BContextSizeConfig() // 0.75 MiB/token, MaxContext 262144
	const perToken = int64(786432)
	if got := EstimateKVStoreBytes(cfg.KV, 1); got != perToken {
		t.Fatalf("kv/token = %d, want 0.75 MiB (%d) for the Qwen3.6-27B geometry", got, perToken)
	}
	scratch := EstimateHALTransientMemoryPlan(cfg.Scratch).Total()

	// Box where the 20 GiB device weights fit, leaving exactly 8 GiB of residual budget: 8 GiB /
	// 0.75 MiB = 10922 tokens is the largest fitting context, far below the 256K (262144) override.
	const weightBytes = int64(20) << 30
	const avail = weightBytes + int64(8)<<30
	weights := MemoryPlan{{Class: MemoryWeights, Bytes: weightBytes, Scope: MemoryScopeDevice}}
	wantFit := int((int64(8)<<30 - scratch) / perToken)
	if wantFit >= cfg.MaxContext || wantFit <= MinAutoContextTokens {
		t.Fatalf("fixture does not produce a mid-range fitting bound: %d", wantFit)
	}

	// (a) over-maximal 256K override clamps to the largest fitting context, whose KV bytes fit
	// the residual budget (the budget the derivation subtracted scratch from can back them).
	gotTokens, gotPlan := AutoSizeContextPlan(cfg, weights, avail, 262144)
	if gotTokens != wantFit {
		t.Fatalf("clamped tokens = %d, want largest fitting %d (override 262144, MaxContext %d)", gotTokens, wantFit, cfg.MaxContext)
	}
	if gotTokens >= 262144 {
		t.Fatalf("clamped tokens = %d, must sit below the over-maximal 262144 override", gotTokens)
	}
	kvTotal := gotPlan.ByClass()[MemoryKVCache]
	if kvTotal <= 0 || kvTotal > int64(8)<<30 {
		t.Fatalf("clamped plan KV = %d bytes, want >0 and within the residual 8 GiB budget", kvTotal)
	}
	if want := cfg.PerContextMemoryPlan(wantFit); !reflect.DeepEqual(gotPlan, want) {
		t.Fatalf("clamped plan = %#v, want per-context plan for %d tokens", gotPlan, wantFit)
	}

	// (b) an override that fits the residual budget is honored verbatim.
	fitOverride, gotPlanB := AutoSizeContextPlan(cfg, weights, avail, 4096)
	if fitOverride != 4096 {
		t.Fatalf("fitting override tokens = %d, want verbatim 4096", fitOverride)
	}
	if want := cfg.PerContextMemoryPlan(4096); !reflect.DeepEqual(gotPlanB, want) {
		t.Fatalf("fitting override plan = %#v, want per-context plan for 4096 tokens", gotPlanB)
	}

	// (c) unknown ceiling (avail = FreeUnknown): fail-open, the override stays verbatim.
	unknownTokens, _ := AutoSizeContextPlan(cfg, weights, FreeUnknown, 262144)
	if unknownTokens != 262144 {
		t.Fatalf("unknown-ceiling override tokens = %d, want verbatim 262144 (fail-open when avail <= 0)", unknownTokens)
	}
}

// #13025: LargestFittingContextTokens is the exported form largestFittingContext, and it must
// agree with what AutoSizeContextPlan derives for the same inputs — a refusal site that names
// "the maximum context that WOULD fit" can never contradict an unset-override plan built from
// the same weights/budget. One table, four regimes: fits fully (→ MaxContext), constrained
// (strictly between floor and MaxContext), weights-overflow (→ floor), unprobeable ceiling
// (avail <= 0 → MaxContext, not a floor clamp).
func TestLargestFittingContextParityWithAutoSizePlan(t *testing.T) {
	qwen := qwen36_27BContextSizeConfig() // 0.75 MiB/token, MaxContext 262144
	tiny := tinyContextSizeConfig()       // MaxContext 4096
	scratch := EstimateHALTransientMemoryPlan(qwen.Scratch).Total()
	const perToken = int64(786432)
	if got := EstimateKVStoreBytes(qwen.KV, 1); got != perToken {
		t.Fatalf("kv/token = %d, want 0.75 MiB (%d) for the Qwen3.6-27B geometry", got, perToken)
	}

	weights := MemoryPlan{{Class: MemoryWeights, Bytes: int64(20) << 30, Scope: MemoryScopeDevice}}
	constrainedAvail := int64(16384)*perToken + weights.DeviceTotal() + scratch

	cases := []struct {
		name      string
		cfg       ContextSizeConfig
		weights   MemoryPlan
		avail     int64
		want      int
		wantRange func(t *testing.T, got int, max int)
	}{
		{
			name:    "fully-fitting-budget-returns-max-context",
			cfg:     tiny,
			weights: nil,
			avail:   1 << 30, // 1 GiB ≫ tiny geometry's KV + scratch need
			want:    tiny.MaxContext,
			wantRange: func(t *testing.T, got, max int) {
				if got != max {
					t.Fatalf("fully-fitting budget: got %d, want MaxContext %d", got, max)
				}
			},
		},
		{
			name:    "constrained-budget-strictly-between-floor-and-max",
			cfg:     qwen,
			weights: weights,
			avail:   constrainedAvail,
			want:    16384,
			wantRange: func(t *testing.T, got, max int) {
				if got != 16384 {
					t.Fatalf("constrained budget: got %d, want 16384 (largest fitting)", got)
				}
				if got <= MinAutoContextTokens || got >= max {
					t.Fatalf("constrained budget: got %d, want strictly inside (%d, %d)", got, MinAutoContextTokens, max)
				}
			},
		},
		{
			name:    "tiny-budget-weights-overflow-returns-floor",
			cfg:     tiny,
			weights: MemoryPlan{{Class: MemoryWeights, Bytes: 1 << 30, Scope: MemoryScopeDevice}},
			avail:   1 << 20, // 1 MiB ≪ the 1 GiB weights: floor-clamp regime
			want:    MinAutoContextTokens,
			wantRange: func(t *testing.T, got, _ int) {
				if got != MinAutoContextTokens {
					t.Fatalf("weights-overflow: got %d, want floor %d", got, MinAutoContextTokens)
				}
			},
		},
		{
			name:    "unprobeable-ceiling-fails-open-to-max-context",
			cfg:     tiny,
			weights: nil,
			avail:   FreeUnknown,
			want:    tiny.MaxContext,
			wantRange: func(t *testing.T, got, max int) {
				if got != max {
					t.Fatalf("unprobeable ceiling: got %d, want MaxContext %d (not a floor clamp)", got, max)
				}
			},
		},
	}

	for _, tc := range cases {
		exported := LargestFittingContextTokens(tc.cfg, tc.weights, tc.avail)
		planTokens, _ := AutoSizeContextPlan(tc.cfg, tc.weights, tc.avail, -1) // parity lock
		if exported != planTokens {
			t.Fatalf("%s: LargestFittingContextTokens = %d but AutoSizeContextPlan(unset override) = %d; the exported wrapper and the auto-sizer must not disagree (#13025)", tc.name, exported, planTokens)
		}
		if exported != tc.want {
			t.Fatalf("%s: LargestFittingContextTokens = %d, want %d", tc.name, exported, tc.want)
		}
		tc.wantRange(t, exported, tc.cfg.MaxContext)
	}
}

// #1046: BudgetAfterHeadroom is the single headroom formula the load-time fit check and the context
// auto-sizer share, so a derived context is sized against byte-identically the budget the check
// later enforces.
func TestBudgetAfterHeadroom(t *testing.T) {
	if got := BudgetAfterHeadroom(1000, 0.15); got != 850 {
		t.Fatalf("BudgetAfterHeadroom(1000, 0.15) = %d, want 850", got)
	}
	if got := BudgetAfterHeadroom(1000, 0); got != 1000 {
		t.Fatalf("zero headroom must pass the budget through, got %d", got)
	}
	if got := BudgetAfterHeadroom(1000, 1.5); got != 1000 {
		t.Fatalf("out-of-range headroom must pass the budget through, got %d", got)
	}
	if got := BudgetAfterHeadroom(-5, 0.15); got != -5 {
		t.Fatalf("non-positive budget must pass through unchanged, got %d", got)
	}
}

// #13036 ACCEPTANCE GATE (finding F5): the override path and the negative-override (auto)
// path must agree on the largest fitting context for IDENTICAL (model, host) inputs. The
// verbatim-override tests (TestAutoSizeContextPlanOverrideWinsVerbatim, TestContextOverrideClamp)
// and the auto-derivation test (TestAutoSizeContextPlanDerivesLargestFittingContext) each pin one
// path alone; nothing crossed them. This table does: for every over-maximal explicit override the
// clamped result is compared against what the unset-override (-1) auto path derives from
// byte-identical (cfg, weights, avail) inputs — equal tokens AND a reflect.DeepEqual-equal PLAN
// (the whole plan, not just the token count). Pinned to both fixtures: the Qwen3.6-27B geometry
// (0.75 MiB/token, MaxContext 262144) with the same avail construction as the derivation test
// (20 GiB device weights + scratch + exactly 16384 tokens of KV budget ⇒ largest fitting context
// 16384), and the tiny geometry (MaxContext 4096) against a 1 MiB ceiling with nil weights.
//
// REGIME BOUNDARY — parity holds ONLY in the clamp regime. When an override FITS and is SMALLER
// than the largest fitting context, the override is honored verbatim while the auto path derives
// a LARGER value, so the two intentionally differ in that under-request regime; parity must NOT
// be asserted there. The fail-open regime (avail <= 0, unknown ceiling) is also outside the gate:
// the clamp cannot fire without a known ceiling, so the override stays verbatim and may differ
// from the auto path's full-window fallback.
func TestContextOverrideAndAutoFitAgreeOnIdenticalInputs(t *testing.T) {
	qwen := qwen36_27BContextSizeConfig() // 0.75 MiB/token, MaxContext 262144
	tiny := tinyContextSizeConfig()       // MaxContext 4096
	const perToken = int64(786432)
	if got := EstimateKVStoreBytes(qwen.KV, 1); got != perToken {
		t.Fatalf("kv/token = %d, want 0.75 MiB (%d) for the Qwen3.6-27B geometry", got, perToken)
	}

	// The qwen `avail`, built exactly as TestAutoSizeContextPlanDerivesLargestFittingContext
	// does: 20 GiB device-resident weights + per-token scratch + exactly 16384 tokens of KV,
	// so the largest fitting context is 16384 — far below the declared 262144 window.
	const qwenFit = 16384
	qwenWeights := MemoryPlan{{Class: MemoryWeights, Bytes: int64(20) << 30, Scope: MemoryScopeDevice}}
	qwenAvail := int64(qwenFit)*perToken + qwenWeights.DeviceTotal() + EstimateHALTransientMemoryPlan(qwen.Scratch).Total()

	// The tiny `avail`: sized so the tiny geometry's FULL window fits with no weights. The
	// arithmetic: tiny KV is 384 B/token (2 layers × 2 KV-heads × 8 head-dim × 3 f32 rows ×
	// 4 B) and scratch ≈ 5 KiB, so a 1 MiB ceiling would only fit (1 MiB − scratch)/384 =
	// 2717 tokens — below the 4096 window. A 2 MiB ceiling fits (2 MiB − scratch)/384 = 5448
	// ≥ 4096, so the auto path derives MaxContext itself and both tiny regimes are exercisable.
	tinyAvail := int64(2) << 20
	if tinyFit := LargestFittingContextTokens(tiny, nil, tinyAvail); tinyFit != tiny.MaxContext {
		t.Fatalf("tiny fixture: derived fit = %d, want the full window %d so the whole tiny window fits the 2 MiB ceiling", tinyFit, tiny.MaxContext)
	}
	// The fitting-override arithmetic: 512 tokens need 512*384 + ~5 KiB scratch ≈ 201.6 KiB,
	// far under the 2 MiB ceiling, so 512 fits and stays verbatim in the under-request regime.
	if need := int64(512)*EstimateKVStoreBytes(tiny.KV, 1) + EstimateHALTransientMemoryPlan(tiny.Scratch).Total(); need > tinyAvail {
		t.Fatalf("tiny fixture: 512-token plan needs %d bytes, more than the %d-byte ceiling", need, tinyAvail)
	}

	clampCases := []struct {
		name      string
		cfg       ContextSizeConfig
		weights   MemoryPlan
		avail     int64
		wantFit   int
		overrides []int
	}{
		{
			name:      "qwen-27b-over-maximal-overrides-clamp-to-the-derived-fit",
			cfg:       qwen,
			weights:   qwenWeights,
			avail:     qwenAvail,
			wantFit:   qwenFit,
			overrides: []int{262144, 262144 * 2, 1 << 30},
		},
		{
			name:      "tiny-over-maximal-overrides-clamp-to-the-derived-full-window",
			cfg:       tiny,
			weights:   nil,
			avail:     tinyAvail,
			wantFit:   tiny.MaxContext,
			overrides: []int{4097, 9000, 1 << 20},
		},
	}

	for _, tc := range clampCases {
		autoTokens, autoPlan := AutoSizeContextPlan(tc.cfg, tc.weights, tc.avail, -1)
		if autoTokens != tc.wantFit {
			t.Fatalf("%s: auto path derived %d, want the largest fitting context %d", tc.name, autoTokens, tc.wantFit)
		}
		for _, override := range tc.overrides {
			if override <= tc.wantFit {
				t.Fatalf("%s: fixture override %d is not over-maximal for fit %d", tc.name, override, tc.wantFit)
			}
			gotTokens, gotPlan := AutoSizeContextPlan(tc.cfg, tc.weights, tc.avail, override)

			// THE GATE: clamp-regime parity with the unset-override path on identical inputs —
			// same tokens and a byte-identical (DeepEqual) plan, not merely an equal count.
			if gotTokens != autoTokens {
				t.Fatalf("%s: override %d clamped to %d but the auto path derived %d for identical (weights, avail); the two paths must agree (#13036)", tc.name, override, gotTokens, autoTokens)
			}
			if !reflect.DeepEqual(gotPlan, autoPlan) {
				t.Fatalf("%s: override %d plan = %#v but the auto path plan = %#v for identical inputs; the emitted plans must be identical, not just the token count (#13036)", tc.name, override, gotPlan, autoPlan)
			}
			if gotTokens != tc.wantFit {
				t.Fatalf("%s: override %d resolved to %d, want the largest fitting context %d", tc.name, override, gotTokens, tc.wantFit)
			}
			if gotTokens >= override {
				t.Fatalf("%s: clamped tokens %d must sit strictly below the over-maximal override %d", tc.name, gotTokens, override)
			}
		}
	}

	// UNDER-REQUEST REGIME (NOT the gate): a fitting 512-token override on the tiny fixture
	// stays verbatim while the auto path derives a LARGER context (the full window) from the
	// very same (cfg, weights, avail). This asymmetry is the intended policy — the clamp only
	// ever shrinks — so parity is deliberately NOT asserted here; do not "fix" it.
	fitOverrideTokens, fitOverridePlan := AutoSizeContextPlan(tiny, nil, tinyAvail, 512)
	if fitOverrideTokens != 512 {
		t.Fatalf("fitting override tokens = %d, want verbatim 512 (512 fits the 1 MiB tiny ceiling)", fitOverrideTokens)
	}
	if want := tiny.PerContextMemoryPlan(512); !reflect.DeepEqual(fitOverridePlan, want) {
		t.Fatalf("fitting override plan = %#v, want per-context plan for 512 tokens", fitOverridePlan)
	}
	if autoTokens, _ := AutoSizeContextPlan(tiny, nil, tinyAvail, -1); autoTokens == fitOverrideTokens {
		t.Fatalf("under-request regime: fitting override 512 must differ from the auto-derived %d (the clamp only shrinks, it never expands)", autoTokens)
	}

	// FAIL-OPEN REGIME (unknown ceiling, avail <= 0): an over-maximal override stays verbatim —
	// the unknown-ceiling contract. Parity is not claimed here because the clamp the gate
	// exercises cannot fire when the capacity is unprobeable; the auto path falls back to the
	// full declared window instead.
	unknownTokens, _ := AutoSizeContextPlan(qwen, qwenWeights, FreeUnknown, 1<<30)
	if unknownTokens != 1<<30 {
		t.Fatalf("unknown-ceiling override tokens = %d, want verbatim %d (fail-open: unknown capacity never rewrites an operator's request)", unknownTokens, 1<<30)
	}
}

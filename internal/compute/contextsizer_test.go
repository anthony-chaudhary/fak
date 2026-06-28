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

// #1046: a known ceiling too small to hold even the weights clamps to the floor (MinAutoContextTokens),
// not 0 — so the plan keeps a small KV demand and the LOAD-TIME fit check, not this sizer, refuses a
// genuinely-too-small box. An explicit override still wins verbatim regardless of the ceiling.
func TestAutoSizeContextPlanFloorAndOverridePolicy(t *testing.T) {
	cfg := tinyContextSizeConfig() // MaxContext 4096, well above the floor
	weights := MemoryPlan{{Class: MemoryWeights, Bytes: 1 << 30, Scope: MemoryScopeDevice}}

	floorTokens, _ := AutoSizeContextPlan(cfg, weights, 1<<20 /* 1 MiB << 1 GiB weights */, -1)
	if floorTokens != MinAutoContextTokens {
		t.Fatalf("weights-overflow derived = %d, want floor %d", floorTokens, MinAutoContextTokens)
	}
	overrideTokens, _ := AutoSizeContextPlan(cfg, weights, 1<<20, 1234)
	if overrideTokens != 1234 {
		t.Fatalf("explicit override = %d, want 1234 verbatim even with a known ceiling", overrideTokens)
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

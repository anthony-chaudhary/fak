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

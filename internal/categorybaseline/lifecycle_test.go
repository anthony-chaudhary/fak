package categorybaseline

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLifecycleFullCycle(t *testing.T) {
	root := t.TempDir()

	// Initial load on clean root returns empty normalized registry.
	reg := Load(root)
	if reg.Schema != Schema {
		t.Fatalf("expected schema %q, got %q", Schema, reg.Schema)
	}
	if len(reg.Categories) != 0 {
		t.Fatalf("expected 0 categories, got %d", len(reg.Categories))
	}

	// Upsert first category.
	catA := Category{
		Name:           "engine",
		Layers:         []string{"l1-cache", "l2-cache", "l3-cache"},
		CompletedLayer: "l1-cache",
		NextLayer:      "l2-cache",
		Witness:        "fak check engine-l1",
	}
	var ok bool
	reg, ok = Upsert(reg, catA)
	if !ok {
		t.Fatal("upsert catA failed")
	}
	if len(reg.Categories) != 1 {
		t.Fatalf("expected 1 category, got %d", len(reg.Categories))
	}

	// Upsert second category out-of-alphabetical order to verify sorting.
	catB := Category{
		Name:           "adjudicator",
		Layers:         []string{"spec", "core", "extended"},
		CompletedLayer: "spec",
		NextLayer:      "core",
		Witness:        "fak check adjudicator-spec",
	}
	reg, ok = Upsert(reg, catB)
	if !ok {
		t.Fatal("upsert catB failed")
	}
	if len(reg.Categories) != 2 {
		t.Fatalf("expected 2 categories, got %d", len(reg.Categories))
	}
	// Verify sorted order.
	if reg.Categories[0].Name != "adjudicator" || reg.Categories[1].Name != "engine" {
		t.Fatalf("categories not sorted: %+v", reg.Categories)
	}

	// Save to disk and reload.
	if err := Save(root, reg); err != nil {
		t.Fatalf("save failed: %v", err)
	}
	loaded := Load(root)
	if !reflect.DeepEqual(reg, loaded) {
		t.Fatalf("loaded mismatch:\nwant: %+v\ngot:  %+v", reg, loaded)
	}

	// Evaluate category decisions.
	dec1 := Evaluate(loaded, "engine", "l1-cache", false)
	if !dec1.Hold || dec1.NextLayer != "l2-cache" || dec1.CompletedLayer != "l1-cache" {
		t.Fatalf("expected hold on l1-cache, got: %+v", dec1)
	}
	dec2 := Evaluate(loaded, "engine", "l2-cache", false)
	if dec2.Hold {
		t.Fatalf("expected no hold on l2-cache, got: %+v", dec2)
	}

	// Update existing category: advance baseline to l2-cache.
	catAUpdated := Category{
		Name:           "engine",
		Layers:         []string{"l1-cache", "l2-cache", "l3-cache"},
		CompletedLayer: "l2-cache",
		NextLayer:      "l3-cache",
		Witness:        "fak check engine-l2",
	}
	reg, ok = Upsert(loaded, catAUpdated)
	if !ok {
		t.Fatal("upsert update failed")
	}
	if len(reg.Categories) != 2 {
		t.Fatalf("expected 2 categories after update, got %d", len(reg.Categories))
	}

	// Re-evaluate: l2-cache is now held, l3-cache is next.
	dec3 := Evaluate(reg, "engine", "l2-cache", false)
	if !dec3.Hold || dec3.NextLayer != "l3-cache" {
		t.Fatalf("expected hold on l2-cache after update, got: %+v", dec3)
	}

	// Remove category.
	reg = Remove(reg, "adjudicator")
	if len(reg.Categories) != 1 || reg.Categories[0].Name != "engine" {
		t.Fatalf("expected only engine remaining, got: %+v", reg.Categories)
	}

	// Evaluate removed category: fails open to no hold.
	dec4 := Evaluate(reg, "adjudicator", "spec", false)
	if dec4.Hold {
		t.Fatalf("expected removed category to not hold, got: %+v", dec4)
	}

	// Removing non-existent category is a safe no-op.
	reg = Remove(reg, "non-existent")
	if len(reg.Categories) != 1 {
		t.Fatalf("expected 1 category after no-op remove, got %d", len(reg.Categories))
	}
}

func TestCalculationAndComparison(t *testing.T) {
	reg := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "inference",
				Layers:         []string{"cpu-scalar", "cpu-simd", "gpu-fp16", "gpu-quant"},
				CompletedLayer: "cpu-simd",
				NextLayer:      "gpu-fp16",
				Witness:        "bench-simd-v2",
			},
		},
	})

	tests := []struct {
		name       string
		category   string
		layer      string
		regression bool
		wantHold   bool
		wantNext   string
	}{
		{
			name:       "work below completed layer is held",
			category:   "inference",
			layer:      "cpu-scalar",
			regression: false,
			wantHold:   true,
			wantNext:   "gpu-fp16",
		},
		{
			name:       "work at completed layer is held",
			category:   "inference",
			layer:      "cpu-simd",
			regression: false,
			wantHold:   true,
			wantNext:   "gpu-fp16",
		},
		{
			name:       "work at next layer is not held",
			category:   "inference",
			layer:      "gpu-fp16",
			regression: false,
			wantHold:   false,
		},
		{
			name:       "work beyond next layer is not held",
			category:   "inference",
			layer:      "gpu-quant",
			regression: false,
			wantHold:   false,
		},
		{
			name:       "regression work on completed layer is never held",
			category:   "inference",
			layer:      "cpu-simd",
			regression: true,
			wantHold:   false,
		},
		{
			name:       "regression work below completed layer is never held",
			category:   "inference",
			layer:      "cpu-scalar",
			regression: true,
			wantHold:   false,
		},
		{
			name:       "unknown layer fails open",
			category:   "inference",
			layer:      "quantum-tensor",
			regression: false,
			wantHold:   false,
		},
		{
			name:       "unknown category fails open",
			category:   "unknown-cat",
			layer:      "cpu-scalar",
			regression: false,
			wantHold:   false,
		},
		{
			name:       "blank strings fail open",
			category:   "",
			layer:      "",
			regression: false,
			wantHold:   false,
		},
		{
			name:       "name normalization handles uppercase underscores and spaces",
			category:   " INFERENCE ",
			layer:      "CPU_SIMD",
			regression: false,
			wantHold:   true,
			wantNext:   "gpu-fp16",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dec := Evaluate(reg, tc.category, tc.layer, tc.regression)
			if dec.Hold != tc.wantHold {
				t.Fatalf("Hold mismatch: got %v, want %v (decision: %+v)", dec.Hold, tc.wantHold, dec)
			}
			if tc.wantHold && dec.NextLayer != tc.wantNext {
				t.Fatalf("NextLayer mismatch: got %q, want %q", dec.NextLayer, tc.wantNext)
			}
			if tc.wantHold && dec.Witness != "bench-simd-v2" {
				t.Fatalf("Witness mismatch: got %q, want %q", dec.Witness, "bench-simd-v2")
			}
		})
	}
}

func TestValidationAndNormalizationEdgeCases(t *testing.T) {
	// Case 1: Inverted layer order (NextLayer before CompletedLayer) should be dropped.
	r1 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "inverted",
				Layers:         []string{"alpha", "beta", "gamma"},
				CompletedLayer: "gamma",
				NextLayer:      "beta",
				Witness:        "wit",
			},
		},
	})
	if len(r1.Categories) != 0 {
		t.Fatalf("expected inverted category dropped, got: %+v", r1.Categories)
	}

	// Case 2: CompletedLayer equals NextLayer should be dropped.
	r2 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "equal-layers",
				Layers:         []string{"alpha", "beta"},
				CompletedLayer: "alpha",
				NextLayer:      "alpha",
				Witness:        "wit",
			},
		},
	})
	if len(r2.Categories) != 0 {
		t.Fatalf("expected equal layers category dropped, got: %+v", r2.Categories)
	}

	// Case 3: Missing witness should be dropped.
	r3 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "no-witness",
				Layers:         []string{"alpha", "beta"},
				CompletedLayer: "alpha",
				NextLayer:      "beta",
				Witness:        "   ",
			},
		},
	})
	if len(r3.Categories) != 0 {
		t.Fatalf("expected empty witness category dropped, got: %+v", r3.Categories)
	}

	// Case 4: Missing completed layer in layers slice should be dropped.
	r4 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "missing-completed",
				Layers:         []string{"beta", "gamma"},
				CompletedLayer: "alpha",
				NextLayer:      "beta",
				Witness:        "wit",
			},
		},
	})
	if len(r4.Categories) != 0 {
		t.Fatalf("expected missing completed category dropped, got: %+v", r4.Categories)
	}

	// Case 5: Missing next layer in layers slice should be dropped.
	r5 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "missing-next",
				Layers:         []string{"alpha", "beta"},
				CompletedLayer: "alpha",
				NextLayer:      "omega",
				Witness:        "wit",
			},
		},
	})
	if len(r5.Categories) != 0 {
		t.Fatalf("expected missing next category dropped, got: %+v", r5.Categories)
	}

	// Case 6: Duplicate category name keeps first valid entry.
	r6 := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "duplicate-cat",
				Layers:         []string{"alpha", "beta"},
				CompletedLayer: "alpha",
				NextLayer:      "beta",
				Witness:        "first-witness",
			},
			{
				Name:           "DUPLICATE_CAT",
				Layers:         []string{"alpha", "beta"},
				CompletedLayer: "alpha",
				NextLayer:      "beta",
				Witness:        "second-witness",
			},
		},
	})
	if len(r6.Categories) != 1 || r6.Categories[0].Witness != "first-witness" {
		t.Fatalf("expected first duplicate retained, got: %+v", r6.Categories)
	}

	// Case 7: Upsert with invalid category returns false and preserves registry.
	initial := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "valid",
				Layers:         []string{"one", "two"},
				CompletedLayer: "one",
				NextLayer:      "two",
				Witness:        "witness",
			},
		},
	})
	after, ok := Upsert(initial, Category{
		Name:           "invalid-candidate",
		Layers:         []string{"one"},
		CompletedLayer: "one",
		NextLayer:      "two",
		Witness:        "w",
	})
	if ok || len(after.Categories) != 1 || after.Categories[0].Name != "valid" {
		t.Fatalf("expected upsert refusal and unmodified registry: ok=%v, reg=%+v", ok, after)
	}
}

func TestLoadCorruptAndMismatchedSchema(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, filepath.FromSlash(DefaultPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}

	// Test malformed JSON.
	if err := os.WriteFile(path, []byte("{corrupt-json"), 0o644); err != nil {
		t.Fatal(err)
	}
	rCorrupt := Load(root)
	if rCorrupt.Schema != Schema || len(rCorrupt.Categories) != 0 {
		t.Fatalf("expected empty registry on corrupt json, got: %+v", rCorrupt)
	}

	// Test mismatched schema version.
	badSchemaPayload, err := json.Marshal(map[string]interface{}{
		"schema": "fak-category-baselines/999",
		"categories": []Category{
			{
				Name:           "future",
				Layers:         []string{"a", "b"},
				CompletedLayer: "a",
				NextLayer:      "b",
				Witness:        "w",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, badSchemaPayload, 0o644); err != nil {
		t.Fatal(err)
	}
	rBadSchema := Load(root)
	if rBadSchema.Schema != Schema || len(rBadSchema.Categories) != 0 {
		t.Fatalf("expected empty registry on bad schema, got: %+v", rBadSchema)
	}
}

func BenchmarkCategoryBaseline(b *testing.B) {
	reg := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "inference",
				Layers:         []string{"cpu-scalar", "cpu-simd", "gpu-fp16", "gpu-quant"},
				CompletedLayer: "cpu-simd",
				NextLayer:      "gpu-fp16",
				Witness:        "bench-simd-v2",
			},
			{
				Name:           "cache",
				Layers:         []string{"memory", "disk", "distributed"},
				CompletedLayer: "memory",
				NextLayer:      "disk",
				Witness:        "cache-bench-v1",
			},
		},
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec1 := Evaluate(reg, "inference", "cpu-simd", false)
		if !dec1.Hold {
			b.Fatal("unexpected no-hold")
		}
		dec2 := Evaluate(reg, "cache", "disk", false)
		if dec2.Hold {
			b.Fatal("unexpected hold")
		}
	}
}

func BenchmarkEvaluate(b *testing.B) {
	reg := Normalize(Registry{
		Categories: []Category{
			{
				Name:           "serving",
				Layers:         []string{"tier1", "tier2", "tier3", "tier4"},
				CompletedLayer: "tier2",
				NextLayer:      "tier3",
				Witness:        "witness-serving",
			},
		},
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(reg, "serving", "tier2", false)
	}
}

func BenchmarkNormalize(b *testing.B) {
	raw := Registry{
		Categories: []Category{
			{
				Name:           "  ROUTING_TIER  ",
				Layers:         []string{"tier_1", "tier_2", "tier_3"},
				CompletedLayer: "tier_1",
				NextLayer:      "tier_2",
				Witness:        " wit-1 ",
			},
			{
				Name:           "ADMISSION",
				Layers:         []string{"basic", "strict"},
				CompletedLayer: "basic",
				NextLayer:      "strict",
				Witness:        "wit-2",
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Normalize(raw)
	}
}

package categorybaseline

import (
	"testing"
)

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

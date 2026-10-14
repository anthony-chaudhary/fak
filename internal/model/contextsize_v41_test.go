package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=1ms
func TestContextSizeV41RealizedPrecision(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	for _, hybrid := range []string{"0", "1"} {
		t.Run("hybrid_"+hybrid, func(t *testing.T) {
			t.Setenv("FAK_HYBRID_KV", hybrid)
			want := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
			for _, requested := range []compute.KVPrecision{compute.KVPrecisionF32, compute.KVPrecisionQ8} {
				got := cfg.ContextSizeConfigWithPrecision(requested)
				if !reflect.DeepEqual(got, want) {
					t.Errorf("requested %s changed V4.1's F32 projection: got %#v, want %#v", requested, got, want)
				}
				for _, tokens := range []int{1, 256, 16384} {
					// Preserve the existing conservative uniform tier. This is not
					// a complete bound on V4.1 fixed rings or device-log capacities.
					wantBytes := int64(tokens) * 40 * 512 * 3 * 4
					if bytes := got.KVBytes(tokens); bytes != wantBytes {
						t.Errorf("requested %s tokens=%d KV bytes=%d, want %d", requested, tokens, bytes, wantBytes)
					}
					kvDemands := 0
					for _, demand := range got.PerContextMemoryPlan(tokens) {
						if demand.Class != compute.MemoryKVCache {
							continue
						}
						kvDemands++
						if demand.Bytes != wantBytes || demand.DType != compute.F32.String() {
							t.Errorf("requested %s tokens=%d KV demand=%#v, want %d f32 bytes", requested, tokens, demand, wantBytes)
						}
					}
					if kvDemands != 1 {
						t.Errorf("requested %s tokens=%d KV demands=%d, want one unchanged uniform demand", requested, tokens, kvDemands)
					}
				}
			}

			// Generic dense caches do realize q8. Do not remove their density
			// credit or change any unrelated projection field.
			dense := Config{ModelType: "llama", NumLayers: 2, NumHeads: 2, NumKVHeads: 1,
				HeadDim: 64, HiddenSize: 128, IntermediateSize: 256, VocabSize: 8,
				MaxPositionEmbeddings: 16384}
			f32 := dense.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
			q8 := dense.ContextSizeConfigWithPrecision(compute.KVPrecisionQ8)
			wantQ8 := f32
			wantQ8.KV.Precision = compute.KVPrecisionQ8
			if !reflect.DeepEqual(q8, wantQ8) || q8.KVBytes(256) >= f32.KVBytes(256) {
				t.Fatalf("non-V4.1 q8 projection changed: got %#v, want %#v with a density discount", q8, wantQ8)
			}
		})
	}
}

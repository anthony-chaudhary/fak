package model

import (
	"fmt"
	"testing"
)

var f32DecodeMatRowsSink []float32

// BenchmarkF32DecodeMatRowsSmolLM2Shapes measures the production f32 decode
// row dispatcher at the projection shapes of HuggingFaceTB/SmolLM2-135M-Instruct.
// The shape provenance is docs/benchmarks/IN-KERNEL-MODEL-RESULTS.md: hidden 576,
// 9 query and 3 KV heads of width 64, MLP width 1536, and vocabulary 49152.
//
// The values are deterministic synthetic data, not model weights, so this is a
// shape-level dispatch benchmark rather than an end-to-end model-quality witness.
// TestParallelMatchesSerial independently covers bit identity at all these shapes.
func BenchmarkF32DecodeMatRowsSmolLM2Shapes(b *testing.B) {
	shapes := []struct {
		name    string
		out, in int
	}{
		{name: "q_o_proj", out: 576, in: 576},
		{name: "k_v_proj", out: 192, in: 576},
		{name: "gate_up_proj", out: 1536, in: 576},
		{name: "down_proj", out: 576, in: 1536},
		{name: "lm_head", out: 49152, in: 576},
	}

	for _, shape := range shapes {
		b.Run(fmt.Sprintf("%s_%dx%d", shape.name, shape.out, shape.in), func(b *testing.B) {
			weights := make([]float32, shape.out*shape.in)
			input := make([]float32, shape.in)
			for i := range weights {
				weights[i] = float32((i%251)-125) / 251
			}
			for i := range input {
				input[i] = float32((i%127)-63) / 127
			}

			b.ReportAllocs()
			b.SetBytes(int64(len(weights)) * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f32DecodeMatRowsSink = parMatRows(weights, input, shape.out, shape.in)
			}
		})
	}
}

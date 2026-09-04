package ggufinterop

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func makeBenchmarkFile(tensorCount int, splitCount int, mixed bool) *ggufload.File {
	meta := map[string]ggufload.Value{
		"general.architecture": {Type: ggufload.TypeString, Value: "qwen2"},
	}
	if splitCount > 1 {
		meta["split.count"] = ggufload.Value{Type: ggufload.TypeUint32, Value: uint32(splitCount)}
	}

	types := []ggufload.TensorType{
		ggufload.TensorQ4_K,
		ggufload.TensorQ5_K,
		ggufload.TensorQ6_K,
		ggufload.TensorQ2_K,
		ggufload.TensorQ3_K,
	}
	if mixed {
		types = append(types, ggufload.TensorF32, ggufload.TensorIQ4_NL, ggufload.TensorQ8_0)
	}

	tensors := make([]ggufload.TensorInfo, tensorCount)
	for i := 0; i < tensorCount; i++ {
		tensors[i] = ggufload.TensorInfo{
			Name: fmt.Sprintf("blk.%d.weight", i),
			Type: types[i%len(types)],
			Dims: []uint64{4096, 4096},
		}
	}

	return &ggufload.File{
		Metadata: meta,
		Tensors:  tensors,
	}
}

// BenchmarkGGUFInterop measures mapping throughput for a representative single-file quantized model.
func BenchmarkGGUFInterop(b *testing.B) {
	file := makeBenchmarkFile(128, 1, false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Map(file)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

// BenchmarkGGUFInterop_SplitShard measures mapping throughput for multi-shard GGUF artifacts.
func BenchmarkGGUFInterop_SplitShard(b *testing.B) {
	file := makeBenchmarkFile(128, 4, false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Map(file)
		if res.Outcome != OutcomeDelegate {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

// BenchmarkGGUFInterop_MixedQuant measures mapping throughput when tensors span multiple quant families.
func BenchmarkGGUFInterop_MixedQuant(b *testing.B) {
	file := makeBenchmarkFile(128, 1, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Map(file)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

// BenchmarkGGUFInterop_LargeModel measures mapping throughput over a 512-tensor model topology.
func BenchmarkGGUFInterop_LargeModel(b *testing.B) {
	file := makeBenchmarkFile(512, 1, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Map(file)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

func TestBenchmarkFixtures(t *testing.T) {
	fUniform := makeBenchmarkFile(32, 1, false)
	if res := Map(fUniform); res.Outcome != OutcomeSupported || res.SplitCount != 1 {
		t.Fatalf("expected supported single-split, got %+v", res)
	}

	fSplit := makeBenchmarkFile(32, 4, false)
	if res := Map(fSplit); res.Outcome != OutcomeDelegate || res.SplitCount != 4 {
		t.Fatalf("expected delegate split=4, got %+v", res)
	}

	fMixed := makeBenchmarkFile(32, 1, true)
	if res := Map(fMixed); res.Outcome != OutcomeSupported {
		t.Fatalf("expected supported mixed, got %+v", res)
	}
}

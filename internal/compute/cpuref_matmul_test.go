package compute

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

const (
	cpuRefBenchOut = 1024
	cpuRefBenchIn  = 4096
)

// cpuRefBenchKQuant builds a deterministic raw k-quant weight whose f16 scale fields hold a
// small finite value, so the dequant arithmetic never runs on NaN/Inf.
func cpuRefBenchKQuant(blockBytes int, scaleOffsets ...int) []byte {
	raw := make([]byte, cpuRefBenchOut*(cpuRefBenchIn/q4kSuper)*blockBytes)
	rand.New(rand.NewSource(7)).Read(raw)
	for off := 0; off < len(raw); off += blockBytes {
		for _, so := range scaleOffsets {
			binary.LittleEndian.PutUint16(raw[off+so:], 0x1c00)
		}
	}
	return raw
}

func cpuRefBenchX(n int) Tensor {
	x := make([]float32, n)
	r := rand.New(rand.NewSource(11))
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	return NewF32(&cpuBackend{}, []int{n}, x)
}

func cpuRefBenchWeights() map[string]Tensor {
	be := &cpuBackend{}
	shape := []int{cpuRefBenchOut, cpuRefBenchIn}
	return map[string]Tensor{
		"Q4_K": NewQ4K(be, shape, cpuRefBenchKQuant(q4kSuperBlock, 0, 2)),
		"Q5_K": NewQ5K(be, shape, cpuRefBenchKQuant(176, 0, 2)),
		"Q6_K": NewQ6K(be, shape, cpuRefBenchKQuant(210, 208)),
	}
}

func BenchmarkCPURefMatMulKQuant(b *testing.B) {
	be := &cpuBackend{}
	x := cpuRefBenchX(cpuRefBenchIn)
	ws := cpuRefBenchWeights()
	for _, name := range []string{"Q4_K", "Q5_K", "Q6_K"} {
		w := ws[name]
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				be.MatMul(w, x)
			}
		})
	}
}

func BenchmarkCPURefBatchedMatMulQ4K(b *testing.B) {
	const p = 4
	be := &cpuBackend{}
	w := cpuRefBenchWeights()["Q4_K"]
	x := cpuRefBenchX(p * cpuRefBenchIn)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		be.BatchedMatMul(w, x, p)
	}
}

// fak-test:runtime fast est=1s
func TestCPURefMatMulKQuantReadsWeightInPlace(t *testing.T) {
	be := &cpuBackend{}
	x := cpuRefBenchX(cpuRefBenchIn)
	for name, w := range cpuRefBenchWeights() {
		allocs := testing.AllocsPerRun(5, func() { be.MatMul(w, x) })
		if allocs > 3 {
			t.Errorf("%s MatMul allocs/op = %v, want <= 3 (output row + tensor only)", name, allocs)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCPURefMatMulFeedsAttachedObserver(t *testing.T) {
	var got []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) { got = append(got, e) })
	defer computetrace.SetObserver(nil)
	be := &cpuBackend{}
	be.MatMul(cpuRefBenchWeights()["Q4_K"], cpuRefBenchX(cpuRefBenchIn))
	if len(got) != 1 || got[0].Operation != "matmul" || got[0].Kernel != Q4_K.String()+"_matmul" {
		t.Fatalf("observer events = %+v, want one Q4_K matmul", got)
	}
}

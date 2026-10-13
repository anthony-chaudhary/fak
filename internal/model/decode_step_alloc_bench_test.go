package model

import (
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

func decodeStepAllocModel() *Model {
	return NewSynthetic(Config{
		HiddenSize: 256, NumLayers: 4, NumHeads: 8, NumKVHeads: 4, HeadDim: 32,
		IntermediateSize: 512, VocabSize: 2048, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	})
}

var decodeStepAllocPrompt = []int{1, 5, 2, 7, 3, 9, 11, 4, 8, 6, 10, 12, 13, 14, 15, 16}

const decodeStepAllocWindow = 32

func newDecodeStepAllocSession(quant bool) *Session {
	m := decodeStepAllocModel()
	if quant {
		m.Quantize()
	}
	s := m.NewSession()
	s.Quant = quant
	s.Prefill(decodeStepAllocPrompt)
	return s
}

func runDecodeStepAllocWindow(s *Session, base int) []float32 {
	s.Cache.Truncate(base)
	var logits []float32
	id := 3
	for i := 0; i < decodeStepAllocWindow; i++ {
		logits = s.Step(id)
		id = argmaxF32(logits)
	}
	return logits
}

func benchmarkDecodeStepAllocs(b *testing.B, quant bool) {
	s := newDecodeStepAllocSession(quant)
	base := s.Cache.Len()
	runDecodeStepAllocWindow(s, base)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%decodeStepAllocWindow == 0 {
			s.Cache.Truncate(base)
		}
		s.Step(3 + i%7)
	}
}

func BenchmarkSessionDecodeStepF32(b *testing.B) { benchmarkDecodeStepAllocs(b, false) }
func BenchmarkSessionDecodeStepQ8(b *testing.B)  { benchmarkDecodeStepAllocs(b, true) }

// TestSessionDecodeStepGCReceipt measures GC pause quantiles over a fixed decode run with
// the same runtimeobs reader the gateway exports as go_runtime. Opt-in: it is a probe.
func TestSessionDecodeStepGCReceipt(t *testing.T) {
	if os.Getenv("FAK_DECODE_GC_RECEIPT") == "" {
		t.Skip("set FAK_DECODE_GC_RECEIPT=1 to run the decode GC pause probe")
	}
	for _, quant := range []bool{false, true} {
		s := newDecodeStepAllocSession(quant)
		base := s.Cache.Len()
		runDecodeStepAllocWindow(s, base)
		runtime.GC()
		c := runtimeobs.New()
		before := c.Sample()
		steps := 2000
		if n, err := strconv.Atoi(os.Getenv("FAK_DECODE_GC_RECEIPT")); err == nil && n > 1 {
			steps = n
		}
		for i := 0; i < steps; i++ {
			if i%decodeStepAllocWindow == 0 {
				s.Cache.Truncate(base)
			}
			s.Step(3 + i%7)
		}
		d, err := runtimeobs.Diff(before, c.Sample())
		if err != nil {
			t.Fatal(err)
		}
		p := d.GCPausesSummary
		t.Logf("quant=%v steps=%d gc_cycles=%d heap_allocs_bytes/step=%.0f gc_pauses=%d p50=%.1fus p99=%.1fus max=%.1fus gc_cpu=%.4f",
			quant, steps, d.GCCycles.V, float64(d.HeapAllocsBytes.V)/float64(steps), p.Count.V,
			p.P50.V*1e6, p.P99.V*1e6, p.Max.V*1e6, d.GCCPUFraction.V)
	}
}

// TestSessionDecodeStepBufferReuseParity: a session that has already decoded (warm,
// buffers reused) must produce bit-identical logits to a fresh session decoding the same
// tokens, so per-step scratch reuse never leaks state between steps.
func TestSessionDecodeStepBufferReuseParity(t *testing.T) {
	for _, quant := range []bool{false, true} {
		warm := newDecodeStepAllocSession(quant)
		base := warm.Cache.Len()
		runDecodeStepAllocWindow(warm, base)
		got := append([]float32(nil), runDecodeStepAllocWindow(warm, base)...)

		fresh := newDecodeStepAllocSession(quant)
		want := runDecodeStepAllocWindow(fresh, fresh.Cache.Len())
		if len(got) != len(want) {
			t.Fatalf("quant=%v: logits len %d want %d", quant, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("quant=%v: logit %d differs warm=%v fresh=%v", quant, i, got[i], want[i])
			}
		}
	}
}

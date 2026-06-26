package agent

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// overlapBackend wraps a real compute.Backend and FAILS the test if two device ops ever run
// on it at the same time. It models the single-stream device invariant the CUDA backend has
// by construction (one g_stream, one cuBLAS handle, a shared device free-list): two concurrent
// forwards interleaving their op sequences on that shared state is exactly what faulted the
// live L4 serve with a sticky cuda_kernels.cu illegal-memory-access. The detector is a flat
// in-flight counter: the reference ops below are leaf calls (no op re-enters another), so any
// time entry sees a nonzero count, two forwards driven by different goroutines are overlapping.
type overlapBackend struct {
	compute.Backend
	inflight atomic.Int64
	overlaps atomic.Int64
}

func (b *overlapBackend) op() func() {
	if b.inflight.Add(1) > 1 {
		b.overlaps.Add(1)
	}
	return func() { b.inflight.Add(-1) }
}

// The forward path hits these ops every token; guarding them witnesses an interleave. Each
// delegates to the wrapped reference backend so decode still produces real tokens (the test
// asserts BOTH no-overlap and that decode completed).
func (b *overlapBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	defer b.op()()
	return b.Backend.MatMul(w, x)
}

func (b *overlapBackend) BatchedMatMul(w, X compute.Tensor, P int) compute.Tensor {
	defer b.op()()
	return b.Backend.BatchedMatMul(w, X, P)
}

func (b *overlapBackend) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	defer b.op()()
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}

func (b *overlapBackend) Argmax(logits compute.Tensor) int {
	defer b.op()()
	return b.Backend.Argmax(logits)
}

func tinyConcurrencyConfig() model.Config {
	return model.Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         320, // covers the 256-byte vocab + ChatML specials the probe tokenizer emits
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// TestInKernelConcurrentDeviceCompleteSerializes is the paired honesty test for the devMu
// fix: with a backend wired, Complete calls driven concurrently (the way the gateway drives
// the live serve) must NEVER overlap on the single-stream device. Without devMu the forwards
// interleave their per-token ops on the shared backend and the overlap detector fires, which
// is the live crash, made deterministic and GPU-free. To make a missing lock fail reliably
// rather than flakily, each forward runs several decode steps (maxNew) so the interleave
// window is wide, and the goroutines are released together by a start barrier.
func TestInKernelConcurrentDeviceCompleteSerializes(t *testing.T) {
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	be := &overlapBackend{Backend: base}

	m := model.NewSynthetic(tinyConcurrencyConfig())
	tok := loadProbeTok(t)

	p := NewInKernelPlanner(m, tok, "tiny-gpu", false, be)
	p.maxNew = 8

	msgs := []Message{{Role: "user", Content: "hello there, decode a few tokens please"}}

	const N = 4
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make([]error, N)
	gens := make([]int, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start // release all goroutines at once to maximize the overlap window
			comp, err := p.Complete(context.Background(), msgs, nil)
			errs[i] = err
			if comp != nil {
				gens[i] = comp.Usage.CompletionTokens
			}
		}(i)
	}
	close(start)
	wg.Wait()

	if got := be.overlaps.Load(); got != 0 {
		t.Fatalf("device ops OVERLAPPED %d times: the single-stream backend was driven concurrently (the live L4 crash); devMu must serialize Complete on the backend path", got)
	}
	for i := 0; i < N; i++ {
		if errs[i] != nil {
			t.Fatalf("Complete[%d] errored: %v", i, errs[i])
		}
		if gens[i] == 0 {
			t.Fatalf("Complete[%d] generated 0 tokens: decode did not run", i)
		}
	}
}

// TestInKernelCPUPathUnaffectedByDevMu confirms the fix is a no-op on the CPU path: a
// nil-backend planner still completes (the devMu hold only engages when backend != nil),
// so the radix-reuse / CPU-session path is byte-for-byte unchanged.
func TestInKernelCPUPathUnaffectedByDevMu(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	tok := loadProbeTok(t)
	p := NewInKernelPlanner(m, tok, "tiny-cpu", false, nil) // nil backend -> CPU path, devMu never taken
	p.quant = false                                         // exercise the proven f32 reuse path (no Q8 cache on a synthetic)
	p.maxNew = 4
	comp, err := p.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CPU Complete errored: %v", err)
	}
	if comp == nil || comp.Usage.CompletionTokens == 0 {
		t.Fatal("CPU Complete generated nothing")
	}
}

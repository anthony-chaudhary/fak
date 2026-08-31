package ggufload

// gguf_parload.go — the bounded-worker-pool parallel GGUF quant-on-load pipeline.
//
// WHY. The GGUF quant-on-load loop is CPU-bound in per-tensor dequant (Q4_K/Q6_K -> f32)
// plus the f32->Q8 re-quant, and it was historically SERIAL: `for _, info := range
// s.File.Tensors { ... }` ran the whole load on ONE core. On the GLM-5.2 serve box
// (8x A100, 256 cores, 2 TB RAM) that meant the 466 GB UD-Q4_K_M load streamed at ~0.12
// GB/s — ~100 min — while 255 cores sat idle (a plain `cp` of the same shards runs at
// ~2.8 GB/s, so disk was never the limit; the single-core dequant was). This pipeline
// fans the expensive, pure per-tensor work (read + dequant + normalize + expert split)
// across a worker pool and applies the builder mutations SERIALLY in original tensor
// order — so the built model is byte-identical to the serial loader (the builder's packed
// f32 blob grows in insertion order), only faster.
//
// BYTE-IDENTITY. The only shared mutable state — the model.QuantBuilder, the glm_moe_dsa
// KV-b merge buffer, and the LoadProfiler — is touched by the SINGLE collector goroutine,
// in the SAME order the serial loop touched it. Workers do only pure work: TensorBytes
// copies into a fresh buffer (ReadAt is safe for concurrent use), dequantF32 allocates
// fresh, and the split/normalize helpers are pure over their inputs and the read-only
// Config. So the parallel and serial loads produce the same Model tensor-for-tensor
// (pinned by TestParallelQuantLoadMatchesSerial).
//
// MEMORY. The f32 round-trip of one batched expert blob can be several GB transient; the
// collector releases a window slot only after it has APPLIED a tensor, so at most
// `workers` tensors are dequanted-but-not-applied at once — peak transient ~ workers x
// max-tensor-f32. loadWorkers caps the default so that product stays well inside host RAM;
// FAK_GGUF_LOAD_WORKERS tunes it.

import (
	"context"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// loadWorkerCap bounds the DEFAULT per-tensor load concurrency. The dequant is largely
// memory-bandwidth bound, so a modest pool already saturates the win, and a small cap keeps
// the peak transient (window x one expert blob's multi-GB f32) comfortably inside host RAM
// on the big-model serve box. FAK_GGUF_LOAD_WORKERS overrides this in either direction.
const loadWorkerCap = 16

// activeParallelLoads marks worker-pool load scopes. Inside one of these scopes the
// tensor workers already consume the shared CPU budget, so nested block dequant stays
// serial rather than multiplying runnable work by GOMAXPROCS. Standalone dequant and
// serial loads retain the existing block-level parallel path.
var activeParallelLoads atomic.Int32

// loadWorkers returns the per-tensor load concurrency: min(GOMAXPROCS, loadWorkerCap) by
// default, or the FAK_GGUF_LOAD_WORKERS override (>=1). It never returns < 1.
func loadWorkers() int {
	if v := strings.TrimSpace(os.Getenv("FAK_GGUF_LOAD_WORKERS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			return n
		}
	}
	n := runtime.GOMAXPROCS(0)
	if n > loadWorkerCap {
		n = loadWorkerCap
	}
	if n < 1 {
		n = 1
	}
	return n
}

// pendingTensor is one builder mutation a load worker produced from a GGUF tensor. The
// collector applies these in original tensor order; the order within one GGUF tensor's slice
// is the order the serial loader emitted them (e.g. expert 0..E-1), so insertion order — and
// thus the built model — is byte-identical to the serial path.
type pendingTensor struct {
	resident     bool       // true -> AddResident*(raw) by residentType; false -> AddF32Tensor(f32)
	residentType TensorType // which resident raw-quant store, when resident
	lazyQ4K      bool
	sourceInfo   TensorInfo
	lazyReader   io.ReaderAt
	isKVBHalf    bool // true -> bufferGLMKVBHalf(layer, half, f32); merge applied on the 2nd half
	name         string
	shape        []int
	raw          []byte    // resident raw super-block bytes (resident==true)
	f32          []float32 // dequantized + normalized values (resident==false, or a KV-b half)
	layer        int
	half         string
}

// residentExpertBlockGeometry returns the GGUF block geometry for an expert tensor type that can
// be held RESIDENT (raw bytes, dequant fused in the GEMV) for the CPU-offloaded GLM MoE experts.
// ok=false keeps that expert blob on the f32 dequant→Q8 fallback.
func residentExpertBlockGeometry(t TensorType) (blockWeights, blockBytes int, ok bool) {
	switch t {
	case TensorQ8_0:
		return qk8_0, blockQ8_0Bytes, true
	case TensorQ4_0:
		// Legacy ggml 32-weight blocks (f16 scale + 16 nibble bytes, 18 B = 0.5625 B/weight;
		// #5497). Resident target is the raw expert-quant store (AddResidentQ4_0 →
		// kQuantMatRows). Without this row a native-q4_0 checkpoint — chosen precisely because
		// it is the small artifact — takes the Q4→f32→Q8 round trip and ends up resident at
		// Q8_0's 1.0625 B/weight, roughly DOUBLE its own on-disk size, having held both copies
		// live at the round-trip's peak.
		return qk4, blockQ4_0Bytes, true
	case TensorQ4_K:
		return qkK, blockQ4KBytes, true
	case TensorQ5_K:
		return qkK, blockQ5KBytes, true
	case TensorQ6_K:
		return qkK, blockQ6KBytes, true
	case TensorIQ3_XXS:
		return qkK, blockIQ3XXSBytes, true
	case TensorIQ2_XXS:
		return qkK, blockIQ2XXSBytes, true
	case TensorIQ2_XS:
		return qkK, blockIQ2XSBytes, true
	case TensorIQ1_S:
		return qkK, blockIQ1SBytes, true
	case TensorIQ2_S:
		return qkK, blockIQ2SBytes, true
	case TensorIQ1_M:
		return qkK, blockIQ1MBytes, true
	case TensorIQ4_XS:
		return qkK, blockIQ4XSBytes, true
	case TensorQ2_0:
		// Ternary group-128 blocks (f16 scale + 128 2-bit codes, 34 B; #4868 T1). Resident
		// target is the model's ternary store (AddResidentQ2 → q2MatRows, #4870): the raw
		// GGUF bytes feed the CPU GEMV directly, no re-quantize and no ~15× f32 round trip.
		return 128, blockQ2_0Bytes, true
	}
	return 0, 0, false
}

// tensorWork is one GGUF tensor's parallel-load result: the progress byte count, the builder
// mutations to apply, the per-quant-type accounting for the load-path breakdown, or an error.
type tensorWork struct {
	tickBytes int64
	pending   []pendingTensor
	err       error

	// Load-path accounting (the per-quant-type visibility, recorded once per GGUF tensor by
	// the serial collector). acctType == "" means "do not tally" (skipped tensors).
	acctType     string // GGUF quant type, e.g. "Q4_K"/"Q6_K"
	acctExpert   bool   // came from a batched/shared MoE expert blob (the 417 GB bulk)
	acctResident bool   // true = took the raw-resident fast path; false = f32 round-trip
	acctBytes    int64  // on-disk payload bytes for this GGUF tensor
	acctTensors  int    // number of model tensors produced (E for an expert blob, else 1)
}

// parallelQuantLoad runs computeFn over every GGUF tensor on a bounded worker pool and calls
// applyFn on each result IN ORIGINAL TENSOR ORDER from a single collector goroutine. computeFn
// must be pure / safe for concurrent use; applyFn owns all shared mutable state (builder,
// merge buffer, profiler) and is never called concurrently. The first error from either
// stops application (remaining results are still drained so no worker blocks) and is returned.
func (s *WeightSource) parallelQuantLoad(computeFn func(TensorInfo) tensorWork, applyFn func(tensorWork) error) error {
	return s.parallelQuantLoadContext(context.Background(), computeFn, applyFn)
}

// parallelQuantLoadContext is parallelQuantLoad with cooperative cancellation. Cancellation
// stops new tensor admission, then joins the feeder and every worker before returning. Work
// already admitted is drained in tensor order so the earliest tensor/apply error remains the
// deterministic cause rather than being replaced by the cancellation used to stop the pool.
func (s *WeightSource) parallelQuantLoadContext(ctx context.Context, computeFn func(TensorInfo) tensorWork, applyFn func(tensorWork) error) error {
	tensors := s.File.Tensors
	n := len(tensors)
	if n == 0 {
		return ctx.Err()
	}
	workers := loadWorkers()
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := range tensors {
			if err := ctx.Err(); err != nil {
				return err
			}
			w := computeFn(tensors[i])
			if w.err != nil {
				return w.err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := applyFn(w); err != nil {
				return err
			}
		}
		return nil
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	activeParallelLoads.Add(1)
	defer activeParallelLoads.Add(-1)

	results := make([]tensorWork, n)
	done := make([]chan struct{}, n)
	for i := range done {
		done[i] = make(chan struct{})
	}
	sem := make(chan struct{}, workers)
	jobs := make(chan int)
	var admitted atomic.Int32

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				// Jobs accepted by the feeder own a slot and must run to completion. The
				// cancellation signal stops later admission; computeFn may also observe it
				// cooperatively, but the pool never replaces an admitted tensor's real error.
				results[i] = computeFn(tensors[i])
				if results[i].err != nil {
					cancel()
				}
				close(done[i])
			}
		}()
	}
	feederDone := make(chan struct{})
	go func() {
		defer close(feederDone)
		defer close(jobs)
		for i := 0; i < n; i++ {
			if runCtx.Err() != nil {
				return
			}
			select {
			case sem <- struct{}{}:
			case <-runCtx.Done():
				return
			}
			if runCtx.Err() != nil {
				<-sem
				return
			}
			select {
			case jobs <- i:
				admitted.Add(1)
			case <-runCtx.Done():
				<-sem
				return
			}
		}
	}()

	var firstErr error
	for i := 0; i < n; i++ {
		select {
		case <-done[i]:
		case <-feederDone:
			if i >= int(admitted.Load()) {
				wg.Wait()
				if firstErr != nil {
					return firstErr
				}
				return ctx.Err()
			}
			<-done[i]
		}
		if firstErr == nil {
			if results[i].err != nil {
				firstErr = results[i].err
				cancel()
			} else if err := applyFn(results[i]); err != nil {
				firstErr = err
				cancel()
			}
		}
		results[i] = tensorWork{}
		<-sem
	}
	<-feederDone
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}

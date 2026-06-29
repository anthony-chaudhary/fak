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
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// loadWorkerCap bounds the DEFAULT per-tensor load concurrency. The dequant is largely
// memory-bandwidth bound, so a modest pool already saturates the win, and a small cap keeps
// the peak transient (window x one expert blob's multi-GB f32) comfortably inside host RAM
// on the big-model serve box. FAK_GGUF_LOAD_WORKERS overrides this in either direction.
const loadWorkerCap = 16

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
	isKVBHalf    bool       // true -> bufferGLMKVBHalf(layer, half, f32); merge applied on the 2nd half
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
	case TensorQ4_K:
		return qkK, blockQ4KBytes, true
	case TensorQ5_K:
		return qkK, blockQ5KBytes, true
	case TensorQ6_K:
		return qkK, blockQ6KBytes, true
	case TensorIQ3_XXS:
		return qkK, blockIQ3XXSBytes, true
	case TensorIQ4_XS:
		return qkK, blockIQ4XSBytes, true
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
	tensors := s.File.Tensors
	n := len(tensors)
	if n == 0 {
		return nil
	}
	workers := loadWorkers()
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		// Serial fallback: identical order, no goroutines (cheap small loads + a clean
		// -race baseline). Byte-identical to the pool path by construction.
		for i := range tensors {
			w := computeFn(tensors[i])
			if w.err != nil {
				return w.err
			}
			if err := applyFn(w); err != nil {
				return err
			}
		}
		return nil
	}

	results := make([]tensorWork, n)
	done := make([]chan struct{}, n)
	for i := range done {
		done[i] = make(chan struct{})
	}
	// sem is the look-ahead window: the feeder acquires a slot before queuing a tensor and
	// the collector releases one after APPLYING it, so at most `workers` tensors are in
	// flight (dequanted but not yet applied) — the peak-transient bound.
	sem := make(chan struct{}, workers)
	jobs := make(chan int)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = computeFn(tensors[i]) // write happens-before close(done[i])
				close(done[i])
			}
		}()
	}
	go func() {
		for i := 0; i < n; i++ {
			sem <- struct{}{} // block until the collector frees a window slot
			jobs <- i
		}
		close(jobs)
	}()

	var applyErr error
	for i := 0; i < n; i++ {
		<-done[i] // receive synchronizes-with the worker's write to results[i]
		if applyErr == nil {
			if results[i].err != nil {
				applyErr = results[i].err
			} else if err := applyFn(results[i]); err != nil {
				applyErr = err
			}
		}
		results[i] = tensorWork{} // drop the reference so a multi-GB f32 can be GC'd now
		<-sem                     // free a window slot for the feeder
	}
	wg.Wait()
	return applyErr
}

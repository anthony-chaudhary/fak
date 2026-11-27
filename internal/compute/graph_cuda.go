//go:build cuda && cgo

// graph_cuda.go — the runtime enable seam for the CUDA-graph decode replay path
// (#483). The graph gate `graphEnabled` is read once at init() from FAK_CUDA_GRAPH,
// but a host (e.g. `fak serve --cuda-graph`) needs to flip it from a parsed flag
// AFTER init() has run. graphEnabled is consulted per token at GraphBegin time (not
// cached), so a post-init flip cleanly activates the fully-wired capture/replay path
// — and updates Caps().GraphCompile so a downstream consumer sees consistent consent.
//
// This lives in a separate cuda-tagged file (not cuda.go) so the enable seam is a
// one-line, low-blast-radius edit; it shares the `compute` package with cuda.go and
// so reads its private graphEnabled var directly.
package compute

// EnableCUDAGraph turns on the CUDA-graph decode replay path at runtime. No-op-safe
// to call when no CUDA device was found at init (graphEnabled simply gates a path the
// cpu-ref backend never advertises). The non-cuda build provides an empty twin so a
// host can call this unconditionally.
func EnableCUDAGraph() {
	joinDeferredBackends()
	graphEnabled = true
}

// CUDAGraphReplays reports how many captured decode graphs have replayed and fenced cleanly.
func CUDAGraphReplays() uint64 { return cudaGraphReplays.Load() }

// SetCUDAGraphKVCapacity raises the fixed device-KV capacity (in positions) that each
// graph-capturing session preallocates, so a real prompt never grows the cache mid-capture
// (a cudaMalloc during capture is illegal — #932). A host wires it to --context-budget-tokens
// alongside EnableCUDAGraph or FAK_CUDA_GRAPH=1. It only RAISES the default (1024): a smaller request is ignored,
// so the decode-bench default capacity is never weakened. Must be called before the first
// NewKV (the serve flips it at startup, before any session is created). Note the prealloc is
// 3 buffers (K, Kraw, V) × numKVHeads × headDim × positions × 4B per layer — a large budget
// is real VRAM, so size it to the context you actually serve, not the architectural maximum.
func SetCUDAGraphKVCapacity(positions int) {
	joinDeferredBackends()
	if positions > cudaKVMaxPos {
		cudaKVMaxPos = positions
	}
}

// CUDAGraphEnabled reports whether CUDA-graph decode replay is currently opted in.
func CUDAGraphEnabled() bool {
	joinDeferredBackends()
	return graphEnabled
}

// CUDAGraphKVCapacity reports the configured fixed device-KV capacity (in positions)
// that graph-capturing sessions preallocate when CUDA-graph decode is opted in.
func CUDAGraphKVCapacity() int {
	joinDeferredBackends()
	return cudaKVMaxPos
}

// CUDAGraphKVPreallocDemand returns the additional device-memory demand required to
// preallocate fixed graph-capacity KV storage for a capturing session, minus any device
// attention KV rows already charged in base plan (alreadyChargedKVBytes). When graphs
// are disabled, cfg.GraphCapture is false, positions <= 0, or already-charged KV already
// covers the prealloc, it returns nil so graph decode stays strictly opt-in.
func CUDAGraphKVPreallocDemand(graphOn bool, cfg KVConfig, positions int, alreadyChargedKVBytes int64) MemoryPlan {
	prealloc := GraphKVPreallocFor(graphOn, cfg, positions, FreeUnknown)
	if !prealloc.Preallocate || prealloc.Bytes <= 0 {
		return nil
	}
	if alreadyChargedKVBytes < 0 {
		alreadyChargedKVBytes = 0
	}
	additional := prealloc.Bytes - alreadyChargedKVBytes
	if additional <= 0 {
		return nil
	}
	return MemoryPlan{{
		Class:  MemoryKVCache,
		Bytes:  additional,
		Detail: "cuda-graph-kv-prealloc",
		Scope:  MemoryScopeDevice,
		DType:  cfg.Precision.StorageLabel(),
	}}
}

func graphKVAlreadyChargedBytes(plan MemoryPlan) int64 {
	var total int64
	for _, d := range plan {
		if d.Class != MemoryKVCache || d.Bytes <= 0 {
			continue
		}
		if d.Scope != "" && d.Scope != MemoryScopeDevice {
			continue
		}
		// Only deduct attention KV row allocations that fixed graph prealloc replaces,
		// not independent recurrent/conv caches (e.g. GDN state) or host-scoped caches.
		switch d.Detail {
		case "", "kv", "hal-kv-store", "compressed-kv-rows", "cuda-graph-kv-prealloc":
			total = saturatingAddInt64(total, d.Bytes)
		}
	}
	return total
}

func withCUDAGraphKVPreallocFor(base MemoryPlan, graphOn bool, cfg KVConfig, positions int) MemoryPlan {
	extra := CUDAGraphKVPreallocDemand(graphOn, cfg, positions, graphKVAlreadyChargedBytes(base))
	if len(extra) == 0 {
		return base
	}
	out := make(MemoryPlan, 0, len(base)+len(extra))
	out = append(out, base...)
	out = append(out, extra...)
	return out
}

// WithCUDAGraphKVPrealloc returns a MemoryPlan with any unmet fixed CUDA-graph KV
// preallocation charged to device capacity when CUDA-graph decode is opted in and cfg
// is capture-eligible.
func WithCUDAGraphKVPrealloc(base MemoryPlan, cfg KVConfig) MemoryPlan {
	joinDeferredBackends()
	return withCUDAGraphKVPreallocFor(base, graphEnabled, cfg, cudaKVMaxPos)
}

// RefuseCUDAGraphFitIfTooBig evaluates base augmented with fixed CUDA-graph KV
// preallocation against b, failing pre-launch with a typed *FitError (FitTooBig) when
// graph KV prealloc would exceed device capacity while keeping unknown backends open.
func RefuseCUDAGraphFitIfTooBig(b Backend, base MemoryPlan, cfg KVConfig, headroom float64) error {
	return RefuseMemoryPlanIfTooBig(b, WithCUDAGraphKVPrealloc(base, cfg), headroom)
}

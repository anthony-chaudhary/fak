//go:build cuda

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
func EnableCUDAGraph() { graphEnabled = true }

// SetCUDAGraphKVCapacity raises the fixed device-KV capacity (in positions) that each
// graph-capturing session preallocates, so a real prompt never grows the cache mid-capture
// (a cudaMalloc during capture is illegal — #932). A host wires it to --context-budget-tokens
// alongside EnableCUDAGraph. It only RAISES the default (1024): a smaller request is ignored,
// so the decode-bench default capacity is never weakened. Must be called before the first
// NewKV (the serve flips it at startup, before any session is created). Note the prealloc is
// 3 buffers (K, Kraw, V) × numKVHeads × headDim × positions × 4B per layer — a large budget
// is real VRAM, so size it to the context you actually serve, not the architectural maximum.
func SetCUDAGraphKVCapacity(positions int) {
	if positions > cudaKVMaxPos {
		cudaKVMaxPos = positions
	}
}

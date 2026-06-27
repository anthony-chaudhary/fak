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

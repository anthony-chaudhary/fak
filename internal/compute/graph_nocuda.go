//go:build !cuda

// graph_nocuda.go — the non-cuda twin of EnableCUDAGraph (graph_cuda.go). The default
// `go build ./cmd/fak` excludes the CUDA backend entirely, so there is no graph path to
// enable; this empty function lets a host (`fak serve --cuda-graph`) call the enable seam
// unconditionally without a build-tag branch at the call site. The flag is simply inert
// on a CPU-only binary — exactly as a graph-replay lever for a device decode should be.
package compute

// EnableCUDAGraph is a no-op in the non-cuda build (no device, no graph path).
func EnableCUDAGraph() {}

// SetCUDAGraphKVCapacity is a no-op in the non-cuda build (no device-KV to preallocate). It
// exists so a host can wire --context-budget-tokens to the graph-KV capacity unconditionally,
// without a build-tag branch at the call site. The cuda twin (graph_cuda.go) does the work.
func SetCUDAGraphKVCapacity(int) {}

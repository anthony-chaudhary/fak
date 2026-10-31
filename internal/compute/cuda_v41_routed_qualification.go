//go:build cuda && cgo && (!fak_cuda_v41_deferred_full_qualification || !fak_cuda_v41_routed_mmid)

package compute

// V41RoutedLayerQualified stays false until the complete deferred-status model
// route has device numerical, ownership, cancellation and serving qualification.
// Building candidate kernels or adding interface methods must not enable it.
// Private witnesses call candidate primitives directly, never this serving gate.
func (*cudaBackend) V41RoutedLayerQualified() bool { return false }

//go:build vulkan && (windows || linux) && cgo

package compute

func (v *vulkanBackend) v41ReadIndexerOwned(original Backend, tensor Tensor) ([]float32, bool) {
	// Exact receiver identity excludes embedded wrappers and preserves their Read.
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v {
		return original.Read(tensor), false
	}
	// Invoke canonical Read directly. Its device branch always makes a fresh Go
	// slice; its hostBuf branch aliases storage and must retain the outer copy.
	values := v.Read(tensor)
	_, device := tensor.buf.(*vulkanBuf)
	return values, device && tensor.be == v
}

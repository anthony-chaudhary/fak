//go:build vulkan && (windows || linux) && cgo

package compute

// SupportsQ2K advertises native packed Q2_K matrix multiplication.
func (v *vulkanBackend) SupportsQ2K() bool { return true }

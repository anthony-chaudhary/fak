//go:build cuda && cgo

package compute

// This declaration describes the existing eager packed-input implementation.
// It does not enable loader residency, advertise tiled output, bypass runtime
// preflight, qualify H200, or change Qwen35GDNCaptureLegal's F32/Q8 contract.
func (*cudaBackend) Qwen35GDNPackedInputCapability() Qwen35GDNPackedInputCapability {
	return Qwen35GDNPackedInputCapability{
		Contract: Qwen35GDNPackedInputContractV1,
		Q4K:      true,
		Q6K:      true,
	}
}

var _ qwen35GDNPackedImplementation = (*cudaBackend)(nil)
var _ Qwen35GDNPackedInputBackend = (*cudaBackend)(nil)

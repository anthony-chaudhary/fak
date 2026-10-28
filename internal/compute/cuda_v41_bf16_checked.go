//go:build cuda && cgo && fak_cuda_v41_bf16_publish

package compute

var _ V41BF16CheckedBackend = (*cudaBackend)(nil)

// V41BF16CheckedAvailable binds the final-norm consumer to the existing checked
// CUDA publication implementation. It retains that implementation's explicit
// build fence; ordinary CUDA builds do not acquire this capability.
func (c *cudaBackend) V41BF16CheckedAvailable() bool { return c != nil }

// V41BF16Checked admits only the two final-norm sites, then delegates to the
// existing finite/RNE publication. That operation owns the CUDA lock, device-only
// output, checked status readback and failure retirement. Do not take cudaMu here
// or translate a selected failure into an unavailable/host-replay result.
func (c *cudaBackend) V41BF16Checked(input Tensor, site uint32) (Tensor, error) {
	if _, err := validateV41BF16Checked(input, site); err != nil {
		return Tensor{}, &CUDAOpError{Op: "V41BF16Checked", Site: "v41-final-norm-bf16-geometry", Err: err}
	}
	return c.V41BF16Publish(input, site)
}

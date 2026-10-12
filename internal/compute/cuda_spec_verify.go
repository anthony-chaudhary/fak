//go:build cuda && cgo

package compute

// validateSpecVerifyOperandsLocked checks the buffers the verification kernel will
// consume. The caller holds cudaMu and has validated tensor pointers and geometry.
func (c *cudaBackend) validateSpecVerifyOperandsLocked(q, k, v, out *Tensor, expectedQ int) error {
	// Validate every pointer before allocating an output or crossing the C ABI.
	// Ready is deliberately not required: queued producers share g_stream's FIFO.
	validate := func(name string, t Tensor) error {
		refuse := func(reason string) error {
			return &CUDAOpError{Op: "SpecVerifyAttention", Site: "spec-verify-preflight", Msg: name + " " + reason}
		}
		if t.Backend() != c {
			return refuse("must be owned by the executing CUDA backend; upload host tensors before verification")
		}
		if t.Dtype != F32 || t.Layout != RowMajor {
			return refuse("must be row-major f32")
		}
		buf, ok := t.buf.(*cudaBuf)
		if !ok || buf == nil || buf.ptr == nil || buf.device != 0 {
			return refuse("must have a live default-device CUDA allocation")
		}
		if err := buf.invalidStateError(name); err != nil {
			return &CUDAOpError{Op: "SpecVerifyAttention", Site: "spec-verify-preflight", Msg: name + " has invalid CUDA state", Err: err}
		}
		if t.Numel() <= 0 || buf.n/F32.Bytes() < t.Numel() {
			return refuse("CUDA allocation is too small for its f32 elements")
		}
		return nil
	}
	for _, operand := range []struct {
		name string
		t    *Tensor
	}{{"q", q}, {"k", k}, {"v", v}} {
		if err := validate(operand.name, *operand.t); err != nil {
			return err
		}
	}
	if out.buf != nil && out.Numel() == expectedQ {
		if err := validate("out", *out); err != nil {
			return err
		}
	}

	return nil
}

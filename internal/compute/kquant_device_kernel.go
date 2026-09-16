package compute

// KQuantDeviceKernel is an OPTIONAL backend capability: it reports whether this backend
// has a resident device MatMul kernel for a quantized weight dtype. It exists because a
// MODEL-level check (e.g. "does this k-quant kind have a HAL descriptor") does not prove
// the ACTUAL Backend can run that dtype: a backend can advertise a resident device seam
// and still panic on a dtype its MatMul switch has no case for.
//
// It is fail-closed by construction. A backend that does not implement it is treated as
// supporting NO quantized device weight beyond what its own MatMul already proves;
// callers that must not panic probe explicitly via BackendSupportsDeviceWeightDtype.
// Absent implementation => all false.
//
// Implementations MUST match the backend's MatMul dtype switch exactly: returning true
// for a dtype with no case would hand the caller a panic instead of a clean decline.
type KQuantDeviceKernel interface {
	// SupportsDeviceWeightDtype reports whether this backend's resident device MatMul
	// has a kernel for a weight of dtype dt.
	SupportsDeviceWeightDtype(dt Dtype) bool
}

// kquantDeviceKernel type-asserts be for the optional device capability, returning nil when
// the backend carries none. It mirrors routedExpertDeviceKernel so the fail-closed rule
// lives in one place.
func kquantDeviceKernel(be Backend) KQuantDeviceKernel {
	if be == nil {
		return nil
	}
	k, ok := be.(KQuantDeviceKernel)
	if !ok {
		return nil
	}
	return k
}

// BackendSupportsDeviceWeightDtype reports whether be has a resident device MatMul kernel
// for a weight of dtype dt. Fail-closed: a backend that does not advertise the
// KQuantDeviceKernel capability returns false (never a panic, never an assumed yes).
func BackendSupportsDeviceWeightDtype(be Backend, dt Dtype) bool {
	k := kquantDeviceKernel(be)
	if k == nil {
		return false
	}
	return k.SupportsDeviceWeightDtype(dt)
}

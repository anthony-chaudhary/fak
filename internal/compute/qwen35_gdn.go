package compute

import "fmt"

// Qwen35GDNCUDAPath is the stable whole-operation identity shared with the
// model.Qwen35GDNBackend contract. It is duplicated at the cycle-free compute
// boundary deliberately: compute cannot import model, while a structural
// implementation must still return the exact production path.
const Qwen35GDNCUDAPath = "cuda/qwen35-gdn-ssm-decode-v1"

// Qwen35GDNParityCosineMin is the device/reference acceptance floor for the
// deterministic whole-operation fixture. The value records the gate; only a
// non-skipped real-device test run witnesses that the implementation clears it.
const Qwen35GDNParityCosineMin = 0.999

// Qwen35GDNGeometryError is a fail-closed refusal raised before any GDN kernel
// is launched. Operand is either a tensor name or "geometry" for a relation
// between scalar dimensions; Want describes the accepted shape/relation.
type Qwen35GDNGeometryError struct {
	Operand string
	Got     []int
	Want    string
	Reason  string
}

func (e *Qwen35GDNGeometryError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN geometry error"
	}
	if e.Reason != "" {
		return fmt.Sprintf("compute: cuda Qwen3.5 GDN geometry refused for %s: %s", e.Operand, e.Reason)
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN geometry refused for %s: shape %v, want %s", e.Operand, e.Got, e.Want)
}

// Qwen35GDNResidencyError refuses a tensor that is not an F32 row-major buffer
// resident on the CUDA backend executing the operation. In particular, the
// implementation never obtains a HostBuffer and never falls back to CPU math.
type Qwen35GDNResidencyError struct {
	Operand string
	Reason  string
}

func (e *Qwen35GDNResidencyError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN residency error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN residency refused for %s: %s", e.Operand, e.Reason)
}

// Qwen35GDNKernelError is the typed fail-closed return for an ABI or CUDA
// launch failure inside the whole operation. Stage names the fused stage; Code
// preserves the flat C ABI status for device-side diagnosis.
type Qwen35GDNKernelError struct {
	Stage string
	Code  int
}

func (e *Qwen35GDNKernelError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN kernel error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN kernel failed closed at %s (code %d); no CPU fallback", e.Stage, e.Code)
}

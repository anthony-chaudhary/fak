package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen35GDNCUDAPath is the production path identity reserved for a Qwen3.5/3.6
// Gated-DeltaNet/SSM token mixer implemented by the CUDA compute backend. Merely
// returning this string is not readiness: NewBackendSessionChecked continues to
// refuse the path until the model HAL actually invokes Qwen35GDNDecode.
const Qwen35GDNCUDAPath = "cuda/qwen35-gdn-ssm-decode-v1"

// Qwen35GDNParityCosineMin is the explicit acceptance floor for the future
// deterministic CUDA fixture against the existing f32 CPU/reference semantics.
// It records a requirement, not a witnessed CUDA pass.
const Qwen35GDNParityCosineMin = 0.999

// Qwen35GDNBackend is the intended whole-operation CUDA/HAL seam for the hybrid
// token mixer. No production backend implements it yet; the contract exists so a
// future CUDA leaf has a concrete operation to implement and parity-test rather
// than advertising a marker while falling through to generic QKV or host code.
// The signature uses only compute-owned and built-in types, so a compute backend
// can implement it structurally without importing model and creating a cycle.
type Qwen35GDNBackend interface {
	Qwen35GDNPath() string
	Qwen35GDNDecode(
		normalizedInput,
		inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj,
		convState, recurrentState compute.Tensor,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
		rmsNormEpsilon float32,
	) (output, nextConvState, nextRecurrentState compute.Tensor, err error)
}

// UnsupportedBackendForwardError is the fail-closed verdict returned when a
// recognized model forward cannot execute wholly on the selected compute backend.
// For #4714 it prevents the Qwen3.6 GDN/SSM hybrid from entering tokenHALOutput's
// standard Q/K/V loop or silently using the legacy CPU recurrent implementation.
type UnsupportedBackendForwardError struct {
	Backend         string
	Forward         ForwardPathKind
	IntendedPath    string
	ParityCosineMin float64
	Reason          string
}

func (e *UnsupportedBackendForwardError) Error() string {
	return fmt.Sprintf(
		"model: backend %q cannot execute forward %q via %q: %s; refusing generic QKV/CPU fallback (required deterministic CPU-reference parity cosine >= %.3f; issue #4714)",
		e.Backend, e.Forward, e.IntendedPath, e.Reason, e.ParityCosineMin,
	)
}

// ValidateBackendForwardPath checks the architecture/backend pair before a HAL
// session is constructed. A nil backend means the caller selected no HAL (the
// legacy NewSession CPU/reference path) and remains valid; NewBackendSessionChecked
// resolves its nil argument to compute.Default before calling this method. A
// recognized qwen35-family hybrid with any compute backend fails closed today:
// CUDA has no Qwen35GDNBackend implementation, and the model HAL has no branch that
// invokes the operation even if a marker-only test double claims it. The latter
// check is intentional; capability advertisement alone must never turn into readiness.
func (m *Model) ValidateBackendForwardPath(be compute.Backend) error {
	if m == nil || be == nil || !m.Cfg.IsQwen35Hybrid() {
		return nil
	}
	reason := "backend does not implement model.Qwen35GDNBackend"
	if gdn, ok := be.(Qwen35GDNBackend); ok {
		reason = fmt.Sprintf("backend advertises Qwen35GDNBackend path %q, but the model HAL does not yet dispatch linear-attention layers through Qwen35GDNDecode", gdn.Qwen35GDNPath())
	}
	return &UnsupportedBackendForwardError{
		Backend:         be.Name(),
		Forward:         ForwardQwen35GDN,
		IntendedPath:    Qwen35GDNCUDAPath,
		ParityCosineMin: Qwen35GDNParityCosineMin,
		Reason:          reason,
	}
}

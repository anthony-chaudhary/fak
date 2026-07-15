package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen35GDNCUDAPath is the production path identity reserved for a Qwen3.5/3.6
// Gated-DeltaNet/SSM token mixer implemented by the CUDA compute backend.
const Qwen35GDNCUDAPath = "cuda/qwen35-gdn-ssm-decode-v1"

// Qwen35GDNParityCosineMin is the deterministic device/reference acceptance floor.
const Qwen35GDNParityCosineMin = 0.999

// Qwen35GDNBackend is the whole-operation CUDA/HAL seam for the hybrid token mixer.
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

type qwen35GDNPathMarker interface {
	Qwen35GDNPath() string
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

// ValidateBackendForwardConfig checks an architecture/backend pair without requiring a
// constructed or weight-loaded Model. Serve header preflight can therefore refuse a
// missing, marker-only, or wrong-path hybrid backend before it allocates model storage.
// A nil backend still means the caller selected the legacy CPU/reference path; that path
// remains admitted and never enters the compute HAL.
func ValidateBackendForwardConfig(cfg Config, be compute.Backend) error {
	if be == nil || !cfg.IsQwen35Hybrid() {
		return nil
	}
	gdn, ok := be.(Qwen35GDNBackend)
	if ok && gdn.Qwen35GDNPath() == Qwen35GDNCUDAPath {
		return nil
	}
	reason := "backend does not structurally implement model.Qwen35GDNBackend"
	if marker, marked := be.(qwen35GDNPathMarker); marked {
		if !ok {
			reason = fmt.Sprintf("backend advertises marker path %q but does not structurally implement model.Qwen35GDNBackend", marker.Qwen35GDNPath())
		} else {
			reason = fmt.Sprintf("backend implements model.Qwen35GDNBackend with wrong path %q", marker.Qwen35GDNPath())
		}
	}
	return &UnsupportedBackendForwardError{
		Backend:         be.Name(),
		Forward:         ForwardQwen35GDN,
		IntendedPath:    Qwen35GDNCUDAPath,
		ParityCosineMin: Qwen35GDNParityCosineMin,
		Reason:          reason,
	}
}

// ValidateBackendForwardPath is the Model-bound twin retained for callers that already
// constructed a Model. All admission logic lives in ValidateBackendForwardConfig so the
// pre-load and session-construction decisions cannot drift.
func (m *Model) ValidateBackendForwardPath(be compute.Backend) error {
	if m == nil {
		return nil
	}
	return ValidateBackendForwardConfig(m.Cfg, be)
}

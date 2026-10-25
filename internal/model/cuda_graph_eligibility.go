package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// CUDA-graph capture-eligibility verdicts. They are stable vocabulary for the serve
// startup message and the HAL capture decision.
const (
	CUDAGraphEligible         = "eligible"
	CUDAGraphNoCaptureBackend = "backend-has-no-graph-capture"
)

// cudaGraphCaptureEligible is the one static capture predicate. It decides both whether a
// session's device KV is preallocated (through compute.KVConfig.GraphCapture) and, with
// cudaGraphCaptureReady's session checks, whether the HAL opens a capture.
func cudaGraphCaptureEligible(cfg Config, be compute.Backend) (bool, string) {
	if _, ok := be.(graphBackend); !ok {
		return false, CUDAGraphNoCaptureBackend
	}
	if cfg.IsQwen35Hybrid() {
		return qwen35HybridGraphEligible(cfg, be)
	}
	return true, CUDAGraphEligible
}

// cudaGraphCaptureReady adds the per-session conditions to the static predicate: an open,
// unfailed session with no captured target hidden, no split GPU-layer placement, a KV store
// holding the fixed graph capacity (a growable store may cudaMalloc on append, illegal inside
// a capture) and, for a hybrid, private ready GDN state and a KV depth below the QSA sparse
// gather threshold (that gather reads the host).
func (s *Session) cudaGraphCaptureReady(isSplit bool) bool {
	if s == nil || s.M == nil || s.halClosed || s.halFailure != nil || s.captureTargetHidden || isSplit {
		return false
	}
	if ok, _ := cudaGraphCaptureEligible(s.M.Cfg, s.Backend); !ok {
		return false
	}
	if s.halKV == nil {
		return false
	}
	if fixed, ok := s.halKV.(interface{ GraphCaptureReady() bool }); ok && !fixed.GraphCaptureReady() {
		return false
	}
	if cfg := s.M.Cfg; cfg.IsQwen35Hybrid() {
		if cfg.HasQSASparseAttn() && s.halKV.Len()+1 >= compute.QSADynamicGatingThreshold {
			return false
		}
		return s.qwen35HAL.graphCaptureReady(cfg)
	}
	return true
}

// CUDAGraphKVSizing is the startup view of one model's graph-replay KV: whether its sessions
// may capture, the KV rows they write and the fixed prealloc each would take.
type CUDAGraphKVSizing struct {
	Eligible bool
	Reason   string
	KVRows   int
	Prealloc compute.GraphKVPrealloc
}

// CUDAGraphKVSizingFor applies the capture predicate and the compute prealloc rule to cfg on be
// at positions, with free device bytes (compute.FreeUnknown fails open). It is what a
// session's NewKV will decide, so a serve can report it before loading weights.
func CUDAGraphKVSizingFor(cfg Config, be compute.Backend, graphOn bool, positions int, free int64) CUDAGraphKVSizing {
	eligible, reason := cudaGraphCaptureEligible(cfg, be)
	kc := halKVConfig(cfg)
	kc.GraphCapture = eligible
	return CUDAGraphKVSizing{
		Eligible: eligible,
		Reason:   reason,
		KVRows:   kc.NumLayers,
		Prealloc: compute.GraphKVPreallocFor(graphOn, kc, positions, free),
	}
}

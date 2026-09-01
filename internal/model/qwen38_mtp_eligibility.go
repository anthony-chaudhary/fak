package model

// Qwen38MTPEngine identifies the fak-native execution path selected for a Qwen3.8 request.
type Qwen38MTPEngine string

const (
	Qwen38EngineMTP          Qwen38MTPEngine = "fak-native-qwen3.8-mtp"
	Qwen38EngineTargetDecode Qwen38MTPEngine = "fak-native-qwen3.8-target-decode"
)

// Qwen38MTPBackend identifies the backend that actually executes the admitted MTP matrices.
type Qwen38MTPBackend string

const (
	Qwen38MTPBackendMetal Qwen38MTPBackend = "metal"
)

// Qwen38MTPDowngradeReason is the closed reason vocabulary for declining MTP acceleration.
type Qwen38MTPDowngradeReason string

const (
	Qwen38MTPEligible             Qwen38MTPDowngradeReason = ""
	Qwen38MTPModelUnsupported     Qwen38MTPDowngradeReason = "model_or_artifact_unsupported"
	Qwen38MTPBackendUnsupported   Qwen38MTPDowngradeReason = "backend_unsupported"
	Qwen38MTPPrecisionUnsupported Qwen38MTPDowngradeReason = "precision_unsupported"
	Qwen38MTPSamplingUnsupported  Qwen38MTPDowngradeReason = "sampling_mode_unsupported"
	Qwen38MTPDepthUnsupported     Qwen38MTPDowngradeReason = "depth_unsupported"
	Qwen38MTPSessionNotFresh      Qwen38MTPDowngradeReason = "session_not_fresh"
	Qwen38MTPMemoryUnsafe         Qwen38MTPDowngradeReason = "memory_headroom_unsafe"
	Qwen38MTPDisabledByPolicy     Qwen38MTPDowngradeReason = "disabled_by_operator_policy"
)

// Qwen38MTPEligibilityInput describes the complete, deliberately narrow MTP envelope.
// Model is authoritative when supplied: eligibility inspects its actual retained
// tensor stores rather than trusting an artifact label. F32 remains only for the
// pre-Q4_K compatibility caller that has no loaded model.
type Qwen38MTPEligibilityInput struct {
	Qwen38MTPArtifact bool
	MTPBackendReady   bool
	Backend           Qwen38MTPBackend
	Model             *Model
	F32               bool
	Greedy            bool
	Depth             int
	FreshSession      bool
	MemoryHeadroomOK  bool
	OperatorEnabled   bool
}

// Qwen38MTPEligibility is receipt-ready: Engine always names the fak-native path that
// executes, and DowngradeReason is non-empty exactly when ordinary target decode is selected.
type Qwen38MTPEligibility struct {
	Eligible         bool                     `json:"eligible"`
	Engine           Qwen38MTPEngine          `json:"engine"`
	Backend          Qwen38MTPBackend         `json:"backend,omitempty"`
	MTPTensorFormat  Qwen38MTPTensorFormat    `json:"mtp_tensor_format,omitempty"`
	RequestedDepth   int                      `json:"requested_depth"`
	TargetEquivalent bool                     `json:"target_equivalent"`
	DowngradeReason  Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
}

// EvaluateQwen38MTPEligibility selects MTP only inside its witnessed envelope. Every
// failure remains fak-native and reversibly downgrades to ordinary Qwen3.8 target decode;
// this function never selects llama.cpp or another engine.
func EvaluateQwen38MTPEligibility(in Qwen38MTPEligibilityInput) Qwen38MTPEligibility {
	reason := Qwen38MTPEligible
	format := Qwen38MTPFormatNone
	if !in.Qwen38MTPArtifact {
		reason = Qwen38MTPModelUnsupported
	} else if !in.MTPBackendReady {
		reason = Qwen38MTPBackendUnsupported
	} else if in.Model != nil {
		layout, err := in.Model.Qwen38MTPTensorLayout()
		if err != nil {
			reason = Qwen38MTPPrecisionUnsupported
		} else {
			format = layout.Format
			if format == Qwen38MTPFormatQ4K && in.Backend != Qwen38MTPBackendMetal {
				reason = Qwen38MTPBackendUnsupported
			}
		}
	} else if in.F32 {
		format = Qwen38MTPFormatF32
	} else {
		reason = Qwen38MTPPrecisionUnsupported
	}
	switch {
	case reason != Qwen38MTPEligible:
	case !in.Greedy:
		reason = Qwen38MTPSamplingUnsupported
	case in.Depth <= 0 || in.Depth > Qwen35MTPMaxDraftDepth:
		reason = Qwen38MTPDepthUnsupported
	case !in.FreshSession:
		reason = Qwen38MTPSessionNotFresh
	case !in.MemoryHeadroomOK:
		reason = Qwen38MTPMemoryUnsafe
	case !in.OperatorEnabled:
		reason = Qwen38MTPDisabledByPolicy
	}
	if reason != Qwen38MTPEligible {
		return Qwen38MTPEligibility{
			Engine:          Qwen38EngineTargetDecode,
			RequestedDepth:  in.Depth,
			DowngradeReason: reason,
		}
	}
	return Qwen38MTPEligibility{
		Eligible:         true,
		Engine:           Qwen38EngineMTP,
		Backend:          in.Backend,
		MTPTensorFormat:  format,
		RequestedDepth:   in.Depth,
		TargetEquivalent: true,
	}
}

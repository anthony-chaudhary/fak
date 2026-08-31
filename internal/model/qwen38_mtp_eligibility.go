package model

// Qwen38MTPEngine identifies the fak-native execution path selected for a Qwen3.8 request.
type Qwen38MTPEngine string

const (
	Qwen38EngineMTP          Qwen38MTPEngine = "fak-native-qwen3.8-mtp"
	Qwen38EngineTargetDecode Qwen38MTPEngine = "fak-native-qwen3.8-target-decode"
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
// MTP is currently a Qwen3.8 native-F32, greedy, depth-one, fresh-session path.
type Qwen38MTPEligibilityInput struct {
	Qwen38MTPArtifact bool
	MTPBackendReady   bool
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
	Eligible        bool                     `json:"eligible"`
	Engine          Qwen38MTPEngine          `json:"engine"`
	DowngradeReason Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
}

// EvaluateQwen38MTPEligibility selects MTP only inside its witnessed envelope. Every
// failure remains fak-native and reversibly downgrades to ordinary Qwen3.8 target decode;
// this function never selects llama.cpp or another engine.
func EvaluateQwen38MTPEligibility(in Qwen38MTPEligibilityInput) Qwen38MTPEligibility {
	reason := Qwen38MTPEligible
	switch {
	case !in.Qwen38MTPArtifact:
		reason = Qwen38MTPModelUnsupported
	case !in.MTPBackendReady:
		reason = Qwen38MTPBackendUnsupported
	case !in.F32:
		reason = Qwen38MTPPrecisionUnsupported
	case !in.Greedy:
		reason = Qwen38MTPSamplingUnsupported
	case in.Depth != 1:
		reason = Qwen38MTPDepthUnsupported
	case !in.FreshSession:
		reason = Qwen38MTPSessionNotFresh
	case !in.MemoryHeadroomOK:
		reason = Qwen38MTPMemoryUnsafe
	case !in.OperatorEnabled:
		reason = Qwen38MTPDisabledByPolicy
	}
	if reason != Qwen38MTPEligible {
		return Qwen38MTPEligibility{Engine: Qwen38EngineTargetDecode, DowngradeReason: reason}
	}
	return Qwen38MTPEligibility{Eligible: true, Engine: Qwen38EngineMTP}
}

package model

import "fmt"

const Qwen38MTPReceiptSchema = "fak/qwen38-mtp-receipt/v1"

// Qwen38MTPReceiptOutcome identifies whether MTP ran and whether it completed.
type Qwen38MTPReceiptOutcome string

const (
	Qwen38MTPOutcomeTargetOnly Qwen38MTPReceiptOutcome = "target_only"
	Qwen38MTPOutcomeSucceeded  Qwen38MTPReceiptOutcome = "speculative_succeeded"
	Qwen38MTPOutcomeFailed     Qwen38MTPReceiptOutcome = "speculative_failed"
)

// Qwen38MTPFailureReason is the closed vocabulary for failures after an MTP attempt starts.
type Qwen38MTPFailureReason string

const (
	Qwen38MTPFailureNone           Qwen38MTPFailureReason = ""
	Qwen38MTPDraftFailed           Qwen38MTPFailureReason = "draft_failed"
	Qwen38MTPVerificationFailed    Qwen38MTPFailureReason = "verification_failed"
	Qwen38MTPRollbackFailed        Qwen38MTPFailureReason = "rollback_failed"
	Qwen38MTPSynchronizationFailed Qwen38MTPFailureReason = "synchronization_failed"
	Qwen38MTPRecoveryFailed        Qwen38MTPFailureReason = "recovery_failed"
)

// Additional downgrade reasons used after a speculative attempt or its evidence gate fails.
const (
	Qwen38MTPCorrectnessDiverged    Qwen38MTPDowngradeReason = "correctness_diverged"
	Qwen38MTPQualityOutsideEnvelope Qwen38MTPDowngradeReason = "quality_outside_envelope"
	Qwen38MTPNetLatencyRegressed    Qwen38MTPDowngradeReason = "net_latency_regressed"
	Qwen38MTPReceiptInvalid         Qwen38MTPDowngradeReason = "receipt_invalid"
	Qwen38MTPAttemptFailed          Qwen38MTPDowngradeReason = "attempt_failed"
)

// Qwen38MTPAcceptanceBucket records verifier disposition at one draft depth.
type Qwen38MTPAcceptanceBucket struct {
	Depth    int `json:"depth"`
	Proposed int `json:"proposed"`
	Accepted int `json:"accepted"`
	Rejected int `json:"rejected"`
}

// Qwen38MTPTokenAccounting records aggregate and per-depth speculative token disposition.
type Qwen38MTPTokenAccounting struct {
	Proposed     int                         `json:"proposed"`
	Accepted     int                         `json:"accepted"`
	Rejected     int                         `json:"rejected"`
	Distribution []Qwen38MTPAcceptanceBucket `json:"distribution,omitempty"`
}

// Qwen38MTPLatencyNS records net end-to-end cost. Total must equal every phase,
// including setup and recovery rather than reporting only the accelerated region.
type Qwen38MTPLatencyNS struct {
	Setup    uint64 `json:"setup"`
	Draft    uint64 `json:"draft"`
	Verify   uint64 `json:"verify"`
	Rollback uint64 `json:"rollback"`
	Sync     uint64 `json:"sync"`
	Recovery uint64 `json:"recovery"`
	Total    uint64 `json:"total"`
}

// Qwen38MTPMemoryBytes records speculative workspace and peak live memory.
// Peak is a high-water mark and therefore must cover, but does not add, phase workspaces.
type Qwen38MTPMemoryBytes struct {
	DraftWorkspace  uint64 `json:"draft_workspace"`
	VerifyWorkspace uint64 `json:"verify_workspace"`
	RollbackState   uint64 `json:"rollback_state"`
	Peak            uint64 `json:"peak"`
}

// Qwen38MTPReceipt is the schema witness for a Qwen3.8 MTP decision or attempt.
// Engine and FallbackEngine use a closed fak-native vocabulary; no llama.cpp or
// unspecified execution path can validate.
type Qwen38MTPReceipt struct {
	SchemaVersion   string                   `json:"schema_version"`
	Outcome         Qwen38MTPReceiptOutcome  `json:"outcome"`
	Engine          Qwen38MTPEngine          `json:"engine"`
	FallbackEngine  Qwen38MTPEngine          `json:"fallback_engine,omitempty"`
	RequestedDepth  int                      `json:"requested_depth"`
	EffectiveDepth  int                      `json:"effective_depth"`
	Tokens          Qwen38MTPTokenAccounting `json:"tokens"`
	LatencyNS       Qwen38MTPLatencyNS       `json:"latency_ns"`
	MemoryBytes     Qwen38MTPMemoryBytes     `json:"memory_bytes"`
	DowngradeReason Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
	FailureReason   Qwen38MTPFailureReason   `json:"failure_reason,omitempty"`
}

// Validate rejects receipts that cannot prove a fully fak-native execution path
// with internally consistent token and end-to-end cost accounting.
func (r Qwen38MTPReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPReceiptSchema {
		return fmt.Errorf("model: qwen3.8 MTP receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPReceiptSchema)
	}
	if !validQwen38MTPEngine(r.Engine) {
		return fmt.Errorf("model: qwen3.8 MTP receipt engine %q is absent or not fak-native", r.Engine)
	}
	if r.FallbackEngine != "" && !validQwen38MTPEngine(r.FallbackEngine) {
		return fmt.Errorf("model: qwen3.8 MTP receipt fallback engine %q is not fak-native", r.FallbackEngine)
	}
	if r.RequestedDepth < 0 || r.EffectiveDepth < 0 || r.EffectiveDepth > r.RequestedDepth {
		return fmt.Errorf("model: qwen3.8 MTP receipt impossible depths requested=%d effective=%d", r.RequestedDepth, r.EffectiveDepth)
	}
	if err := r.Tokens.validate(r.EffectiveDepth); err != nil {
		return err
	}
	if err := r.LatencyNS.validate(); err != nil {
		return err
	}
	if err := r.MemoryBytes.validate(); err != nil {
		return err
	}
	if !validQwen38MTPDowngradeReason(r.DowngradeReason) {
		return fmt.Errorf("model: qwen3.8 MTP receipt unknown downgrade reason %q", r.DowngradeReason)
	}
	if !validQwen38MTPFailureReason(r.FailureReason) {
		return fmt.Errorf("model: qwen3.8 MTP receipt unknown failure reason %q", r.FailureReason)
	}

	speculativeCost := r.LatencyNS.Draft + r.LatencyNS.Verify + r.LatencyNS.Rollback + r.LatencyNS.Sync + r.LatencyNS.Recovery
	switch r.Outcome {
	case Qwen38MTPOutcomeTargetOnly:
		if r.Engine != Qwen38EngineTargetDecode || r.FallbackEngine != "" {
			return fmt.Errorf("model: target-only Qwen3.8 receipt must name only fak-native target decode")
		}
		if r.RequestedDepth == 0 || r.EffectiveDepth != 0 || r.Tokens.Proposed != 0 || len(r.Tokens.Distribution) != 0 || speculativeCost != 0 {
			return fmt.Errorf("model: target-only Qwen3.8 receipt contains speculative work")
		}
		if r.DowngradeReason == Qwen38MTPEligible || r.FailureReason != Qwen38MTPFailureNone {
			return fmt.Errorf("model: target-only Qwen3.8 receipt requires downgrade reason and no failure reason")
		}
	case Qwen38MTPOutcomeSucceeded:
		if r.Engine != Qwen38EngineMTP || r.FallbackEngine != "" {
			return fmt.Errorf("model: successful Qwen3.8 MTP receipt must name only the fak-native MTP engine")
		}
		if r.RequestedDepth == 0 || r.EffectiveDepth == 0 || r.Tokens.Proposed == 0 || r.Tokens.Accepted == 0 {
			return fmt.Errorf("model: successful Qwen3.8 MTP receipt has no successful speculative work")
		}
		if r.DowngradeReason != Qwen38MTPEligible || r.FailureReason != Qwen38MTPFailureNone {
			return fmt.Errorf("model: successful Qwen3.8 MTP receipt cannot contain failure or downgrade reasons")
		}
		if r.LatencyNS.Draft == 0 || r.LatencyNS.Verify == 0 || r.MemoryBytes.Peak == 0 {
			return fmt.Errorf("model: successful Qwen3.8 MTP receipt omits draft, verify, or memory cost")
		}
	case Qwen38MTPOutcomeFailed:
		if r.Engine != Qwen38EngineMTP || r.FallbackEngine != Qwen38EngineTargetDecode {
			return fmt.Errorf("model: failed Qwen3.8 MTP receipt must identify MTP and fak-native target fallback")
		}
		if r.RequestedDepth == 0 || r.FailureReason == Qwen38MTPFailureNone || r.DowngradeReason == Qwen38MTPEligible {
			return fmt.Errorf("model: failed Qwen3.8 MTP receipt requires requested depth, failure reason, and downgrade reason")
		}
		if r.LatencyNS.Total == 0 {
			return fmt.Errorf("model: failed Qwen3.8 MTP receipt omits incurred cost")
		}
	default:
		return fmt.Errorf("model: qwen3.8 MTP receipt unknown outcome %q", r.Outcome)
	}
	return nil
}

func (t Qwen38MTPTokenAccounting) validate(effectiveDepth int) error {
	if t.Proposed < 0 || t.Accepted < 0 || t.Rejected < 0 || t.Proposed != t.Accepted+t.Rejected {
		return fmt.Errorf("model: qwen3.8 MTP receipt impossible token totals proposed=%d accepted=%d rejected=%d", t.Proposed, t.Accepted, t.Rejected)
	}
	proposed, accepted, rejected := 0, 0, 0
	seen := make(map[int]bool, len(t.Distribution))
	for _, bucket := range t.Distribution {
		if bucket.Depth <= 0 || bucket.Depth > effectiveDepth || seen[bucket.Depth] || bucket.Proposed < 0 || bucket.Accepted < 0 || bucket.Rejected < 0 || bucket.Proposed != bucket.Accepted+bucket.Rejected {
			return fmt.Errorf("model: qwen3.8 MTP receipt impossible acceptance bucket at depth %d", bucket.Depth)
		}
		seen[bucket.Depth] = true
		proposed += bucket.Proposed
		accepted += bucket.Accepted
		rejected += bucket.Rejected
	}
	if proposed != t.Proposed || accepted != t.Accepted || rejected != t.Rejected {
		return fmt.Errorf("model: qwen3.8 MTP receipt distribution totals proposed=%d accepted=%d rejected=%d, want %d/%d/%d", proposed, accepted, rejected, t.Proposed, t.Accepted, t.Rejected)
	}
	return nil
}

func (l Qwen38MTPLatencyNS) validate() error {
	parts := []uint64{l.Setup, l.Draft, l.Verify, l.Rollback, l.Sync, l.Recovery}
	var sum uint64
	for _, part := range parts {
		if ^uint64(0)-sum < part {
			return fmt.Errorf("model: qwen3.8 MTP receipt latency accounting overflow")
		}
		sum += part
	}
	if l.Total != sum {
		return fmt.Errorf("model: qwen3.8 MTP receipt non-additive latency total=%d want=%d", l.Total, sum)
	}
	return nil
}

func (m Qwen38MTPMemoryBytes) validate() error {
	if m.Peak < m.DraftWorkspace || m.Peak < m.VerifyWorkspace || m.Peak < m.RollbackState {
		return fmt.Errorf("model: qwen3.8 MTP receipt peak memory %d is below a phase workspace", m.Peak)
	}
	return nil
}

func validQwen38MTPEngine(engine Qwen38MTPEngine) bool {
	return engine == Qwen38EngineMTP || engine == Qwen38EngineTargetDecode
}

func validQwen38MTPFailureReason(reason Qwen38MTPFailureReason) bool {
	switch reason {
	case Qwen38MTPFailureNone, Qwen38MTPDraftFailed, Qwen38MTPVerificationFailed, Qwen38MTPRollbackFailed, Qwen38MTPSynchronizationFailed, Qwen38MTPRecoveryFailed:
		return true
	default:
		return false
	}
}

func validQwen38MTPDowngradeReason(reason Qwen38MTPDowngradeReason) bool {
	switch reason {
	case Qwen38MTPEligible,
		Qwen38MTPModelUnsupported, Qwen38MTPBackendUnsupported, Qwen38MTPPrecisionUnsupported,
		Qwen38MTPSamplingUnsupported, Qwen38MTPDepthUnsupported, Qwen38MTPSessionNotFresh,
		Qwen38MTPMemoryUnsafe, Qwen38MTPDisabledByPolicy, Qwen38MTPCorrectnessDiverged,
		Qwen38MTPQualityOutsideEnvelope, Qwen38MTPNetLatencyRegressed, Qwen38MTPReceiptInvalid,
		Qwen38MTPAttemptFailed:
		return true
	default:
		return false
	}
}

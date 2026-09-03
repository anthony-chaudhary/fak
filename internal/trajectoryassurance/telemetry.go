package trajectoryassurance

import (
	"errors"
)

// VerdictTelemetry captures adjudication counters, refusal reasons, and taint tags.
type VerdictTelemetry struct {
	AllowCount      int      `json:"allow_count"`
	DenyCount       int      `json:"deny_count"`
	QuarantineCount int      `json:"quarantine_count"`
	RefusalReasons  []string `json:"refusal_reasons,omitempty"`
	TaintTags       []string `json:"taint_tags,omitempty"`
}

// CompressTelemetry captures compaction attempts, bail reasons, prefix preservation, and token savings.
type CompressTelemetry struct {
	Attempts          int      `json:"attempts"`
	BailReasons       []string `json:"bail_reasons,omitempty"`
	PrefixPreserved   bool     `json:"prefix_preserved"`
	TokenShedRatio    float64  `json:"token_shed_ratio"`
	PostFireHitTokens int64    `json:"post_fire_hit_tokens"`
}

// DelegationTelemetry captures lease activity and effect reconciliation for delegation.
type DelegationTelemetry struct {
	LaneLeaseIDs      []string `json:"lane_lease_ids,omitempty"`
	ConcurrentWorkers int      `json:"concurrent_workers"`
	LeaseCollisions   int      `json:"lease_collisions"`
	ReconciledEffects int      `json:"reconciled_effects"`
	DivergedEffects   int      `json:"diverged_effects"`
	UnobservedEffects int      `json:"unobserved_effects"`
}

// ProgressTelemetry captures trajectory progress indicators, curve state, and intervention regret.
type ProgressTelemetry struct {
	WitnessRung        string  `json:"witness_rung,omitempty"`
	CurveState         string  `json:"curve_state,omitempty"`
	RegimeAction       string  `json:"regime_action,omitempty"`
	InterventionRegret float64 `json:"intervention_regret,omitempty"`
}

// InferenceTelemetry captures inference engine verification and KV cache efficiency.
type InferenceTelemetry struct {
	RuntimeReceipt         string  `json:"runtime_receipt,omitempty"`
	FakNativeVerified      bool    `json:"fak_native_verified"`
	KVBlockAllocEfficiency float64 `json:"kv_block_alloc_efficiency"`
}

// FakCoreTelemetry aggregates kernel telemetry across adjudication, compaction, delegation, progress, and inference.
type FakCoreTelemetry struct {
	Adjudication *VerdictTelemetry    `json:"adjudication,omitempty"`
	Compaction   *CompressTelemetry   `json:"compaction,omitempty"`
	Delegation   *DelegationTelemetry `json:"delegation,omitempty"`
	Progress     *ProgressTelemetry   `json:"progress,omitempty"`
	Inference    *InferenceTelemetry  `json:"inference,omitempty"`
}

// Validate checks telemetry fields for consistency and validity.
func (t *FakCoreTelemetry) Validate() error {
	if t == nil {
		return nil
	}
	if t.Adjudication != nil {
		if t.Adjudication.AllowCount < 0 || t.Adjudication.DenyCount < 0 || t.Adjudication.QuarantineCount < 0 {
			return errors.New("negative adjudication counts")
		}
	}
	if t.Compaction != nil {
		if t.Compaction.Attempts < 0 {
			return errors.New("negative compaction attempts")
		}
		if t.Compaction.TokenShedRatio < 0 || t.Compaction.TokenShedRatio > 1.0 {
			return errors.New("compaction token shed ratio must be between 0 and 1")
		}
		if t.Compaction.PostFireHitTokens < 0 {
			return errors.New("negative post fire hit tokens")
		}
	}
	if t.Delegation != nil {
		if t.Delegation.ConcurrentWorkers < 0 || t.Delegation.LeaseCollisions < 0 ||
			t.Delegation.ReconciledEffects < 0 || t.Delegation.DivergedEffects < 0 ||
			t.Delegation.UnobservedEffects < 0 {
			return errors.New("negative delegation counts")
		}
	}
	if t.Progress != nil {
		if t.Progress.InterventionRegret < 0 {
			return errors.New("negative intervention regret")
		}
	}
	if t.Inference != nil {
		if t.Inference.KVBlockAllocEfficiency < 0 || t.Inference.KVBlockAllocEfficiency > 1.0 {
			return errors.New("kv block allocation efficiency must be between 0 and 1")
		}
	}
	return nil
}

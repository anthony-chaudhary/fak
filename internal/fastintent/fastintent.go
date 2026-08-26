// Package fastintent joins the latency-intent plan, provider readback, and
// quality-constrained evaluator into one replayable receipt.
package fastintent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

const Schema = "fak-fast-intent-replay/1"

type ProviderRealization struct {
	Provider string                        `json:"provider"`
	Receipt  modelroute.ServiceTierReceipt `json:"receipt"`
}

type ReplayBundle struct {
	Schema         string                           `json:"schema"`
	Plan           orchestration.FastExecutionPlan  `json:"plan"`
	Providers      []ProviderRealization            `json:"providers"`
	Evaluation     ultracodebench.FastProfileReport `json:"evaluation"`
	Verdict        string                           `json:"verdict"`
	EvidenceDigest string                           `json:"evidence_digest"`
}

// Join accepts only realized provider outcomes and a quality-constrained paired
// evaluation. It never upgrades an abstention or hides a tier downgrade.
func Join(plan orchestration.FastExecutionPlan, providers []ProviderRealization, evaluation ultracodebench.FastProfileReport) (ReplayBundle, error) {
	r := ReplayBundle{Schema: Schema, Plan: plan, Providers: append([]ProviderRealization(nil), providers...), Evaluation: evaluation, Verdict: evaluation.Verdict}
	if plan.Schema != orchestration.FastIntentSchemaVersion || plan.Requested.QualityFloor == "" {
		return r, fmt.Errorf("fast intent: valid plan and quality floor required")
	}
	if len(providers) < 2 {
		return r, fmt.Errorf("fast intent: at least two provider realizations required")
	}
	seen := map[string]bool{}
	for _, p := range providers {
		if p.Provider == "" || seen[p.Provider] {
			return r, fmt.Errorf("fast intent: providers must be named and unique")
		}
		seen[p.Provider] = true
		q := p.Receipt
		if q.Realized == modelroute.ServiceModeUnknown {
			return r, fmt.Errorf("fast intent: %s has no realized tier", p.Provider)
		}
		if q.Requested != q.Realized && q.DowngradeReason == "" {
			return r, fmt.Errorf("fast intent: %s silently downgraded", p.Provider)
		}
	}
	if evaluation.Schema != ultracodebench.FastProfileSchema || evaluation.BundleDigest == "" {
		return r, fmt.Errorf("fast intent: replayable quality evaluation required")
	}
	switch evaluation.Verdict {
	case "GAIN", "NO_GAIN", "ABSTAIN":
	default:
		return r, fmt.Errorf("fast intent: invalid evaluator verdict %q", evaluation.Verdict)
	}
	canonical := r
	canonical.EvidenceDigest = ""
	b, err := json.Marshal(canonical)
	if err != nil {
		return r, err
	}
	sum := sha256.Sum256(b)
	r.EvidenceDigest = "sha256:" + hex.EncodeToString(sum[:])
	return r, nil
}

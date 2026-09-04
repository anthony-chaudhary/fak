package ultracodedogfood

import (
	"encoding/json"
	"fmt"
)

// Invariant: UltraCode dogfood lifecycle sessions are fail-closed and deterministic.
// Guard: Ambiguous boundary evidence, missing provider cache receipts, or mismatched outcome digests force an ABSTAIN verdict or validation failure.
const LifecycleSessionSchema = "fak-ultracode-lifecycle-session/1"

// LifecycleSession records an end-to-end task execution across multiple lifecycle boundaries.
type LifecycleSession struct {
	Schema                 string          `json:"schema"`
	Model                  string          `json:"model"`
	Runtime                string          `json:"runtime"`
	Tokenizer              string          `json:"tokenizer"`
	TaskDigest             string          `json:"task_digest"`
	FullContextInputTokens int64           `json:"full_context_input_tokens"`
	Cells                  []LifecycleCell `json:"cells"`
}

// LifecycleCell records the execution boundary evidence and token metrics for a single lifecycle stage.
type LifecycleCell struct {
	Boundary                 string                     `json:"boundary"`
	BoundaryEvidence         *LifecycleBoundaryEvidence `json:"boundary_evidence"`
	AcceptedOutcome          bool                       `json:"accepted_outcome"`
	OutcomeDigest            string                     `json:"outcome_digest"`
	ScopedContextInputTokens int64                      `json:"scoped_context_input_tokens"`
	ProviderCacheReadTokens  *int64                     `json:"provider_cache_read_tokens"`
	ProviderCacheEvidence    string                     `json:"provider_cache_evidence"`
}

// LifecycleBoundaryEvidence stores authoritative proof and receipts confirming the boundary transition.
type LifecycleBoundaryEvidence struct {
	Kind    string `json:"kind"`
	Receipt string `json:"receipt"`
}

// LifecycleReport encapsulates the evaluation verdict and token accounting across all lifecycle cells.
type LifecycleReport struct {
	Schema             string                `json:"schema"`
	Verdict            string                `json:"verdict"`
	Reason             string                `json:"reason,omitempty"`
	ScopeAvoidedTokens int64                 `json:"scope_avoided_tokens"`
	Cells              []LifecycleReportCell `json:"cells"`
}

// LifecycleReportCell summarizes the verified outcome and avoided token delta for a specific boundary cell.
type LifecycleReportCell struct {
	Boundary                string `json:"boundary"`
	Status                  string `json:"status"`
	ScopeAvoidedTokens      int64  `json:"scope_avoided_tokens"`
	ProviderCacheReadTokens *int64 `json:"provider_cache_read_tokens,omitempty"`
}

var lifecycleOrder = []string{"cold", "warm_reuse", "ttl_expiry", "ttl_recovered", "explicit_clear", "clear_recovered", "compaction", "compaction_recovered"}

// EvaluateLifecycleSession keeps fak-owned omission separate from provider-owned
// prefix reuse. Missing authoritative lifecycle or cache evidence is UNKNOWN,
// never estimated from latency or prompt-evaluation duration.
func EvaluateLifecycleSession(data []byte) (LifecycleReport, error) {
	var s LifecycleSession
	if err := json.Unmarshal(data, &s); err != nil {
		return LifecycleReport{}, fmt.Errorf("decode lifecycle session: %w", err)
	}
	r := LifecycleReport{Schema: "fak-ultracode-lifecycle-report/1", Verdict: "PASS"}
	if s.Schema != LifecycleSessionSchema || s.Model == "" || s.Runtime == "" || s.Tokenizer == "" || s.TaskDigest == "" || s.FullContextInputTokens <= 0 {
		return LifecycleReport{}, fmt.Errorf("incomplete lifecycle session envelope")
	}
	if len(s.Cells) != len(lifecycleOrder) {
		return LifecycleReport{}, fmt.Errorf("want %d lifecycle cells, got %d", len(lifecycleOrder), len(s.Cells))
	}
	var firstScope int64
	for i, c := range s.Cells {
		if c.Boundary != lifecycleOrder[i] {
			return LifecycleReport{}, fmt.Errorf("cell %d: want boundary %q, got %q", i, lifecycleOrder[i], c.Boundary)
		}
		rc := LifecycleReportCell{Boundary: c.Boundary, Status: "KNOWN"}
		if c.BoundaryEvidence == nil || c.BoundaryEvidence.Kind == "" || c.BoundaryEvidence.Receipt == "" || c.ProviderCacheReadTokens == nil || c.ProviderCacheEvidence == "" {
			rc.Status = "UNKNOWN"
			r.Verdict = "ABSTAIN"
			r.Reason = "AMBIGUOUS_LIFECYCLE_BOUNDARY"
			r.Cells = append(r.Cells, rc)
			continue
		}
		if !c.AcceptedOutcome || c.OutcomeDigest != s.TaskDigest {
			return LifecycleReport{}, fmt.Errorf("%s: unequal accepted outcome", c.Boundary)
		}
		if c.ScopedContextInputTokens <= 0 || c.ScopedContextInputTokens >= s.FullContextInputTokens {
			return LifecycleReport{}, fmt.Errorf("%s: scoped context does not avoid work", c.Boundary)
		}
		rc.ScopeAvoidedTokens = s.FullContextInputTokens - c.ScopedContextInputTokens
		rc.ProviderCacheReadTokens = c.ProviderCacheReadTokens
		if firstScope == 0 {
			firstScope = rc.ScopeAvoidedTokens
		} else if rc.ScopeAvoidedTokens != firstScope {
			return LifecycleReport{}, fmt.Errorf("%s: scope avoidance changed across lifecycle", c.Boundary)
		}
		r.Cells = append(r.Cells, rc)
	}
	r.ScopeAvoidedTokens = firstScope
	if r.Verdict == "PASS" {
		for _, i := range []int{0, 2, 4} {
			if *r.Cells[i].ProviderCacheReadTokens != 0 {
				return LifecycleReport{}, fmt.Errorf("%s: cache contribution did not reset", r.Cells[i].Boundary)
			}
		}
		for _, i := range []int{1, 3, 5, 7} {
			if *r.Cells[i].ProviderCacheReadTokens <= 0 {
				return LifecycleReport{}, fmt.Errorf("%s: cache contribution did not recover", r.Cells[i].Boundary)
			}
		}
	}
	return r, nil
}

package orchestration

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strings"
	"time"
)

const EffectReceiptSchema = "fak.orchestration_effect_receipt.v1"

type EffectVerificationState string

const (
	EffectVerified EffectVerificationState = "VERIFIED"
	EffectFailed   EffectVerificationState = "FAILED"
	EffectUnknown  EffectVerificationState = "UNKNOWN"
)

type EffectReconciliation string

const (
	EffectReconciled EffectReconciliation = "RECONCILED"
	EffectDiverged   EffectReconciliation = "DIVERGED"
	EffectUnobserved EffectReconciliation = "UNOBSERVED"
)

// AdmittedEffect identifies one child and the exact effect admitted for it.
// Admission, activation, and child reports are retained only to make their
// non-witness status explicit; none can establish EffectVerified.
type AdmittedEffect struct {
	RunID                  string
	ChildID                string
	SuccessorID            string
	EffectClass            string
	ExpectedScrubbedDigest string
	Admitted               bool
	Activated              bool
	ChildReportedSuccess   bool
}

// EffectWitnessAuthority identifies who performed post-effect readback.
// AuthorChildID is empty for an external authority and equals ChildID when the
// purported witness was authored by the child whose effect is being checked.
type EffectWitnessAuthority struct {
	AuthorityID   string `json:"authority_id"`
	AuthorChildID string `json:"author_child_id,omitempty"`
}

// EffectReceipt is a data-only, privacy-safe post-effect readback. Digests must
// be lowercase SHA-256 values over scrubbed artifacts; raw material has no
// representation in the supported schema.
type EffectReceipt struct {
	Schema                 string                  `json:"schema"`
	RunID                  string                  `json:"run_id"`
	ChildID                string                  `json:"child_id"`
	SuccessorID            string                  `json:"successor_id"`
	EffectClass            string                  `json:"effect_class"`
	ExpectedScrubbedDigest string                  `json:"expected_scrubbed_digest"`
	ObservedScrubbedDigest string                  `json:"observed_scrubbed_digest"`
	ReadbackMethod         string                  `json:"readback_method"`
	Witness                EffectWitnessAuthority  `json:"witness"`
	Reconciliation         EffectReconciliation    `json:"reconciliation"`
	State                  EffectVerificationState `json:"state"`
	ObservedAt             time.Time               `json:"observed_at"`
	// UnsupportedFields is a fail-closed ingest sentinel containing field names
	// only; adapters must never retain rejected field values.
	UnsupportedFields []string `json:"unsupported_fields,omitempty"`
}

type EffectCohortResult struct {
	State    EffectVerificationState `json:"state"`
	Children []EffectChildResult     `json:"children"`
}

type EffectChildResult struct {
	ChildID string                  `json:"child_id"`
	State   EffectVerificationState `json:"state"`
	Reason  string                  `json:"reason,omitempty"`
}

// DecodeEffectReceiptJSON decodes exactly one receipt and rejects every field
// outside the privacy-safe schema. Errors intentionally omit input values.
func DecodeEffectReceiptJSON(data []byte) (EffectReceipt, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var receipt EffectReceipt
	if err := decoder.Decode(&receipt); err != nil {
		return EffectReceipt{}, errors.New("invalid effect receipt")
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return EffectReceipt{}, errors.New("invalid effect receipt")
	}
	return receipt, nil
}

// ValidateEffectCohort reconciles receipts against an admitted launch cohort.
// It is pure: now and freshness are explicit inputs and validation performs no
// filesystem, network, callback, or command work.
func ValidateEffectCohort(runID string, admitted []AdmittedEffect, receipts []EffectReceipt, now time.Time, freshness time.Duration) EffectCohortResult {
	byChild := make(map[string]AdmittedEffect, len(admitted))
	results := make(map[string]EffectChildResult, len(admitted))
	cohortFailed := len(admitted) == 0

	for _, a := range admitted {
		id := strings.TrimSpace(a.ChildID)
		if id == "" || !a.Admitted || strings.TrimSpace(a.RunID) != strings.TrimSpace(runID) ||
			strings.TrimSpace(a.SuccessorID) == "" || strings.TrimSpace(a.EffectClass) == "" || !canonicalSHA256(a.ExpectedScrubbedDigest) {
			cohortFailed = true
			results[id] = EffectChildResult{ChildID: id, State: EffectFailed, Reason: "invalid admission"}
			continue
		}
		if _, exists := byChild[id]; exists {
			cohortFailed = true
			results[id] = EffectChildResult{ChildID: id, State: EffectFailed, Reason: "duplicate admitted child"}
			continue
		}
		byChild[id] = a
	}

	seen := make(map[string]bool, len(receipts))
	for _, receipt := range receipts {
		childID := strings.TrimSpace(receipt.ChildID)
		a, launched := byChild[childID]
		if !launched {
			cohortFailed = true
			results[childID] = EffectChildResult{ChildID: childID, State: EffectFailed, Reason: "unlaunched child"}
			continue
		}
		if seen[childID] {
			cohortFailed = true
			results[childID] = EffectChildResult{ChildID: childID, State: EffectFailed, Reason: "duplicate receipt"}
			continue
		}
		seen[childID] = true
		state, reason := validateEffectReceipt(runID, a, receipt, now, freshness)
		results[childID] = EffectChildResult{ChildID: childID, State: state, Reason: reason}
		if state == EffectFailed {
			cohortFailed = true
		}
	}

	for childID := range byChild {
		if !seen[childID] {
			results[childID] = EffectChildResult{ChildID: childID, State: EffectUnknown, Reason: "missing receipt"}
		}
	}

	out := EffectCohortResult{State: EffectVerified, Children: make([]EffectChildResult, 0, len(results))}
	for _, result := range results {
		out.Children = append(out.Children, result)
		if result.State == EffectUnknown && out.State == EffectVerified {
			out.State = EffectUnknown
		}
	}
	if cohortFailed {
		out.State = EffectFailed
	}
	sort.Slice(out.Children, func(i, j int) bool { return out.Children[i].ChildID < out.Children[j].ChildID })
	return out
}

func validateEffectReceipt(runID string, admitted AdmittedEffect, receipt EffectReceipt, now time.Time, freshness time.Duration) (EffectVerificationState, string) {
	if receipt.Schema != EffectReceiptSchema {
		return EffectUnknown, "unsupported schema"
	}
	if len(receipt.UnsupportedFields) != 0 {
		return EffectFailed, "raw or private fields present"
	}
	if receipt.State != EffectVerified && receipt.State != EffectFailed && receipt.State != EffectUnknown {
		return EffectUnknown, "unsupported state"
	}
	if receipt.Reconciliation != EffectReconciled && receipt.Reconciliation != EffectDiverged && receipt.Reconciliation != EffectUnobserved {
		return EffectUnknown, "unsupported reconciliation"
	}
	if strings.TrimSpace(receipt.Witness.AuthorityID) == "" || strings.TrimSpace(receipt.ReadbackMethod) == "" {
		return EffectUnknown, "missing witness authority or readback method"
	}
	if strings.TrimSpace(receipt.Witness.AuthorChildID) == admitted.ChildID || strings.TrimSpace(receipt.Witness.AuthorityID) == admitted.ChildID {
		return EffectUnknown, "self-authored witness"
	}
	if freshness <= 0 || receipt.ObservedAt.IsZero() || receipt.ObservedAt.After(now) || now.Sub(receipt.ObservedAt) > freshness {
		return EffectUnknown, "stale or future receipt"
	}
	if strings.TrimSpace(receipt.RunID) != strings.TrimSpace(runID) || receipt.SuccessorID != admitted.SuccessorID || receipt.EffectClass != admitted.EffectClass {
		return EffectFailed, "run or effect identity mismatch"
	}
	if !canonicalSHA256(receipt.ExpectedScrubbedDigest) || !canonicalSHA256(receipt.ObservedScrubbedDigest) || receipt.ExpectedScrubbedDigest != admitted.ExpectedScrubbedDigest {
		return EffectFailed, "invalid or mismatched digest binding"
	}
	if receipt.ExpectedScrubbedDigest != receipt.ObservedScrubbedDigest {
		return EffectFailed, "observed digest mismatch"
	}
	if receipt.Reconciliation == EffectDiverged || receipt.State == EffectFailed {
		return EffectFailed, "effect readback failed"
	}
	if receipt.Reconciliation != EffectReconciled || receipt.State != EffectVerified {
		return EffectUnknown, "effect not authoritatively reconciled"
	}
	return EffectVerified, ""
}

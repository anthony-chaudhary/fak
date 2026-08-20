package modelsetreceipt

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	ExpectationSchema = "fak.model-set-startup-expectation/1"
	ReceiptSchema     = "fak.model-set-startup-receipt/1"
)

type Status string

const (
	StatusCompatible   Status = "compatible"
	StatusIncompatible Status = "incompatible"
)

type RoleStatus string

const (
	RoleCompatible   RoleStatus = "compatible"
	RoleIncompatible RoleStatus = "incompatible"
)

type Code string

const (
	CodeAsOfRequired           Code = "MODEL_SET_RECEIPT_AS_OF_REQUIRED"
	CodeRequirementsInvalid    Code = "MODEL_SET_RECEIPT_REQUIREMENTS_INVALID"
	CodeResolutionInvalid      Code = "MODEL_SET_RECEIPT_RESOLUTION_INVALID"
	CodeInventoryInvalid       Code = "MODEL_SET_RECEIPT_INVENTORY_INVALID"
	CodeRequirementsDigest     Code = "MODEL_SET_RECEIPT_REQUIREMENTS_DIGEST_MISMATCH"
	CodeResolutionDigest       Code = "MODEL_SET_RECEIPT_RESOLUTION_DIGEST_MISMATCH"
	CodeInventoryDigest        Code = "MODEL_SET_RECEIPT_INVENTORY_DIGEST_MISMATCH"
	CodeRoleMissing            Code = "MODEL_SET_RECEIPT_ROLE_MISSING"
	CodeSelectionMismatch      Code = "MODEL_SET_RECEIPT_SELECTION_MISMATCH"
	CodeInventoryEntryMissing  Code = "MODEL_SET_RECEIPT_INVENTORY_ENTRY_MISSING"
	CodeIdentityMismatch       Code = "MODEL_SET_RECEIPT_IDENTITY_MISMATCH"
	CodeEvidenceMismatch       Code = "MODEL_SET_RECEIPT_EVIDENCE_MISMATCH"
	CodeFactSetMismatch        Code = "MODEL_SET_RECEIPT_FACT_SET_MISMATCH"
	CodeEvidenceStale          Code = "MODEL_SET_RECEIPT_EVIDENCE_STALE"
	CodeRuntimeMismatch        Code = "MODEL_SET_RECEIPT_RUNTIME_MISMATCH"
	CodeRequiredFactUnknown    Code = "MODEL_SET_RECEIPT_REQUIRED_FACT_UNKNOWN"
	CodeRequiredFactMismatch   Code = "MODEL_SET_RECEIPT_REQUIRED_FACT_MISMATCH"
	CodeRequiredRoleUnresolved Code = "MODEL_SET_RECEIPT_REQUIRED_ROLE_UNRESOLVED"
	CodeCredentialMaterial     Code = "MODEL_SET_RECEIPT_CREDENTIAL_MATERIAL"
)

// Digests binds each independently canonicalized model-set input.
type Digests struct {
	Requirements string `json:"requirements"`
	Resolution   string `json:"resolution"`
	Inventory    string `json:"inventory"`
}

// Expectation is the narrow projection a model-set lock persists for startup
// attestation. It contains no artifact bytes, endpoint credentials, or secrets.
type Expectation struct {
	Schema  string        `json:"schema"`
	Digests Digests       `json:"digests"`
	Roles   []RoleBinding `json:"roles"`
}

// RoleBinding binds a role-local selection to immutable identity and the exact
// evidence that made it eligible when resolution ran.
type RoleBinding struct {
	RoleID         string `json:"role_id"`
	Required       bool   `json:"required"`
	AlternativeID  string `json:"alternative_id"`
	CandidateID    string `json:"candidate_id"`
	IdentityDigest string `json:"identity_digest"`
	EvidenceDigest string `json:"evidence_digest"`
	FactSetDigest  string `json:"fact_set_digest"`
}

// Selection is the fresh resolver's role-local startup outcome.
type Selection struct {
	AlternativeID string `json:"alternative_id"`
	CandidateID   string `json:"candidate_id"`
}

// FactBinding records credential-free evidence references while values
// remain content-bound rather than copied into startup logs.
type FactBinding struct {
	Name        string        `json:"name"`
	ValueDigest string        `json:"value_digest"`
	Witnesses   []EvidenceRef `json:"witnesses"`
}

type EvidenceRef struct {
	Kind       string `json:"kind"`
	Source     string `json:"source"`
	ObservedAt string `json:"observed_at"`
	ExpiresAt  string `json:"expires_at"`
}

type RoleReceipt struct {
	RoleID       string        `json:"role_id"`
	Required     bool          `json:"required"`
	Status       RoleStatus    `json:"status"`
	Expected     *RoleBinding  `json:"expected,omitempty"`
	Observed     *RoleBinding  `json:"observed,omitempty"`
	Reevaluated  *Selection    `json:"reevaluated_selection,omitempty"`
	FactBindings []FactBinding `json:"fact_bindings"`
}

// Failure is one stable, actionable startup refusal. SourceCode preserves the
// originating inventory or resolver code without requiring string parsing.
type Failure struct {
	Code           Code   `json:"code"`
	SourceCode     string `json:"source_code,omitempty"`
	RoleID         string `json:"role_id,omitempty"`
	CandidateID    string `json:"candidate_id,omitempty"`
	Field          string `json:"field"`
	Expected       string `json:"expected,omitempty"`
	Actual         string `json:"actual,omitempty"`
	EvidenceSource string `json:"evidence_source,omitempty"`
	Remediation    string `json:"remediation"`
}

// Receipt is the immutable startup decision artifact. Expected contains the
// lock projection; Observed contains hashes independently recomputed now.
type Receipt struct {
	Schema      string        `json:"schema"`
	EvaluatedAt string        `json:"evaluated_at"`
	Status      Status        `json:"status"`
	Expected    Digests       `json:"expected"`
	Observed    Digests       `json:"observed"`
	Roles       []RoleReceipt `json:"roles"`
	Failures    []Failure     `json:"failures"`
}

// ValidationError means an expectation or receipt is structurally unsafe to
// consume. Problems are sorted and stable for machine-independent diagnostics.
type ValidationError struct {
	Problems []string
}

func (e *ValidationError) Error() string {
	if e == nil || len(e.Problems) == 0 {
		return "model-set receipt validation failed"
	}
	return "model-set receipt validation failed: " + strings.Join(e.Problems, "; ")
}

func validationError(problems ...string) error {
	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	unique := problems[:0]
	for _, problem := range problems {
		if len(unique) == 0 || unique[len(unique)-1] != problem {
			unique = append(unique, problem)
		}
	}
	return &ValidationError{Problems: append([]string(nil), unique...)}
}

// IncompatibleError makes a receipt's nonzero startup outcome explicit while
// preserving the complete typed receipt for independent read-back.
type IncompatibleError struct {
	Failures []Failure
}

func (e *IncompatibleError) Error() string {
	if e == nil || len(e.Failures) == 0 {
		return "model-set startup incompatible"
	}
	return fmt.Sprintf("model-set startup incompatible: %d failure(s)", len(e.Failures))
}

// FailureList returns a defensive copy of typed incompatibilities.
func FailureList(err error) []Failure {
	var failed *IncompatibleError
	if !errors.As(err, &failed) || failed == nil {
		return nil
	}
	return append([]Failure(nil), failed.Failures...)
}

package modelsetresolve

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
)

// Schema identifies the in-memory resolver result contract.
const Schema = "fak.model-set-resolution/1"

// RoleStatus makes required failure and optional absence distinct outcomes.
type RoleStatus string

const (
	StatusSelected           RoleStatus = "selected"
	StatusRequiredUnresolved RoleStatus = "required-unresolved"
	StatusOptionalUnresolved RoleStatus = "optional-unresolved"
)

// RejectionCode is a stable machine-readable incompatibility class.
type RejectionCode string

const (
	CodeUnavailable         RejectionCode = "MODEL_SET_INVENTORY_ENTRY_UNAVAILABLE"
	CodeFactUnknown         RejectionCode = "MODEL_SET_FACT_UNKNOWN"
	CodeFactType            RejectionCode = "MODEL_SET_FACT_TYPE_MISMATCH"
	CodeFamily              RejectionCode = "MODEL_SET_FAMILY_MISMATCH"
	CodeQuantization        RejectionCode = "MODEL_SET_QUANTIZATION_MISMATCH"
	CodeToolCalling         RejectionCode = "MODEL_SET_TOOL_CALLING_REQUIRED"
	CodeStructuredOutput    RejectionCode = "MODEL_SET_STRUCTURED_OUTPUT_REQUIRED"
	CodeToolProtocol        RejectionCode = "MODEL_SET_TOOL_PROTOCOL_MISMATCH"
	CodeInputTokens         RejectionCode = "MODEL_SET_INPUT_TOKENS_INSUFFICIENT"
	CodeModality            RejectionCode = "MODEL_SET_MODALITY_MISSING"
	CodeRuntime             RejectionCode = "MODEL_SET_RUNTIME_MISMATCH"
	CodeServingProtocol     RejectionCode = "MODEL_SET_SERVING_PROTOCOL_MISMATCH"
	CodePlatform            RejectionCode = "MODEL_SET_PLATFORM_MISMATCH"
	CodeAccelerator         RejectionCode = "MODEL_SET_ACCELERATOR_MISMATCH"
	CodeMemory              RejectionCode = "MODEL_SET_MEMORY_EXCEEDED"
	CodeLocality            RejectionCode = "MODEL_SET_LOCALITY_MISMATCH"
	CodePrivacy             RejectionCode = "MODEL_SET_PRIVACY_MISMATCH"
	CodeLicense             RejectionCode = "MODEL_SET_LICENSE_REJECTED"
	CodeEvidenceStale       RejectionCode = "MODEL_SET_EVIDENCE_STALE"
	CodeEvidenceKindMissing RejectionCode = "MODEL_SET_EVIDENCE_KIND_MISSING"
)

// Selection identifies the winning role-local alternative and inventory
// candidate. The inventory remains the authority for immutable identity.
type Selection struct {
	AlternativeID string `json:"alternative_id"`
	CandidateID   string `json:"candidate_id"`
}

// Rejection records one hard reason a candidate failed one alternative.
type Rejection struct {
	RoleID           string        `json:"role_id"`
	AlternativeID    string        `json:"alternative_id"`
	AlternativeIndex int           `json:"alternative_index"`
	CandidateID      string        `json:"candidate_id"`
	Code             RejectionCode `json:"code"`
	Constraint       string        `json:"constraint"`
	Expected         string        `json:"expected"`
	Actual           string        `json:"actual"`
	EvidenceSource   string        `json:"evidence_source,omitempty"`
	Remediation      string        `json:"remediation"`
}

// RoleResolution is the complete deterministic outcome for one intent role.
type RoleResolution struct {
	RoleID     string      `json:"role_id"`
	Required   bool        `json:"required"`
	Status     RoleStatus  `json:"status"`
	Selection  *Selection  `json:"selection,omitempty"`
	Rejections []Rejection `json:"rejections"`
}

// Resolution contains role outcomes in stable role-ID order.
type Resolution struct {
	Schema      string           `json:"schema"`
	EvaluatedAt string           `json:"evaluated_at"`
	Roles       []RoleResolution `json:"roles"`
}

// Rejections returns a defensive, globally sorted copy of all role reasons.
func (r Resolution) Rejections() []Rejection {
	var out []Rejection
	for _, role := range r.Roles {
		out = append(out, role.Rejections...)
	}
	sortRejections(out)
	return out
}

// InputError preserves the source packages' typed validation diagnostics.
type InputError struct {
	IntentDiagnostics    []harnessmodelset.Diagnostic
	InventoryDiagnostics modelinventory.Diagnostics
}

func (e *InputError) Error() string {
	if e == nil {
		return "model-set resolution input rejected"
	}
	parts := make([]string, 0, 2)
	if len(e.IntentDiagnostics) != 0 {
		parts = append(parts, fmt.Sprintf("intent diagnostics=%d", len(e.IntentDiagnostics)))
	}
	if len(e.InventoryDiagnostics) != 0 {
		parts = append(parts, fmt.Sprintf("inventory diagnostics=%d", len(e.InventoryDiagnostics)))
	}
	return "model-set resolution input rejected: " + strings.Join(parts, ", ")
}

// RequiredRolesError reports only required roles; their detailed reasons stay
// in the returned Resolution so callers never need to parse error text.
type RequiredRolesError struct {
	RoleIDs []string
}

func (e *RequiredRolesError) Error() string {
	if e == nil || len(e.RoleIDs) == 0 {
		return "required model-set role unresolved"
	}
	return "required model-set roles unresolved: " + strings.Join(e.RoleIDs, ", ")
}

func sortRejections(rejections []Rejection) {
	sort.SliceStable(rejections, func(i, j int) bool {
		a, b := rejections[i], rejections[j]
		if a.RoleID != b.RoleID {
			return a.RoleID < b.RoleID
		}
		if a.AlternativeIndex != b.AlternativeIndex {
			return a.AlternativeIndex < b.AlternativeIndex
		}
		if a.CandidateID != b.CandidateID {
			return a.CandidateID < b.CandidateID
		}
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.Constraint != b.Constraint {
			return a.Constraint < b.Constraint
		}
		if a.EvidenceSource != b.EvidenceSource {
			return a.EvidenceSource < b.EvidenceSource
		}
		if a.Expected != b.Expected {
			return a.Expected < b.Expected
		}
		if a.Actual != b.Actual {
			return a.Actual < b.Actual
		}
		return a.Remediation < b.Remediation
	})
}

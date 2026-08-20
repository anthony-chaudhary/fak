package harnessmodelset

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// DiagnosticCode is a stable machine-readable refusal class.
type DiagnosticCode string

const (
	CodeJSONInvalid          DiagnosticCode = "HARNESS_MODEL_SET_JSON_INVALID"
	CodeJSONTrailing         DiagnosticCode = "HARNESS_MODEL_SET_JSON_TRAILING"
	CodeFieldUnknown         DiagnosticCode = "HARNESS_MODEL_SET_FIELD_UNKNOWN"
	CodeFieldDuplicate       DiagnosticCode = "HARNESS_MODEL_SET_FIELD_DUPLICATE"
	CodeFieldRequired        DiagnosticCode = "HARNESS_MODEL_SET_FIELD_REQUIRED"
	CodeValueInvalid         DiagnosticCode = "HARNESS_MODEL_SET_VALUE_INVALID"
	CodeIntentVersionUnknown DiagnosticCode = "HARNESS_MODEL_SET_INTENT_VERSION_UNKNOWN"
	CodeRolesEmpty           DiagnosticCode = "HARNESS_MODEL_SET_ROLES_EMPTY"
	CodeRoleDuplicate        DiagnosticCode = "HARNESS_MODEL_SET_ROLE_DUPLICATE"
	CodeAlternativesEmpty    DiagnosticCode = "HARNESS_MODEL_SET_ALTERNATIVES_EMPTY"
	CodeAlternativeDuplicate DiagnosticCode = "HARNESS_MODEL_SET_ALTERNATIVE_DUPLICATE"
	CodeConstraintsEmpty     DiagnosticCode = "HARNESS_MODEL_SET_CONSTRAINTS_EMPTY"
	CodeConstraintAmbiguous  DiagnosticCode = "HARNESS_MODEL_SET_CONSTRAINT_AMBIGUOUS"
	CodeConstraintConflict   DiagnosticCode = "HARNESS_MODEL_SET_CONSTRAINT_CONFLICT"
	CodeValueDuplicate       DiagnosticCode = "HARNESS_MODEL_SET_VALUE_DUPLICATE"
	CodeFreshnessInvalid     DiagnosticCode = "HARNESS_MODEL_SET_FRESHNESS_INVALID"
)

// Diagnostic identifies one invalid field and a deterministic repair.
type Diagnostic struct {
	Code        DiagnosticCode `json:"code"`
	Path        string         `json:"path"`
	Message     string         `json:"message"`
	Remediation string         `json:"remediation"`
}

// DiagnosticReport is the stable JSON envelope for fail-closed diagnostics.
type DiagnosticReport struct {
	Schema      string       `json:"schema"`
	Diagnostics []Diagnostic `json:"diagnostics"`
}

// ValidationError carries sorted typed diagnostics. Callers should branch on
// Diagnostic.Code rather than parsing Error's human-readable rendering.
type ValidationError struct {
	Diagnostics []Diagnostic
}

func (e *ValidationError) Error() string {
	if e == nil || len(e.Diagnostics) == 0 {
		return "harness model-set intent rejected"
	}
	parts := make([]string, 0, len(e.Diagnostics))
	for _, d := range e.Diagnostics {
		parts = append(parts, fmt.Sprintf("%s at %s: %s", d.Code, d.Path, d.Message))
	}
	return "harness model-set intent rejected: " + strings.Join(parts, "; ")
}

// Report returns a copy of the stable diagnostic envelope.
func (e *ValidationError) Report() DiagnosticReport {
	if e == nil {
		return DiagnosticReport{Schema: DiagnosticsSchemaV1, Diagnostics: []Diagnostic{}}
	}
	diagnostics := append([]Diagnostic(nil), e.Diagnostics...)
	return DiagnosticReport{Schema: DiagnosticsSchemaV1, Diagnostics: diagnostics}
}

// CanonicalJSON renders a byte-stable diagnostic report with a trailing newline.
func (e *ValidationError) CanonicalJSON() []byte {
	raw, _ := json.MarshalIndent(e.Report(), "", "  ")
	return append(raw, '\n')
}

// Diagnostics returns a defensive copy of typed validation diagnostics.
func Diagnostics(err error) []Diagnostic {
	var validationErr *ValidationError
	if !errors.As(err, &validationErr) {
		return nil
	}
	return append([]Diagnostic(nil), validationErr.Diagnostics...)
}

func validationError(diagnostics ...Diagnostic) error {
	if len(diagnostics) == 0 {
		return nil
	}
	sort.SliceStable(diagnostics, func(i, j int) bool {
		if diagnostics[i].Path != diagnostics[j].Path {
			return diagnostics[i].Path < diagnostics[j].Path
		}
		if diagnostics[i].Code != diagnostics[j].Code {
			return diagnostics[i].Code < diagnostics[j].Code
		}
		if diagnostics[i].Message != diagnostics[j].Message {
			return diagnostics[i].Message < diagnostics[j].Message
		}
		return diagnostics[i].Remediation < diagnostics[j].Remediation
	})
	unique := diagnostics[:0]
	for _, diagnostic := range diagnostics {
		if len(unique) == 0 || unique[len(unique)-1] != diagnostic {
			unique = append(unique, diagnostic)
		}
	}
	return &ValidationError{Diagnostics: append([]Diagnostic(nil), unique...)}
}

func diagnostic(code DiagnosticCode, path, message, remediation string) Diagnostic {
	return Diagnostic{Code: code, Path: path, Message: message, Remediation: remediation}
}

package modelsetlock

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Code is a stable machine-readable lock failure class.
type Code string

const (
	CodeInputInvalid            Code = "MODEL_SET_LOCK_INPUT_INVALID"
	CodeResolutionMismatch      Code = "MODEL_SET_LOCK_RESOLUTION_MISMATCH"
	CodeSelectedIdentityMissing Code = "MODEL_SET_LOCK_SELECTED_IDENTITY_MISSING"
	CodeSchemaUnknown           Code = "MODEL_SET_LOCK_SCHEMA_UNKNOWN"
	CodeMalformed               Code = "MODEL_SET_LOCK_MALFORMED"
	CodeDigestMismatch          Code = "MODEL_SET_LOCK_DIGEST_MISMATCH"
	CodeNonCanonical            Code = "MODEL_SET_LOCK_NON_CANONICAL"
	CodeIO                      Code = "MODEL_SET_LOCK_IO"
)

// Failure identifies one refused field and a deterministic repair.
type Failure struct {
	Code        Code   `json:"code"`
	Field       string `json:"field"`
	Detail      string `json:"detail"`
	Remediation string `json:"remediation"`
}

// Error carries sorted typed failures. Callers should branch on Failure.Code.
type Error struct {
	Failures []Failure
}

func (e *Error) Error() string {
	if e == nil || len(e.Failures) == 0 {
		return "model-set lock rejected"
	}
	parts := make([]string, 0, len(e.Failures))
	for _, item := range e.Failures {
		parts = append(parts, fmt.Sprintf("%s at %s: %s", item.Code, item.Field, item.Detail))
	}
	return "model-set lock rejected: " + strings.Join(parts, "; ")
}

// Failures returns a defensive copy of typed lock failures.
func Failures(err error) []Failure {
	var lockErr *Error
	if !errors.As(err, &lockErr) {
		return nil
	}
	return append([]Failure(nil), lockErr.Failures...)
}

// HasCode reports whether err contains the stable failure code.
func HasCode(err error, code Code) bool {
	for _, item := range Failures(err) {
		if item.Code == code {
			return true
		}
	}
	return false
}

func lockError(items ...Failure) error {
	if len(items) == 0 {
		return nil
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].Field != items[j].Field {
			return items[i].Field < items[j].Field
		}
		if items[i].Code != items[j].Code {
			return items[i].Code < items[j].Code
		}
		if items[i].Detail != items[j].Detail {
			return items[i].Detail < items[j].Detail
		}
		return items[i].Remediation < items[j].Remediation
	})
	unique := items[:0]
	for _, item := range items {
		if len(unique) == 0 || unique[len(unique)-1] != item {
			unique = append(unique, item)
		}
	}
	return &Error{Failures: append([]Failure(nil), unique...)}
}

func failure(code Code, field, detail, remediation string) Failure {
	return Failure{Code: code, Field: field, Detail: detail, Remediation: remediation}
}

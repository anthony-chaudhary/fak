package vdso

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
	"unicode"
)

const (
	// MetaProducerDiagnostics carries the canonical replay-safe producer diagnostic receipt.
	// Producers may attach it to a fresh result; tier-2 returns the same receipt on hits.
	MetaProducerDiagnostics    = "producer_diagnostics"
	producerDiagnosticsVersion = 1
)

// ProducerDiagnostic is the narrow diagnostic shape that tier-2 may replay.
// Only deterministic, scrubbed warnings are admitted; arbitrary logs and side effects are not.
type ProducerDiagnostic struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Message  string `json:"message"`
}

type producerDiagnosticReceipt struct {
	Version     int                  `json:"version"`
	Diagnostics []ProducerDiagnostic `json:"diagnostics"`
}

var (
	diagnosticCodeRE   = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,63}$`)
	diagnosticSecretRE = regexp.MustCompile(`(?i)(authorization\s*:\s*(?:bearer|basic)|(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)\s*[:=])`)
	diagnosticTimeRE   = regexp.MustCompile(`(?i)(\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}|\b(?:timestamp|request[_ -]?id|trace[_ -]?id|nonce)\s*[:=]|\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b)`)
)

// ProducerDiagnosticReceipt returns the deterministic receipt used for a fresh result
// and all later cache hits. Invalid, secret-bearing, or nondeterministic diagnostics fail closed.
func ProducerDiagnosticReceipt(diagnostics ...ProducerDiagnostic) (string, error) {
	if len(diagnostics) == 0 {
		return "", nil
	}
	for _, diagnostic := range diagnostics {
		if err := validateProducerDiagnostic(diagnostic); err != nil {
			return "", err
		}
	}
	encoded, err := json.Marshal(producerDiagnosticReceipt{Version: producerDiagnosticsVersion, Diagnostics: diagnostics})
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func canonicalProducerDiagnostics(raw string) (string, error) {
	if raw == "" {
		return "", nil
	}
	var receipt producerDiagnosticReceipt
	if err := json.Unmarshal([]byte(raw), &receipt); err != nil {
		return "", errors.New("vdso: invalid producer diagnostic receipt")
	}
	if receipt.Version != producerDiagnosticsVersion || len(receipt.Diagnostics) == 0 {
		return "", errors.New("vdso: incompatible producer diagnostic receipt")
	}
	return ProducerDiagnosticReceipt(receipt.Diagnostics...)
}

func validateProducerDiagnostic(diagnostic ProducerDiagnostic) error {
	if diagnostic.Severity != "warning" {
		return errors.New("vdso: only producer warnings are replay-safe")
	}
	if !diagnosticCodeRE.MatchString(diagnostic.Code) {
		return errors.New("vdso: invalid producer diagnostic code")
	}
	message := diagnostic.Message
	if message == "" || strings.TrimSpace(message) != message || len(message) > 512 {
		return errors.New("vdso: invalid producer diagnostic message")
	}
	for _, r := range message {
		if unicode.IsControl(r) {
			return errors.New("vdso: producer diagnostic contains control characters")
		}
	}
	if diagnosticSecretRE.MatchString(message) {
		return errors.New("vdso: producer diagnostic may contain a secret")
	}
	if diagnosticTimeRE.MatchString(message) {
		return errors.New("vdso: producer diagnostic is nondeterministic")
	}
	return nil
}

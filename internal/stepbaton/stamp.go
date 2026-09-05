// Package stepbaton provides durable, cross-process persistence for
// step-advice decisions captured before trace rotation during session restarts.
package stepbaton

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Schema identifies the versioned wire format for step-advice stamps.
const Schema = "fak.stepadvice.stamp.v1"

// Step-advice vocabulary constants mirroring gateway step classes
// to maintain dependency independence while preserving classifications.
const (
	StepAny        = "any"
	StepBounded    = "bounded"
	StepCheckpoint = "checkpoint"
	StepRebuild    = "rebuild"
	StepUnknown    = "unknown"
)

// ValidStepClass reports whether the class is recognized in the step vocabulary.
func ValidStepClass(s string) bool {
	switch s {
	case StepAny, StepBounded, StepCheckpoint, StepRebuild, StepUnknown:
		return true
	default:
		return false
	}
}

// NormalizeStepClass trims whitespace and maps unclassified input to StepUnknown.
func NormalizeStepClass(s string) string {
	s = strings.TrimSpace(s)
	if ValidStepClass(s) {
		return s
	}
	return StepUnknown
}

// Stamp persists managed-context execution decisions across process boundaries.
type Stamp struct {
	Schema         string `json:"schema"`
	TraceID        string `json:"trace_id,omitempty"`
	StepClass      string `json:"step_class"`
	Basis          string `json:"basis,omitempty"`
	Reason         string `json:"reason,omitempty"`
	ResidentTokens int    `json:"resident_tokens,omitempty"`
	BudgetTokens   int    `json:"budget_tokens,omitempty"`
	Phase          string `json:"phase,omitempty"`
	CapturedAtSHA  string `json:"captured_at_sha,omitempty"`
}

// New constructs a validated Stamp with normalized step classification.
func New(traceID, stepClass, basis, reason, phase, capturedAtSHA string, residentTokens, budgetTokens int) Stamp {
	return Stamp{
		Schema:         Schema,
		TraceID:        strings.TrimSpace(traceID),
		StepClass:      NormalizeStepClass(stepClass),
		Basis:          strings.TrimSpace(basis),
		Reason:         strings.TrimSpace(reason),
		ResidentTokens: residentTokens,
		BudgetTokens:   budgetTokens,
		Phase:          strings.TrimSpace(phase),
		CapturedAtSHA:  strings.TrimSpace(capturedAtSHA),
	}
}

// ShouldCarry reports whether the advice requires explicit injection upon resumption.
func (s Stamp) ShouldCarry() bool {
	return s.StepClass == StepCheckpoint || s.StepClass == StepRebuild
}

// Line formats the stamp into a concise single-line context hint.
func (s Stamp) Line() string {
	var b strings.Builder
	fmt.Fprintf(&b, "managed-context carryover: last live step=%s", s.StepClass)
	if s.Basis != "" {
		fmt.Fprintf(&b, " basis=%s", s.Basis)
	}
	if s.BudgetTokens > 0 {
		fmt.Fprintf(&b, " resident=%d budget=%d", s.ResidentTokens, s.BudgetTokens)
	}
	if s.Phase != "" {
		fmt.Fprintf(&b, " phase=%s", s.Phase)
	}
	if s.CapturedAtSHA != "" {
		fmt.Fprintf(&b, " at=%s", s.CapturedAtSHA)
	}
	if s.Reason != "" {
		fmt.Fprintf(&b, " reason=%q", s.Reason)
	}
	return b.String()
}

// Marshal serializes the stamp into indented JSON with a trailing newline.
func Marshal(s Stamp) ([]byte, error) {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// Unmarshal decodes a stamp from raw JSON bytes.
func Unmarshal(data []byte) (Stamp, error) {
	var s Stamp
	if err := json.Unmarshal(data, &s); err != nil {
		return Stamp{}, err
	}
	return s, nil
}

// Path computes the sanitized filesystem path for a session stamp file.
func Path(dir, sessionID string) string {
	return filepath.Join(dir, "stepadvice-"+sanitizeSegment(sessionID)+".json")
}

func sanitizeSegment(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	// A segment of all-dots (".", "..") would still be path-significant after the
	// per-rune pass (dots are allowed), so fold those to a safe literal.
	if strings.Trim(out, ".") == "" {
		return "unknown"
	}
	return out
}

// Write atomically persists the stamp to disk via sibling temporary file swap.
func Write(path string, s Stamp) error {
	data, err := Marshal(s)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename below succeeds
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// Read retrieves the stamp at path, returning false when the file does not exist.
func Read(path string) (Stamp, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Stamp{}, false, nil
		}
		return Stamp{}, false, err
	}
	s, err := Unmarshal(data)
	if err != nil {
		return Stamp{}, false, fmt.Errorf("stepbaton: read %s: %w", path, err)
	}
	return s, true, nil
}

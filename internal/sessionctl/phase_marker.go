package sessionctl

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

// WorkflowPhase names the active execution phase of an agent workflow (#11052).
type WorkflowPhase string

const (
	// PhaseRedRepro locks implementation edits until a reproduction test exists.
	PhaseRedRepro WorkflowPhase = "RED_REPRODUCTION"
	// PhaseGreenImpl permits minimal implementation changes to pass the reproduction test.
	PhaseGreenImpl WorkflowPhase = "GREEN_IMPLEMENTATION"
	// PhaseVerify checks test suite results and records proof artifacts.
	PhaseVerify WorkflowPhase = "VERIFY_PHASE"
	// PhaseDone signals the task is complete and ready for evidence reporting.
	PhaseDone WorkflowPhase = "PHASE_DONE"
)

const (
	mandateRedRepro  = "Write a test reproducing the defect. Implementation edits are locked."
	mandateGreenImpl = "Make the minimal implementation changes to pass the reproduction test."
	mandateVerify    = "Verify tests pass and capture proof artifacts."
	mandateDone      = "Task complete. Commit and report evidence."
)

var (
	// ErrInvalidPhase indicates a phase value that is not recognized.
	ErrInvalidPhase = errors.New("phase_marker: invalid workflow phase")
	// ErrInvalidTransition indicates an illegal phase change.
	ErrInvalidTransition = errors.New("phase_marker: invalid phase transition")
)

// Valid reports whether p is one of the recognized standard workflow phases.
func (p WorkflowPhase) Valid() bool {
	switch p {
	case PhaseRedRepro, PhaseGreenImpl, PhaseVerify, PhaseDone:
		return true
	default:
		return false
	}
}

// Mandate returns the prescriptive mandate string for the workflow phase.
func (p WorkflowPhase) Mandate() string {
	switch p {
	case PhaseRedRepro:
		return mandateRedRepro
	case PhaseGreenImpl:
		return mandateGreenImpl
	case PhaseVerify:
		return mandateVerify
	case PhaseDone:
		return mandateDone
	default:
		return ""
	}
}

const (
	markerPrefix = "[Workflow Phase: "
	markerSep    = ". Mandate: "
	markerSuffix = "]"
)

// FormatPhaseMarker emits the standardized phase marker for broadcasting.
func FormatPhaseMarker(phase WorkflowPhase) string {
	return FormatPhaseMarkerWithMandate(phase, phase.Mandate())
}

// FormatPhaseMarkerWithMandate emits a phase marker with an explicit mandate text.
func FormatPhaseMarkerWithMandate(phase WorkflowPhase, mandate string) string {
	return fmt.Sprintf("%s%s%s%s%s", markerPrefix, phase, markerSep, mandate, markerSuffix)
}

// ParsePhaseMarker extracts the workflow phase and mandate from a marker string.
func ParsePhaseMarker(marker string) (WorkflowPhase, string, bool) {
	trimmed := strings.TrimSpace(marker)
	if idx := strings.Index(trimmed, markerPrefix); idx >= 0 {
		rest := trimmed[idx+len(markerPrefix):]
		endIdx := strings.Index(rest, markerSuffix)
		if endIdx >= 0 {
			inner := rest[:endIdx]
			sepIdx := strings.Index(inner, markerSep)
			if sepIdx >= 0 {
				p := strings.TrimSpace(inner[:sepIdx])
				m := strings.TrimSpace(inner[sepIdx+len(markerSep):])
				if p != "" {
					return WorkflowPhase(p), m, true
				}
			}
		}
	}

	const rawPrefix = "Workflow Phase: "
	if idx := strings.Index(trimmed, rawPrefix); idx >= 0 {
		rest := trimmed[idx+len(rawPrefix):]
		if endIdx := strings.Index(rest, markerSuffix); endIdx >= 0 {
			rest = rest[:endIdx]
		}
		sepIdx := strings.Index(rest, markerSep)
		if sepIdx >= 0 {
			p := strings.TrimSpace(rest[:sepIdx])
			m := strings.TrimSpace(rest[sepIdx+len(markerSep):])
			if p != "" {
				return WorkflowPhase(p), m, true
			}
		}
	}

	return "", "", false
}

// IsValidTransition reports whether moving from current to next is permitted.
func IsValidTransition(current, next WorkflowPhase) bool {
	if !next.Valid() {
		return false
	}
	if current == "" {
		return true
	}
	if !current.Valid() {
		return false
	}
	if current == next {
		return true
	}
	switch current {
	case PhaseRedRepro:
		return next == PhaseGreenImpl
	case PhaseGreenImpl:
		return next == PhaseVerify || next == PhaseRedRepro
	case PhaseVerify:
		return next == PhaseDone || next == PhaseGreenImpl || next == PhaseRedRepro
	case PhaseDone:
		return next == PhaseRedRepro
	default:
		return false
	}
}

// TransitionPhase evaluates a proposed phase change, returning the resulting phase,
// its formatted marker, and an error if the transition is disallowed.
func TransitionPhase(current, next WorkflowPhase) (WorkflowPhase, string, error) {
	if !next.Valid() {
		return current, "", fmt.Errorf("%w: %q", ErrInvalidPhase, next)
	}
	if current != "" && !current.Valid() {
		return current, "", fmt.Errorf("%w: %q", ErrInvalidPhase, current)
	}
	if !IsValidTransition(current, next) {
		return current, "", fmt.Errorf("%w: cannot transition from %q to %q", ErrInvalidTransition, current, next)
	}
	return next, FormatPhaseMarker(next), nil
}

// PhaseMarkerTracker maintains the active workflow phase and records transition history.
type PhaseMarkerTracker struct {
	mu      sync.RWMutex
	current WorkflowPhase
	history []WorkflowPhase
}

// NewPhaseMarkerTracker initializes a tracker with an initial phase.
func NewPhaseMarkerTracker(initial WorkflowPhase) *PhaseMarkerTracker {
	t := &PhaseMarkerTracker{}
	if initial.Valid() {
		t.current = initial
		t.history = []WorkflowPhase{initial}
	}
	return t
}

// Current returns the active phase.
func (t *PhaseMarkerTracker) Current() WorkflowPhase {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.current
}

// Marker returns the formatted marker for the active phase, or empty if unset.
func (t *PhaseMarkerTracker) Marker() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.current == "" {
		return ""
	}
	return FormatPhaseMarker(t.current)
}

// Mandate returns the prescriptive mandate string for the active phase.
func (t *PhaseMarkerTracker) Mandate() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.current.Mandate()
}

// Transition advances or updates the workflow phase, appending to history on success.
func (t *PhaseMarkerTracker) Transition(next WorkflowPhase) (WorkflowPhase, string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	resultingPhase, marker, err := TransitionPhase(t.current, next)
	if err != nil {
		return t.current, "", err
	}
	t.current = resultingPhase
	t.history = append(t.history, resultingPhase)
	return resultingPhase, marker, nil
}

// History returns an isolated snapshot of all visited phases.
func (t *PhaseMarkerTracker) History() []WorkflowPhase {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]WorkflowPhase, len(t.history))
	copy(out, t.history)
	return out
}

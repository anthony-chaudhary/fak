package lifecycle

import (
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"sync"
)

// TDDPhase represents the phase of a test-driven development state machine.
type TDDPhase string

const (
	PhaseRed      TDDPhase = "PHASE_RED_REPRO"
	PhaseGreen    TDDPhase = "PHASE_GREEN_FIX"
	PhaseVerified TDDPhase = "PHASE_VERIFIED"
)

// String returns the string representation of the TDDPhase.
func (p TDDPhase) String() string {
	return string(p)
}

// TDDStateMachine manages transitions between red reproduction and green fix phases.
type TDDStateMachine struct {
	mu                          sync.RWMutex
	reproTestPath               string
	currentPhase                TDDPhase
	reproductionFailureEvidence string
	fixSuccessEvidence          string
}

// NewTDDStateMachine initializes a new state machine starting in PhaseRed.
func NewTDDStateMachine(reproTestPath string) *TDDStateMachine {
	return &TDDStateMachine{
		reproTestPath: strings.TrimSpace(reproTestPath),
		currentPhase:  PhaseRed,
	}
}

// CurrentPhase returns the active TDDPhase.
func (sm *TDDStateMachine) CurrentPhase() TDDPhase {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.currentPhase
}

// CanModifyFile evaluates whether filePath may be modified under the current phase:
//   - In PhaseRed: test files permitted (allowed=true), non-test files blocked with
//     "Phase Red requires adding a failing reproduction test before editing code."
//   - In PhaseGreen: non-test files permitted, reproduction test file frozen with
//     "Reproduction test is frozen during Phase Green fix."
//   - In PhaseVerified: task complete.
func (sm *TDDStateMachine) CanModifyFile(filePath string) (bool, string) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	switch sm.currentPhase {
	case PhaseRed:
		if IsTestFile(filePath) {
			return true, ""
		}
		return false, "Phase Red requires adding a failing reproduction test before editing code."

	case PhaseGreen:
		if sm.isReproductionTestFile(filePath) {
			return false, "Reproduction test is frozen during Phase Green fix."
		}
		return true, ""

	case PhaseVerified:
		return false, "Task complete."

	default:
		return false, "Unknown phase."
	}
}

// RecordReproductionFailure validates non-empty failure evidence and advances the state
// machine from PhaseRed to PhaseGreen. If testPath is provided, it updates the tracked
// reproduction test file.
func (sm *TDDStateMachine) RecordReproductionFailure(testPath string, failureEvidence string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentPhase != PhaseRed {
		return fmt.Errorf("cannot record reproduction failure in phase %s: must be in %s", sm.currentPhase, PhaseRed)
	}

	if strings.TrimSpace(failureEvidence) == "" {
		return errors.New("failure evidence cannot be empty")
	}

	if trimmed := strings.TrimSpace(testPath); trimmed != "" {
		sm.reproTestPath = trimmed
	}

	sm.reproductionFailureEvidence = failureEvidence
	sm.currentPhase = PhaseGreen
	return nil
}

// RecordFixSuccess validates PhaseGreen and non-empty success evidence, then advances
// the state machine to PhaseVerified.
func (sm *TDDStateMachine) RecordFixSuccess(successEvidence string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentPhase != PhaseGreen {
		return fmt.Errorf("cannot record fix success in phase %s: must be in %s", sm.currentPhase, PhaseGreen)
	}

	if strings.TrimSpace(successEvidence) == "" {
		return errors.New("success evidence cannot be empty")
	}

	sm.fixSuccessEvidence = successEvidence
	sm.currentPhase = PhaseVerified
	return nil
}

// isReproductionTestFile checks whether filePath corresponds to the designated reproduction test.
func (sm *TDDStateMachine) isReproductionTestFile(filePath string) bool {
	if sm.reproTestPath == "" {
		return IsTestFile(filePath)
	}

	normFile := normalizeTestPath(filePath)
	normRepro := normalizeTestPath(sm.reproTestPath)
	if normFile == "" || normRepro == "" {
		return false
	}

	if normFile == normRepro {
		return true
	}

	if strings.HasSuffix(normFile, "/"+normRepro) || strings.HasSuffix(normRepro, "/"+normFile) {
		return true
	}

	if !strings.Contains(normRepro, "/") && path.Base(normFile) == normRepro {
		return true
	}

	return false
}

func normalizeTestPath(p string) string {
	trimmed := strings.TrimSpace(p)
	if trimmed == "" {
		return ""
	}
	slashed := filepath.ToSlash(trimmed)
	slashed = strings.TrimPrefix(slashed, "./")
	return path.Clean(slashed)
}

// IsTestFile reports whether filePath represents a test file across common programming
// languages and directory conventions.
func IsTestFile(filePath string) bool {
	norm := normalizeTestPath(filePath)
	if norm == "" || norm == "." || norm == "/" {
		return false
	}

	parts := strings.Split(norm, "/")
	for i, part := range parts {
		if i < len(parts)-1 {
			low := strings.ToLower(part)
			if low == "tests" || low == "testdata" || low == "__tests__" {
				return true
			}
		}
	}

	base := path.Base(norm)

	// Go test files
	if strings.HasSuffix(base, "_test.go") {
		return true
	}

	// Python test files
	if strings.HasSuffix(base, ".py") {
		if strings.HasPrefix(base, "test_") || strings.HasSuffix(base, "_test.py") {
			return true
		}
	}

	// JavaScript / TypeScript test and spec files
	if strings.HasSuffix(base, ".test.ts") || strings.HasSuffix(base, ".spec.ts") ||
		strings.HasSuffix(base, ".test.tsx") || strings.HasSuffix(base, ".spec.tsx") ||
		strings.HasSuffix(base, ".test.js") || strings.HasSuffix(base, ".spec.js") ||
		strings.HasSuffix(base, ".test.jsx") || strings.HasSuffix(base, ".spec.jsx") ||
		strings.HasSuffix(base, ".test.mjs") || strings.HasSuffix(base, ".spec.mjs") ||
		strings.HasSuffix(base, ".test.cjs") || strings.HasSuffix(base, ".spec.cjs") {
		return true
	}

	// Java / Kotlin test files
	if strings.HasSuffix(base, "Test.java") || strings.HasSuffix(base, "Tests.java") ||
		strings.HasSuffix(base, "TestCase.java") ||
		strings.HasSuffix(base, "Test.kt") || strings.HasSuffix(base, "Tests.kt") {
		return true
	}

	// Ruby test files
	if strings.HasSuffix(base, "_spec.rb") || (strings.HasPrefix(base, "test_") && strings.HasSuffix(base, ".rb")) {
		return true
	}

	// Rust test files
	if strings.HasSuffix(base, "_test.rs") {
		return true
	}

	// C / C++ test files
	if strings.HasSuffix(base, "_test.cc") || strings.HasSuffix(base, "_test.cpp") ||
		strings.HasSuffix(base, "_unittest.cc") || strings.HasSuffix(base, "_unittest.cpp") ||
		strings.HasSuffix(base, "_test.c") {
		return true
	}

	return false
}

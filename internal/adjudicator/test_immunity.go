package adjudicator

import (
	"path"
	"strings"
)

// Reason constants for test immunity phase evaluations.
const (
	ReasonImplementationLockedInRedPhase = "IMPLEMENTATION_LOCKED_IN_RED_PHASE"
	ReasonTestImmuneInGreenPhase         = "TEST_IMMUNE_IN_GREEN_PHASE"
)

// Affordance constants for test immunity phase evaluations.
const (
	AffordanceImplementationLockedInRedPhase = "In Phase Red, write a test reproducing the defect before editing implementation code."
	AffordanceTestImmuneInGreenPhase         = "Reproduction test is frozen during green phase. Modify implementation files to pass the test."
)

// IsTestPath detects whether filePath represents a test file:
// - Go test files (_test.go)
// - TypeScript / JavaScript test files (.test.ts, .spec.ts, .test.tsx, .spec.tsx, .test.js, .spec.js, etc.)
// - Python test files (test_*.py, *_test.py)
// - Java test files (*Test.java)
// - Files or paths located inside tests/ or testdata/ directories
func IsTestPath(filePath string) bool {
	trimmed := strings.TrimSpace(filePath)
	if trimmed == "" {
		return false
	}

	normalized := strings.ReplaceAll(trimmed, "\\", "/")
	clean := path.Clean(normalized)
	if clean == "." || clean == "/" || clean == "" {
		return false
	}

	parts := strings.Split(clean, "/")
	for _, part := range parts {
		low := strings.ToLower(part)
		if low == "tests" || low == "testdata" {
			return true
		}
	}

	base := path.Base(clean)

	// Go test files
	if strings.HasSuffix(base, "_test.go") {
		return true
	}

	// TypeScript / JavaScript test files
	if strings.HasSuffix(base, ".test.ts") || strings.HasSuffix(base, ".spec.ts") ||
		strings.HasSuffix(base, ".test.tsx") || strings.HasSuffix(base, ".spec.tsx") ||
		strings.HasSuffix(base, ".test.js") || strings.HasSuffix(base, ".spec.js") ||
		strings.HasSuffix(base, ".test.jsx") || strings.HasSuffix(base, ".spec.jsx") {
		return true
	}

	// Python test files
	if strings.HasSuffix(base, ".py") {
		if strings.HasPrefix(base, "test_") || strings.HasSuffix(base, "_test.py") {
			return true
		}
	}

	// Java test files
	if strings.HasSuffix(base, "Test.java") || strings.HasSuffix(base, "Tests.java") {
		return true
	}

	return false
}

func isRedPhase(workflowPhase string) bool {
	switch strings.ToLower(strings.TrimSpace(workflowPhase)) {
	case "red_reproduction", "phase_red", "repro_authoring":
		return true
	default:
		return false
	}
}

func isGreenPhase(workflowPhase string) bool {
	switch strings.ToLower(strings.TrimSpace(workflowPhase)) {
	case "green_implementation", "phase_green", "fix_implementation":
		return true
	default:
		return false
	}
}

// CheckTestImmunityPhase evaluates whether filePath can be modified during workflowPhase.
func CheckTestImmunityPhase(filePath string, workflowPhase string) (allowed bool, reason string, affordance string) {
	isTest := IsTestPath(filePath)

	if isRedPhase(workflowPhase) {
		if isTest {
			return true, "", ""
		}
		return false, ReasonImplementationLockedInRedPhase, AffordanceImplementationLockedInRedPhase
	}

	if isGreenPhase(workflowPhase) {
		if !isTest {
			return true, "", ""
		}
		return false, ReasonTestImmuneInGreenPhase, AffordanceTestImmuneInGreenPhase
	}

	return true, "", ""
}

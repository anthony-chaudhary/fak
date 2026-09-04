package mtpeval

import (
	"testing"
)

// Invariant: MTP evaluation code validation must correctly verify python syntax and block balance.
// Guard: ValidatePythonCode refuses unmatched delimiters or invalid python function structures.

func TestMTPEvalLifecycle(t *testing.T) {
	t.Parallel()

	validCode := `def add(a, b):
    return a + b
`
	ok, reason := ValidatePythonCode(validCode)
	if !ok {
		t.Fatalf("expected valid python code, failed with: %s", reason)
	}
}

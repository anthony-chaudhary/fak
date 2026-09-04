package quantroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// Invariant: Quantization route selection must preserve priority candidate ordering without silent format conversion.
// Guard: Select returns CodeNoCompatibleTarget when no candidate provides compatible execution.

func TestQuantRouteLifecycle(t *testing.T) {
	t.Parallel()

	req := artifact()
	candidates := []Candidate{
		{Provider: "fallback", Runtime: runtime("fallback", []quantmeta.Format{quantmeta.Format("gguf")}, nil, nil), Hardware: "cpu"},
	}

	result := Select(req, candidates)
	if result.Code != CodeSelected || result.Candidate == nil {
		t.Fatalf("expected CodeSelected with non-nil candidate, got %+v", result)
	}
}

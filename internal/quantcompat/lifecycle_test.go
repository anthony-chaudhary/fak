package quantcompat

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// Invariant: Quantization compatibility adjudication must accurately evaluate runtime formats and hardware execution targets.
// Guard: Adjudicate rejects invalid artifacts or missing runtimes fail-closed.

func TestQuantCompatLifecycle(t *testing.T) {
	t.Parallel()

	q4 := descriptor(quantmeta.Format("gguf"), "groupwise")
	req := Request{
		Artifact: q4,
		Runtime:  Runtime{"llama.cpp", []quantmeta.Format{quantmeta.Format("gguf")}, []string{"groupwise"}, []string{"cpu"}, nil, nil},
		Hardware: "cpu",
	}

	res := Adjudicate(req)
	if res.Status != StatusDirect || res.Reason != ReasonCompatible {
		t.Fatalf("expected StatusDirect with ReasonCompatible, got %+v", res)
	}
}

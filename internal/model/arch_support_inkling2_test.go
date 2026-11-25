package model

import (
	"errors"
	"strings"
	"testing"
)

func TestInkling2LoopRefuseBackendRelaxesOnlyCUDA(t *testing.T) {
	if err := RefuseInkling2LoopBackend("cuda"); err != nil {
		t.Fatalf("cuda refused: %v", err)
	}
	for backend, want := range map[string]string{"": "cpu", "cpu": "cpu", "vulkan": "vulkan", "metal": "metal", "rocm": "rocm"} {
		err := RefuseInkling2LoopBackend(backend)
		var refusal *UnsupportedInkling2LoopExecutionError
		if !errors.As(err, &refusal) || refusal.Backend != want {
			t.Fatalf("%q: refusal = %v, want typed refusal naming %q", backend, err, want)
		}
		msg := err.Error()
		if !strings.Contains(msg, "the "+want+" backend") || !strings.Contains(msg, "--backend cuda") || strings.Contains(msg, "not implemented") {
			t.Fatalf("%q: stale or unhelpful refusal: %s", backend, msg)
		}
	}
}

func TestInkling2LoopGenericArchSupportStillRefuses(t *testing.T) {
	_, err := ClassifyForwardPath(Config{ModelType: inkling2LoopEnvelopeType}, nil)
	var refusal *UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &refusal) || refusal.Backend != "" {
		t.Fatalf("generic forward = %v, want the backend-free refusal", err)
	}
	if msg := err.Error(); !strings.Contains(msg, "generic native forward") || strings.Contains(msg, "not implemented") {
		t.Fatalf("generic refusal message: %s", msg)
	}
}

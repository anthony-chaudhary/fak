package extensionfault

import (
	"context"
	"errors"
	"testing"
)

// Invariant: Extension fault isolation must gracefully quarantine failing plugins without affecting peer extensions.
// Guard: Call returns ErrUnavailable for quarantined or unregistered extensions.

func TestExtensionFaultLifecycle(t *testing.T) {
	t.Parallel()

	s, err := New()
	if err != nil {
		t.Fatalf("failed creating supervisor: %v", err)
	}
	defer s.Close()

	_, err = s.Call(context.Background(), "unregistered-extension", "test-payload")
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected ErrUnavailable for unregistered extension, got %v", err)
	}
}

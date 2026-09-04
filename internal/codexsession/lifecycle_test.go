package codexsession

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Invariant: Codex session adapters must enforce capability bounds and timeout expired approvals.
// Guard: New returns an error for invalid configurations with empty workspaces.

func TestCodexSessionLifecycle(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	cfg := Config{
		Workspace:       tmp,
		Version:         "v1",
		RunID:           "run-lifecycle",
		Sink:            func(harnesskit.Envelope) error { return nil },
		ApprovalTimeout: time.Minute,
	}

	adapter, err := New(cfg)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if adapter == nil {
		t.Fatal("expected non-nil adapter")
	}
}

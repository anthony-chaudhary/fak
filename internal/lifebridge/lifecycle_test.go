package lifebridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// Invariant: Lifebridge state conversions must map cleanly across session and loop state vocabularies.
// Guard: RunToLoop converts running states and returns false for unsupported state transitions.

func TestLifebridgeLifecycle(t *testing.T) {
	t.Parallel()

	ls, ok := RunToLoop(session.Running)
	if !ok || ls != loopmgr.StateRunning {
		t.Fatalf("expected StateRunning, got %s (ok=%v)", ls, ok)
	}

	rs, ok := LoopToRun(loopmgr.StateRunning)
	if !ok || rs != session.Running {
		t.Fatalf("expected Running, got %v (ok=%v)", rs, ok)
	}
}

package serverlifecycle

import (
	"testing"
	"time"
)

func TestOperatingEnvelopeDefaults(t *testing.T) {
	if defaultReadinessTimeout < 300*time.Second {
		t.Fatalf("readiness deadline = %s, want at least 300s", defaultReadinessTimeout)
	}
	states := []State{StateConfigured, StateStarting, StateReady, StateStale, StateFailed, StateStopped}
	if len(states) < 6 {
		t.Fatalf("lifecycle states = %d, want at least 6", len(states))
	}
	seen := make(map[State]bool, len(states))
	for _, state := range states {
		if state == "" || seen[state] {
			t.Fatalf("state vocabulary is not distinct: %v", states)
		}
		seen[state] = true
	}
}

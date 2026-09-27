package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/observer"
)

func TestRunObserver(t *testing.T) {
	// A real Register has no undo: it would route every later trace-less result in this
	// package through one shared observer session (TestSnapshotQueryAnswersRealSessionImage
	// then saw its benign page quarantined as step_churn). Observe the call instead.
	registered := 0
	orig := registerObserverScreen
	registerObserverScreen = func(screen *observer.ObserverSemanticScreen) {
		if screen == nil {
			t.Error("runObserver registered a nil screen")
		}
		registered++
	}
	t.Cleanup(func() { registerObserverScreen = orig })

	var stdout, stderr bytes.Buffer
	rc := runObserver(&stdout, &stderr, []string{"--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if registered != 1 {
		t.Fatalf("screen registrations = %d, want exactly 1", registered)
	}
	if !strings.Contains(stdout.String(), `"registered": true`) {
		t.Errorf("expected registered in json output, got: %s", stdout.String())
	}
}

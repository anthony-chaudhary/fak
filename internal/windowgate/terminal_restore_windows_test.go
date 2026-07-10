package windowgate

import (
	"testing"
	"time"
)

func TestStartTerminalRestorePulsePinsTargetWindow(t *testing.T) {
	origTerminal := resolveTerminalWindow
	origRestore := restoreResolvedTerminalWindow
	defer func() { resolveTerminalWindow = origTerminal; restoreResolvedTerminalWindow = origRestore }()
	var resolves int
	resolveTerminalWindow = func() uintptr { resolves++; return 42 }
	restored := make(chan uintptr, 4)
	restoreResolvedTerminalWindow = func(hwnd uintptr) bool { restored <- hwnd; return true }
	StartTerminalRestorePulse(35*time.Millisecond, 10*time.Millisecond)
	deadline := time.After(200 * time.Millisecond)
	for len(restored) < 2 {
		select {
		case <-restored:
		case <-deadline:
			t.Fatal("pulse did not restore twice")
		}
	}
	if resolves != 1 {
		t.Fatalf("terminal resolved %d times, want once", resolves)
	}
	for len(restored) > 0 {
		if got := <-restored; got != 42 {
			t.Fatalf("restored hwnd %d, want 42", got)
		}
	}
}

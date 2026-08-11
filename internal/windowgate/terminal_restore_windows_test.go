package windowgate

import (
	"testing"
	"time"
)

func TestStartTerminalRestorePulseRepairsOnlyMinimizedWindow(t *testing.T) {
	origTerminal := resolveTerminalWindow
	origRestore := restoreResolvedTerminalWindow
	origIconic := isResolvedTerminalWindowIconic
	defer func() {
		resolveTerminalWindow = origTerminal
		restoreResolvedTerminalWindow = origRestore
		isResolvedTerminalWindowIconic = origIconic
	}()

	var resolves int
	resolveTerminalWindow = func() uintptr { resolves++; return 42 }
	iconic := make(chan bool, 4)
	iconic <- true
	iconic <- false
	iconic <- false
	isResolvedTerminalWindowIconic = func(hwnd uintptr) bool {
		if hwnd != 42 {
			t.Fatalf("iconic check hwnd = %d, want 42", hwnd)
		}
		select {
		case value := <-iconic:
			return value
		default:
			return false
		}
	}
	restored := make(chan uintptr, 4)
	restoreResolvedTerminalWindow = func(hwnd uintptr) bool { restored <- hwnd; return true }

	StartTerminalRestorePulse(35*time.Millisecond, 10*time.Millisecond)
	select {
	case got := <-restored:
		if got != 42 {
			t.Fatalf("restored hwnd %d, want 42", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pulse did not repair minimized terminal")
	}
	time.Sleep(60 * time.Millisecond)
	select {
	case got := <-restored:
		t.Fatalf("visible terminal was re-foregrounded (hwnd %d), stealing operator focus", got)
	default:
	}
	if resolves != 1 {
		t.Fatalf("terminal resolved %d times, want once", resolves)
	}
}

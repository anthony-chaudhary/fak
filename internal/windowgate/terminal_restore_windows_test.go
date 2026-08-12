package windowgate

import (
	"testing"
	"time"
)

func stubTerminalRestore(t *testing.T, iconic func(uintptr) bool, restore func(uintptr) bool) {
	t.Helper()
	origTerminal := resolveTerminalWindow
	origRestore := restoreResolvedTerminalWindow
	origIconic := isResolvedTerminalWindowIconic
	t.Cleanup(func() {
		resolveTerminalWindow = origTerminal
		restoreResolvedTerminalWindow = origRestore
		isResolvedTerminalWindowIconic = origIconic
	})
	resolveTerminalWindow = func() uintptr { return 42 }
	isResolvedTerminalWindowIconic = iconic
	restoreResolvedTerminalWindow = restore
}

func TestStartTerminalRestorePulseRepairsLaunchMinimizedWindow(t *testing.T) {
	var restored []uintptr
	stubTerminalRestore(t, func(uintptr) bool { return true }, func(hwnd uintptr) bool {
		restored = append(restored, hwnd)
		return true
	})
	StartTerminalRestorePulse(8*time.Second, 500*time.Millisecond)
	if len(restored) != 1 || restored[0] != 42 {
		t.Fatalf("restored = %v, want [42]", restored)
	}
}

func TestStartTerminalRestorePulsePreservesLaterUserMinimize(t *testing.T) {
	checks := 0
	restored := make(chan uintptr, 1)
	stubTerminalRestore(t, func(uintptr) bool {
		checks++
		return checks > 1
	}, func(hwnd uintptr) bool {
		restored <- hwnd
		return true
	})
	StartTerminalRestorePulse(35*time.Millisecond, 10*time.Millisecond)
	time.Sleep(60 * time.Millisecond)
	select {
	case hwnd := <-restored:
		t.Fatalf("user-minimized terminal was restored (hwnd %d)", hwnd)
	default:
	}
	if checks != 1 {
		t.Fatalf("terminal minimized state checked %d times, want one launch-boundary sample", checks)
	}
}

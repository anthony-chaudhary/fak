package windowgate

import "testing"

func stubTerminalRestore(t *testing.T, resolve func() uintptr, iconic func(uintptr) bool, restore func(uintptr) bool) {
	t.Helper()
	origTerminal := resolveTerminalWindow
	origRestore := restoreResolvedTerminalWindow
	origIconic := isResolvedTerminalWindowIconic
	t.Cleanup(func() {
		resolveTerminalWindow = origTerminal
		restoreResolvedTerminalWindow = origRestore
		isResolvedTerminalWindowIconic = origIconic
	})
	resolveTerminalWindow = resolve
	isResolvedTerminalWindowIconic = iconic
	restoreResolvedTerminalWindow = restore
}

func TestTerminalRestoreRepairsPinnedWindowAfterChildStart(t *testing.T) {
	resolveCalls := 0
	iconic := false
	var checked, restored []uintptr
	stubTerminalRestore(t, func() uintptr {
		resolveCalls++
		return 42
	}, func(hwnd uintptr) bool {
		checked = append(checked, hwnd)
		return iconic
	}, func(hwnd uintptr) bool {
		restored = append(restored, hwnd)
		return true
	})

	repair := CaptureTerminalRestore()
	if len(checked) != 0 || len(restored) != 0 {
		t.Fatalf("capture sampled or restored window: checked=%v restored=%v", checked, restored)
	}

	// Simulate the exact desktop failure: the attended window is visible at
	// capture, then managed child startup changes that pinned HWND to iconic.
	iconic = true
	if !repair.RepairAfterStart() {
		t.Fatal("post-start repair returned false")
	}
	if resolveCalls != 1 {
		t.Fatalf("terminal resolved %d times, want exactly once before start", resolveCalls)
	}
	if len(checked) != 1 || checked[0] != 42 {
		t.Fatalf("iconic checks = %v, want pinned HWND [42]", checked)
	}
	if len(restored) != 1 || restored[0] != 42 {
		t.Fatalf("restored = %v, want pinned HWND [42] exactly once", restored)
	}
}

func TestTerminalRestoreLeavesVisibleAndLaterUserMinimizeUntouched(t *testing.T) {
	iconic := false
	checks := 0
	restores := 0
	stubTerminalRestore(t, func() uintptr { return 42 }, func(hwnd uintptr) bool {
		checks++
		return iconic
	}, func(hwnd uintptr) bool {
		restores++
		return true
	})

	repair := CaptureTerminalRestore()
	if repair.RepairAfterStart() {
		t.Fatal("visible terminal reported as repaired")
	}
	iconic = true // deliberate user minimize after the one launch-boundary sample
	if checks != 1 || restores != 0 {
		t.Fatalf("checks=%d restores=%d, want one check and no restore", checks, restores)
	}
}

func TestTerminalRestoreDoesNotResolveOrTouchSiblingWindow(t *testing.T) {
	next := uintptr(42)
	var checked, restored []uintptr
	stubTerminalRestore(t, func() uintptr {
		hwnd := next
		next = 99 // would be an unrelated sibling if capture re-resolved
		return hwnd
	}, func(hwnd uintptr) bool {
		checked = append(checked, hwnd)
		return true
	}, func(hwnd uintptr) bool {
		restored = append(restored, hwnd)
		return true
	})

	repair := CaptureTerminalRestore()
	repair.RepairAfterStart()
	if len(checked) != 1 || checked[0] != 42 || len(restored) != 1 || restored[0] != 42 {
		t.Fatalf("checked=%v restored=%v, want only captured HWND 42", checked, restored)
	}
}

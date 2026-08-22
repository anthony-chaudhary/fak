//go:build windows

package windowgate

import "testing"

func TestTerminalWindowByIdentityPrefersForegroundAncestorWindow(t *testing.T) {
	var enumerated []uint32
	got := terminalWindowByIdentity(
		[]uint32{101, 202, 303},
		9001,
		202,
		func(pid uint32) uintptr {
			enumerated = append(enumerated, pid)
			return 7001 // An unrelated minimized Windows Terminal window.
		},
		func() uintptr { return 8001 },
	)
	if got != 9001 {
		t.Fatalf("terminal window = %d, want foreground ancestor HWND 9001", got)
	}
	if len(enumerated) != 0 {
		t.Fatalf("enumerated process windows before using foreground identity: %v", enumerated)
	}
}

func TestTerminalWindowByIdentityFallsBackToAncestorWindow(t *testing.T) {
	got := terminalWindowByIdentity(
		[]uint32{101, 202},
		9001,
		404,
		func(pid uint32) uintptr {
			if pid == 202 {
				return 7002
			}
			return 0
		},
		func() uintptr { return 8001 },
	)
	if got != 7002 {
		t.Fatalf("terminal window = %d, want ancestor HWND 7002", got)
	}
}

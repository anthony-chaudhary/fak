//go:build windows

package processalive

import "testing"

func TestTerminalHostPIDWalksToWindowsTerminal(t *testing.T) {
	parents := map[int]int{3: 2, 2: 1, 1: 0}
	names := map[int]string{3: "fak.exe", 2: "pwsh.exe", 1: "WindowsTerminal.exe"}
	if got, ok := terminalHostPID(3, parents, names, true); !ok || got != 1 {
		t.Fatalf("got=%d ok=%v", got, ok)
	}
}

func TestTerminalHostPIDFailsClosed(t *testing.T) {
	if got, ok := terminalHostPID(3, map[int]int{3: 2, 2: 3}, map[int]string{3: "fak.exe", 2: "pwsh.exe"}, true); ok || got != 0 {
		t.Fatalf("cycle got=%d ok=%v", got, ok)
	}
	if got, ok := terminalHostPID(3, nil, nil, false); ok || got != 0 {
		t.Fatalf("snapshot failure got=%d ok=%v", got, ok)
	}
}

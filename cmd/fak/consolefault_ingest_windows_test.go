//go:build windows

package main

import (
	"strings"
	"testing"
)

func TestConsoleFaultIngestProbeIncludesWindowsTerminalRendererExit(t *testing.T) {
	if !strings.Contains(consoleFaultIngestPS, `Faulting application name:\s*WindowsTerminal\.exe`) {
		t.Fatal("Windows event probe drops Windows Terminal renderer exits before Go can classify them")
	}
}

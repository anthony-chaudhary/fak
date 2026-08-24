//go:build windows

package main

import (
	"testing"

	"golang.org/x/sys/windows"
)

func TestInfoTerminalInputModeRoutesMouseThroughVT(t *testing.T) {
	before := uint32(windows.ENABLE_MOUSE_INPUT | windows.ENABLE_QUICK_EDIT_MODE | windows.ENABLE_LINE_INPUT)
	after := infoTerminalInputMode(before)
	if after&windows.ENABLE_VIRTUAL_TERMINAL_INPUT == 0 {
		t.Fatal("VT input is disabled")
	}
	if after&windows.ENABLE_EXTENDED_FLAGS == 0 {
		t.Fatal("extended flags are disabled, so Quick Edit cannot be changed reliably")
	}
	if after&windows.ENABLE_MOUSE_INPUT != 0 {
		t.Fatal("ENABLE_MOUSE_INPUT still routes clicks to INPUT_RECORD instead of SGR bytes")
	}
	if after&windows.ENABLE_QUICK_EDIT_MODE != 0 {
		t.Fatal("Quick Edit still captures clicks for text selection")
	}
	if after&windows.ENABLE_LINE_INPUT == 0 {
		t.Fatal("unrelated console mode bits were not preserved")
	}
}

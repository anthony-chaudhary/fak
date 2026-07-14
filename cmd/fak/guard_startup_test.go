package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGuardCapabilityFloorDefaultsScratchpadRoot(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "")
	loadGuardCapabilityFloor("")
	want := filepath.Join(os.TempDir(), "claude")
	if got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"); got != want {
		t.Fatalf("scratchpad roots=%q want %q", got, want)
	}
}

func TestGuardCapabilityFloorPreservesScratchpadOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "session-scratch")
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", want)
	loadGuardCapabilityFloor("")
	if got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"); got != want {
		t.Fatalf("scratchpad override=%q want %q", got, want)
	}
}

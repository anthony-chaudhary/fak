package main

import (
	"os"
	"strings"
	"testing"
)

func TestGuardWindowsArgvCeilingInitialPromptTransport(t *testing.T) {
	prompt := strings.Repeat("prompt-payload-", 3<<10)
	original := []string{"claude.exe", "-p", prompt, "--model", "claude-opus-4-8"}
	if got := guardWindowsCommandLineUnits(original); got <= guardWindowsCommandLineLimit {
		t.Fatalf("test fixture must breach Windows argv ceiling, got %d bytes", got)
	}
	command, stdin, moved := guardPromptStdinTransportForOS(original, "windows")
	if !moved || stdin != prompt {
		t.Fatal("oversized initial prompt was not moved byte-for-byte to stdin")
	}
	if err := guardWindowsArgvPreflight(command, "windows"); err != nil {
		t.Fatalf("transported argv still breaches Windows ceiling: %v", err)
	}
	for _, arg := range command {
		if arg == prompt {
			t.Fatal("oversized initial prompt remains on argv")
		}
	}
}

func TestGuardWindowsArgvCeilingRestartSeedTransport(t *testing.T) {
	seed := strings.Repeat("restart-seed-payload-", 3<<10)
	original := []string{"claude.exe", "-p"}
	command, handback, injected := guardSeedPromptRelaunchCommand(original, "claude", seed, nil)
	if !injected || handback == "" {
		t.Fatal("oversized restart seed was not injected through the restart transport")
	}
	flagIndex := seedPromptArgIndex(command, "--append-system-prompt-file")
	if flagIndex < 0 || flagIndex+1 >= len(command) {
		t.Fatalf("restart seed did not use file transport: %#v", command)
	}
	if err := os.Remove(command[flagIndex+1]); err != nil {
		t.Fatalf("remove restart seed file: %v", err)
	}
	if err := guardWindowsArgvPreflight(command, "windows"); err != nil {
		t.Fatalf("transported argv still breaches Windows ceiling: %v", err)
	}
	for _, arg := range command {
		if arg == seed || len(arg) >= guardWindowsCommandLineLimit {
			t.Fatalf("oversized restart seed remains on argv: arg bytes=%d", len(arg))
		}
	}
}

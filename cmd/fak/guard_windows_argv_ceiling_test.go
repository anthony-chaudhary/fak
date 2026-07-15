package main

import (
	"os"
	"strings"
	"testing"
)

const guardWindowsCommandLineLimit = 32767

// guardWindowsCommandLineBytes is a conservative assembled-argv witness. It
// counts separators and the terminating NUL in addition to every argument byte;
// escaping can only add overhead, so callers keep a safety margin below the OS
// ceiling rather than treating this as an exact CreateProcess encoder.
func guardWindowsCommandLineBytes(command []string) int {
	total := 1 // terminating NUL
	for i, arg := range command {
		if i > 0 {
			total++ // separator
		}
		total += len(arg)
	}
	return total
}

func assertGuardWindowsArgvUnderLimit(t *testing.T, command []string) {
	t.Helper()
	if got := guardWindowsCommandLineBytes(command); got >= guardWindowsCommandLineLimit {
		longestIndex, longestBytes := -1, 0
		for i, arg := range command {
			if len(arg) > longestBytes {
				longestIndex, longestBytes = i, len(arg)
			}
		}
		t.Fatalf("assembled guarded child argv is %d bytes (Windows limit %d); longest arg[%d]=%d bytes", got, guardWindowsCommandLineLimit, longestIndex, longestBytes)
	}
}

func TestGuardWindowsArgvCeilingInitialPromptTransport(t *testing.T) {
	prompt := strings.Repeat("prompt-payload-", 3<<10)
	original := []string{"claude.exe", "-p", prompt, "--model", "claude-opus-4-8"}
	if got := guardWindowsCommandLineBytes(original); got <= guardWindowsCommandLineLimit {
		t.Fatalf("test fixture must breach Windows argv ceiling, got %d bytes", got)
	}
	command, stdin, moved := guardPromptStdinTransportForOS(original, "windows")
	if !moved || stdin != prompt {
		t.Fatal("oversized initial prompt was not moved byte-for-byte to stdin")
	}
	assertGuardWindowsArgvUnderLimit(t, command)
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
	assertGuardWindowsArgvUnderLimit(t, command)
	for _, arg := range command {
		if arg == seed || len(arg) >= guardWindowsCommandLineLimit {
			t.Fatalf("oversized restart seed remains on argv: arg bytes=%d", len(arg))
		}
	}
}

package main

import (
	"io"
	"os/exec"
	"strings"
	"testing"
)

func TestGuardPromptStdinTransportMovesLargeWindowsClaudePrompt(t *testing.T) {
	prompt := "unicode lambda\r\n" + strings.Repeat("p", 40<<10)
	in := []string{"claude.exe", "-p", prompt, "--model", "claude-opus-4-8", "--verbose"}
	got, stdin, moved := guardPromptStdinTransportForOS(in, "windows")
	if !moved {
		t.Fatal("large Windows Claude prompt remained on argv")
	}
	want := []string{in[0], "-p", "--model", "claude-opus-4-8", "--verbose"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("non-prompt argv changed:\n got %#v\nwant %#v", got, want)
	}
	if stdin != prompt {
		t.Fatal("stdin prompt bytes changed")
	}
}

func TestGuardPromptStdinTransportPreservesOtherLaunches(t *testing.T) {
	large := strings.Repeat("p", 40<<10)
	cases := []struct {
		os  string
		cmd []string
	}{
		{"linux", []string{"claude", "-p", large}},
		{"windows", []string{"claude", "-p", strings.Repeat("s", guardWindowsPromptStdinThreshold-1)}},
		{"windows", []string{"codex", "exec", large}},
		{"windows", []string{"claude", "--model", "claude-opus-4-8"}},
	}
	for _, tc := range cases {
		got, stdin, moved := guardPromptStdinTransportForOS(tc.cmd, tc.os)
		if moved || stdin != "" || strings.Join(got, "\x00") != strings.Join(tc.cmd, "\x00") {
			t.Fatalf("launch unexpectedly changed: os=%s cmd=%#v got=%#v", tc.os, tc.cmd, got)
		}
	}
}

func TestApplyGuardPromptStdinTransportPreservesPromptBytes(t *testing.T) {
	prompt := strings.Repeat("exact-byte-lambda", 4000)
	command := []string{"claude.exe", "-p", prompt, "--model", "claude-opus-4-8"}
	child := exec.Command("unused")
	gotArgs, moved := applyGuardPromptStdinTransport(child, command, "windows")
	if !moved {
		t.Fatal("large prompt was not transported")
	}
	got, err := io.ReadAll(child.Stdin)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != prompt {
		t.Fatal("child stdin did not preserve exact prompt bytes")
	}
	for _, arg := range gotArgs {
		if arg == prompt {
			t.Fatal("large prompt remains in transported argv")
		}
	}
}

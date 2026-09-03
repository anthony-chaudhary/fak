package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func TestGuardCodexPromptFuelProvidesThreeIdenticalLaunchReaders(t *testing.T) {
	root := t.TempDir()
	t.Setenv("DISPATCH_WORKSPACE", root)
	prompt := "issue #8124\r\nexact unicode λ\n" + strings.Repeat("fuel\x00", 1024)
	command := []string{"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "-"}

	fuel, err := preparePromptFuel(command, strings.NewReader(prompt), "guard-8124")
	if err != nil {
		t.Fatal(err)
	}
	wantPathRoot := filepath.Join(root, ".dispatch-runs", "prompt-fuel")
	if filepath.Dir(fuel.path) != wantPathRoot {
		t.Fatalf("fuel path = %q, want directory %q", fuel.path, wantPathRoot)
	}
	persisted, err := os.ReadFile(fuel.path)
	if err != nil {
		t.Fatal(err)
	}
	if string(persisted) != prompt {
		t.Fatal("persisted prompt fuel changed bytes")
	}

	meta := guardChildSpawnMetadata{
		AgentRunID:   "guard-8124",
		ToolCallID:   "guard-child:guard-8124",
		PolicyDigest: "sha256:test-policy",
		Backend:      "codex",
		Envelope: toolprocgate.CapabilityEnvelope{
			Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn},
		},
		LaunchPlan: newGuardLaunchPlan(command),
		PromptFuel: fuel,
	}
	for launch := 1; launch <= 3; launch++ {
		meta.RegistryPath = filepath.Join(root, fmt.Sprintf("child-registry-%d.jsonl", launch))
		_, child, err := launchGuardChildWithBroker(command, nil, false, meta, toolprocgate.NewSpawnBroker(), func(toolprocgate.SpawnGrant) (*exec.Cmd, error) {
			return exec.Command("unused"), nil
		})
		if err != nil {
			t.Fatalf("launch %d: %v", launch, err)
		}
		got, err := io.ReadAll(child.Stdin)
		if err != nil {
			t.Fatalf("launch %d read: %v", launch, err)
		}
		if string(got) != prompt {
			t.Fatalf("launch %d prompt bytes changed", launch)
		}
	}

	j := journal.OpenMemory()
	row := appendGuardChildExitWitness(j, "codex", "guard-8124", nil, nil, time.Now(), fuel)
	if row.ChildExit == nil || !strings.Contains(row.ChildExit.LastHook, "prompt_fuel_digest="+promptFuelDigest([]byte(prompt))) ||
		!strings.Contains(row.ChildExit.LastHook, "restart_count=2") {
		t.Fatalf("prompt fuel witness = %+v", row.ChildExit)
	}
}

func TestGuardCodexPromptFuelRefusesMissingOrTamperedBeforeLaunch(t *testing.T) {
	command := []string{"codex", "exec", "-"}
	for _, tc := range []struct {
		name   string
		mutate func(*testing.T, string)
		want   string
	}{
		{
			name: "missing",
			mutate: func(t *testing.T, path string) {
				t.Helper()
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			},
			want: promptFuelMissingReason,
		},
		{
			name: "tampered",
			mutate: func(t *testing.T, path string) {
				t.Helper()
				if err := os.WriteFile(path, []byte("replacement prompt"), 0o600); err != nil {
					t.Fatal(err)
				}
			},
			want: promptFuelTamperedReason,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fuel, err := persistPromptFuel(filepath.Join(t.TempDir(), "prompt.fuel"), []byte("original prompt"))
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(t, fuel.path)
			launcherCalled := false
			meta := guardChildSpawnMetadata{PromptFuel: fuel, LaunchPlan: newGuardLaunchPlan(command)}
			_, child, err := launchGuardChildWithBroker(command, nil, false, meta, nil, func(toolprocgate.SpawnGrant) (*exec.Cmd, error) {
				launcherCalled = true
				return exec.Command("unused"), nil
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("launch error = %v, want %s", err, tc.want)
			}
			if launcherCalled || child != nil {
				t.Fatalf("invalid fuel reached child construction: launcher_called=%v child=%v", launcherCalled, child)
			}
		})
	}
}

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

func TestGuardPromptStdinTransportMovesGuardFrontedClaudePrompt(t *testing.T) {
	prompt := strings.Repeat("resume-carry", 4000)
	in := []string{"fak.exe", "guard", "--provider", "anthropic", "--", "claude.exe", "--resume", "session", "-p", prompt, "--dangerously-skip-permissions"}
	got, stdin, moved := guardPromptStdinTransportForOS(in, "windows")
	if !moved || stdin != prompt {
		t.Fatalf("guard-fronted prompt transport = moved %v stdin bytes %d", moved, len(stdin))
	}
	for _, arg := range got {
		if arg == prompt {
			t.Fatal("guard-fronted prompt remains on argv")
		}
	}
}

func TestTUIAgentPromptTransportMovesLargeWindowsPrompt(t *testing.T) {
	prompt := strings.Repeat("tui-prompt", 4000)
	launch := []string{"claude.exe", "--permission-mode", "bypassPermissions", "-p", prompt}
	got, stdin, moved := tuiAgentPromptTransport(launch, "windows")
	if !moved || stdin != prompt {
		t.Fatalf("TUI prompt transport = moved %v stdin bytes %d", moved, len(stdin))
	}
	for _, arg := range got {
		if arg == prompt {
			t.Fatal("TUI prompt remains on argv")
		}
	}
}

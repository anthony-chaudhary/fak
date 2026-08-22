package main

import (
	"encoding/base64"
	"encoding/binary"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf16"
)

func TestGuardCodexSessionStartTrustedHashMatchesPinnedCodex(t *testing.T) {
	const command = "echo hi"
	const want = "sha256:9e87c3ad0f08b294d080bebd23a4971ebf5c8d908f22a10ac0a01510508c7917"
	if got := guardCodexSessionStartTrustedHash(command); got != want {
		t.Fatalf("trusted hash=%q, want %q", got, want)
	}
}

func TestGuardCodexPowerShellCommandRoundTrip(t *testing.T) {
	got := guardCodexPowerShellCommand([]string{`C:\Program Files\fak's\fak.exe`, "guard-sessionstart", "--state", `C:\state path`})
	const prefix = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand "
	encoded, ok := strings.CutPrefix(got, prefix)
	if !ok {
		t.Fatalf("Windows hook command=%q", got)
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("decode command: %v", err)
	}
	if len(raw)%2 != 0 {
		t.Fatalf("UTF-16LE command has odd byte length %d", len(raw))
	}
	units := make([]uint16, len(raw)/2)
	for i := range units {
		units[i] = binary.LittleEndian.Uint16(raw[i*2:])
	}
	want := `& 'C:\Program Files\fak''s\fak.exe' 'guard-sessionstart' '--state' 'C:\state path'; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE`
	if decoded := string(utf16.Decode(units)); decoded != want {
		t.Fatalf("decoded command=%q, want %q", decoded, want)
	}
}

func TestInstallGuardCodexSessionStartHookAt(t *testing.T) {
	dir := t.TempDir()
	command := []string{"codex", "--no-alt-screen"}
	got, install, err := installGuardCodexSessionStartHookAt(command, true, "fak binary", dir, "trace-8218")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.Provider != "codex" || install.StatePath != filepath.Join(dir, "codex-sessionstart-state") {
		t.Fatalf("install=%+v", install)
	}
	if got[0] != "codex" || got[len(got)-1] != "--no-alt-screen" {
		t.Fatalf("child command was not preserved around overrides: %v", got)
	}
	if len(got) != len(command)+4 || got[1] != "-c" || got[3] != "-c" {
		t.Fatalf("override shape=%v", got)
	}
	hookOverride, stateOverride := got[2], got[4]
	for _, want := range []string{"hooks.SessionStart", "guard-sessionstart", "--provider", "codex", "--managed", "--trace", "trace-8218", "--state", install.StatePath} {
		if !strings.Contains(hookOverride, want) {
			t.Errorf("hook override missing %q: %s", want, hookOverride)
		}
	}
	if !strings.Contains(stateOverride, "hooks.state") || !strings.Contains(stateOverride, guardCodexSessionStartHookKey()) || !strings.Contains(stateOverride, guardCodexSessionStartTrustedHash(guardCodexSessionStartCommand("fak binary", true, "trace-8218", install.StatePath))) {
		t.Fatalf("trust override=%s", stateOverride)
	}
	if strings.Contains(strings.Join(got, " "), "dangerously-bypass-hook-trust") {
		t.Fatalf("adapter widened trust beyond its exact handler: %v", got)
	}
}

func TestInstallGuardCodexSessionStartHookAtRejectsSessionFlagCollisions(t *testing.T) {
	for _, override := range []string{
		"hooks={}",
		"hooks.SessionStart=[]",
		"hooks.state={}",
	} {
		t.Run(override, func(t *testing.T) {
			command := []string{"codex", "-c", override, "--no-alt-screen"}
			got, install, err := installGuardCodexSessionStartHookAt(command, false, "fak", t.TempDir(), "trace")
			if err == nil {
				t.Fatalf("expected collision for %q", override)
			}
			if install.Applied || !reflect.DeepEqual(got, command) {
				t.Fatalf("collision mutated command: install=%+v got=%v", install, got)
			}
		})
	}
}

func TestClassifyGuardCodexSessionStart(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "codex-sessionstart-state")
	decision, err := classifyGuardCodexSessionStart(statePath, "startup", "thread-1")
	if err != nil || decision != (guardCodexSessionStartDecision{Bind: true}) {
		t.Fatalf("initial startup=(%+v, %v)", decision, err)
	}
	if err := writeGuardCodexSessionBinding(statePath, "thread-1", "trace-1"); err != nil {
		t.Fatalf("bind initial thread: %v", err)
	}
	decision, err = classifyGuardCodexSessionStart(statePath, "startup", "thread-1")
	if err != nil || decision != (guardCodexSessionStartDecision{Trace: "trace-1"}) {
		t.Fatalf("duplicate startup=(%+v, %v)", decision, err)
	}
	decision, err = classifyGuardCodexSessionStart(statePath, "clear", "thread-1")
	if err != nil || decision != (guardCodexSessionStartDecision{Trace: "trace-1"}) {
		t.Fatalf("duplicate clear=(%+v, %v)", decision, err)
	}
	decision, err = classifyGuardCodexSessionStart(statePath, "startup", "thread-2")
	if err != nil || decision != (guardCodexSessionStartDecision{Boundary: true, Bind: true, Source: "clear"}) {
		t.Fatalf("/new startup=(%+v, %v)", decision, err)
	}
	decision, err = classifyGuardCodexSessionStart(statePath, "clear", "thread-3")
	if err != nil || decision != (guardCodexSessionStartDecision{Boundary: true, Bind: true, Source: "clear"}) {
		t.Fatalf("/clear=(%+v, %v)", decision, err)
	}

	if err := writeGuardCodexSessionBinding(statePath, "thread-2", "trace-2"); err != nil {
		t.Fatalf("write state: %v", err)
	}
	decision, err = classifyGuardCodexSessionStart(statePath, "resume", "thread-3")
	if err != nil || decision != (guardCodexSessionStartDecision{Bind: true}) {
		t.Fatalf("non-boundary source=(%+v, %v)", decision, err)
	}
}

package main

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/power"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestKeepAwakeFlagParsing(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{input: "off", want: KeepAwakeOff},
		{input: "", want: KeepAwakeOff},
		{input: "false", want: KeepAwakeOff},
		{input: "0", want: KeepAwakeOff},
		{input: "while-active", want: KeepAwakeWhileActive},
		{input: "active", want: KeepAwakeWhileActive},
		{input: "true", want: KeepAwakeWhileActive},
		{input: "1", want: KeepAwakeWhileActive},
		{input: "always", want: KeepAwakeAlways},
		{input: "ALWAYS", want: KeepAwakeAlways},
		{input: "invalid-mode", wantErr: true},
	}

	for _, tt := range tests {
		got, err := validateKeepAwake(tt.input)
		if (err != nil) != tt.wantErr {
			t.Errorf("validateKeepAwake(%q) error = %v, wantErr = %v", tt.input, err, tt.wantErr)
			continue
		}
		if !tt.wantErr && got != tt.want {
			t.Errorf("validateKeepAwake(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestKeepAwakeServeFlag(t *testing.T) {
	fs, sf := newServeFlagSet()
	f := fs.Lookup("keep-awake")
	if f == nil {
		t.Fatal("flag --keep-awake not found on serve flag set")
	}
	if f.DefValue != KeepAwakeOff {
		t.Fatalf("expected default %q, got %q", KeepAwakeOff, f.DefValue)
	}

	if err := fs.Parse([]string{"--keep-awake=always"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *sf.keepAwake != KeepAwakeAlways {
		t.Fatalf("got %q, want %q", *sf.keepAwake, KeepAwakeAlways)
	}

	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--keep-awake=while-active"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *sf2.keepAwake != KeepAwakeWhileActive {
		t.Fatalf("got %q, want %q", *sf2.keepAwake, KeepAwakeWhileActive)
	}

	fs3, sf3 := newServeFlagSet()
	if err := fs3.Parse([]string{"--keep-awake=off"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *sf3.keepAwake != KeepAwakeOff {
		t.Fatalf("got %q, want %q", *sf3.keepAwake, KeepAwakeOff)
	}
}

func TestKeepAwakeAgentFlag(t *testing.T) {
	fs, af := newAgentFlagSet()
	f := fs.Lookup("keep-awake")
	if f == nil {
		t.Fatal("flag --keep-awake not found on agent flag set")
	}
	if f.DefValue != KeepAwakeOff {
		t.Fatalf("expected default %q, got %q", KeepAwakeOff, f.DefValue)
	}

	if err := fs.Parse([]string{"--keep-awake=always"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *af.keepAwake != KeepAwakeAlways {
		t.Fatalf("got %q, want %q", *af.keepAwake, KeepAwakeAlways)
	}

	fs2, af2 := newAgentFlagSet()
	if err := fs2.Parse([]string{"--keep-awake=while-active"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *af2.keepAwake != KeepAwakeWhileActive {
		t.Fatalf("got %q, want %q", *af2.keepAwake, KeepAwakeWhileActive)
	}

	fs3, af3 := newAgentFlagSet()
	if err := fs3.Parse([]string{"--keep-awake=off"}); err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if *af3.keepAwake != KeepAwakeOff {
		t.Fatalf("got %q, want %q", *af3.keepAwake, KeepAwakeOff)
	}
}

func TestKeepAwakeLifecycle(t *testing.T) {
	power.ResetGlobalForTesting()
	defer power.ResetGlobalForTesting()

	// Off should acquire nothing
	rOff, err := acquireProcessKeepAwake(KeepAwakeOff, "test")
	if err != nil || rOff != nil {
		t.Fatalf("expected nil releaser for off, got %v, err=%v", rOff, err)
	}
	if power.IsActive() {
		t.Fatal("expected power not active for off mode")
	}

	// Always should acquire process wake lock
	rAlways, err := acquireProcessKeepAwake(KeepAwakeAlways, "test-always")
	if err != nil || rAlways == nil {
		t.Fatalf("expected releaser for always, got %v, err=%v", rAlways, err)
	}
	if !power.IsActive() {
		t.Fatal("expected power active for always mode")
	}
	if err := rAlways.Release(); err != nil {
		t.Fatalf("Release error: %v", err)
	}
	if power.IsActive() {
		t.Fatal("expected power inactive after release")
	}

	// while-active agent run
	rAgent, err := acquireAgentRunKeepAwake(KeepAwakeWhileActive)
	if err != nil || rAgent == nil {
		t.Fatalf("expected releaser for while-active, got %v, err=%v", rAgent, err)
	}
	if !power.IsActive() {
		t.Fatal("expected power active for agent run")
	}
	if err := rAgent.Release(); err != nil {
		t.Fatalf("Release error: %v", err)
	}
	if power.IsActive() {
		t.Fatal("expected power inactive after release")
	}
}

func TestKeepAwakeActiveMonitor(t *testing.T) {
	power.ResetGlobalForTesting()
	defer power.ResetGlobalForTesting()

	tbl := session.NewTable()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stopMonitor := startKeepAwakeActiveMonitor(ctx, tbl)
	defer stopMonitor()

	if power.IsActive() {
		t.Fatal("expected power inactive with no active sessions")
	}

	// Add running session
	tbl.Restore("test-session", session.State{
		TraceID: "test-session",
		Run:     session.Running,
	})

	// Wait for monitor tick
	deadline := time.Now().Add(2 * time.Second)
	for !power.IsActive() && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if !power.IsActive() {
		t.Fatal("expected power active after running session registered")
	}

	// Stop session
	tbl.Restore("test-session", session.State{
		TraceID: "test-session",
		Run:     session.Stopped,
	})

	deadline = time.Now().Add(2 * time.Second)
	for power.IsActive() && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if power.IsActive() {
		t.Fatal("expected power inactive after session stopped")
	}
}

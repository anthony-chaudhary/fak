package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDispatchRefreshRegistryInProcess(t *testing.T) {
	old := dispatchSessionRegistry
	t.Cleanup(func() { dispatchSessionRegistry = old })

	var gotTimeout time.Duration
	dispatchSessionRegistry = func(root string, timeout time.Duration) (map[string]any, error) {
		gotTimeout = timeout
		return map[string]any{"mode": "registry", "sessions": 3, "auto_resume": 1}, nil
	}
	reg := dispatchRefreshRegistry(t.TempDir(), io.Discard)
	if reg["ok"] != true || reg["sessions"] != 3 || gotTimeout != 120*time.Second {
		t.Fatalf("refresh = %v (timeout %v), want ok with the pass summary under the 120s budget", reg, gotTimeout)
	}

	dispatchSessionRegistry = func(string, time.Duration) (map[string]any, error) {
		return nil, errors.New("fleet session registry: context deadline exceeded")
	}
	reg = dispatchRefreshRegistry(t.TempDir(), io.Discard)
	if reg["ok"] != false || reg["error"] != "fleet session registry: context deadline exceeded" {
		t.Fatalf("failed refresh = %v, want ok=false carrying the error", reg)
	}
}

// The real in-process pass: an expired budget fails without touching the registry dir, and
// a pass inside its budget writes the registry and reports the summary.
func TestDispatchSessionRegistryDeadline(t *testing.T) {
	base := t.TempDir()
	regDir := filepath.Join(base, "reg")
	for k, v := range map[string]string{
		"FLEET_USER_HOME":       filepath.Join(base, "home"),
		"FLEET_CONFIG_HOME":     filepath.Join(base, "cfg"),
		"FLEET_REG_DIR":         regDir,
		"FAK_ACCOUNTS_REGISTRY": filepath.Join(base, "no-accounts-registry.json"),
	} {
		t.Setenv(k, v)
	}
	oldLive := fleetSessionsLiveSIDs
	t.Cleanup(func() { fleetSessionsLiveSIDs = oldLive })
	fleetSessionsLiveSIDs = func() map[string]bool { return map[string]bool{} }

	if _, err := dispatchSessionRegistry(base, time.Nanosecond); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired pass err = %v, want a deadline error", err)
	}
	if _, err := os.Stat(filepath.Join(regDir, "sessions.json")); !os.IsNotExist(err) {
		t.Fatalf("expired pass wrote sessions.json (stat err %v)", err)
	}

	sum, err := dispatchSessionRegistry(base, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if sum["ok"] != true || sum["sessions"] != 0 {
		t.Fatalf("summary = %v, want ok with 0 sessions", sum)
	}
	if _, err := os.Stat(filepath.Join(regDir, "sessions.json")); err != nil {
		t.Fatalf("pass inside its budget did not write sessions.json: %v", err)
	}
}

func TestFleetSessionsProbeSelector(t *testing.T) {
	cases := map[string][]string{
		"none":    {"registry"},
		"blocked": {"registry", "--probe"},
		"stale":   {"registry", "--probe", "stale"},
		"all":     {"--probe=all"},
	}
	for want, argv := range cases {
		if got := fleetSessionsProbeSelector(argv); got != want {
			t.Errorf("probe selector %v = %q, want %q", argv, got, want)
		}
	}
	var stderr bytes.Buffer
	if code := runFleetSessions(io.Discard, &stderr, []string{"registry", "--probe"}); code != 2 {
		t.Fatalf("--probe exit = %d, want 2 (%s)", code, stderr.String())
	}
}

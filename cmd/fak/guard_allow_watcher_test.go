package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestGuardAllowWatcherLiveAddRemoveAndMalformedLastGood(t *testing.T) {
	dir := t.TempDir()
	allowPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, allowPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	reload := guardPolicyReloader("")
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	w := newGuardAllowWatcher(time.Millisecond, reload, nil)
	const tool = "live_overlay_probe"
	verdict := func() abi.VerdictKind {
		return adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, tool, map[string]any{})).Kind
	}
	if got := verdict(); got == abi.VerdictAllow {
		t.Fatalf("precondition verdict = %v, want denied", got)
	}

	if err := saveGuardAllowOverlay(allowPath, guardAllowOverlay{Allow: []string{tool}}); err != nil {
		t.Fatal(err)
	}
	if e := w.Reload(context.Background()); !e.Reloaded || e.Rejected {
		t.Fatalf("add reload = %+v", e)
	}
	if got := verdict(); got != abi.VerdictAllow {
		t.Fatalf("after add = %v, want ALLOW", got)
	}

	if err := os.WriteFile(allowPath, []byte(`{"version":"wrong","allow":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if e := w.Reload(context.Background()); !e.Rejected || e.Reloaded {
		t.Fatalf("malformed reload = %+v", e)
	}
	if got := verdict(); got != abi.VerdictAllow {
		t.Fatalf("malformed edit lost last-good allow: %v", got)
	}

	if err := saveGuardAllowOverlay(allowPath, guardAllowOverlay{}); err != nil {
		t.Fatal(err)
	}
	if e := w.Reload(context.Background()); !e.Reloaded || e.Rejected {
		t.Fatalf("remove reload = %+v", e)
	}
	if got := verdict(); got == abi.VerdictAllow {
		t.Fatalf("after remove = %v, want denied", got)
	}
}

func TestGuardAllowWatcherRunStopsWithContext(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(guardAllowOverlayEnv, filepath.Join(dir, "allow.json"))
	ctx, cancel := context.WithCancel(context.Background())
	w := newGuardAllowWatcher(time.Millisecond, guardPolicyReloader(""), nil)
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("Run error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("watcher did not stop with guard context")
	}
}

func TestGuardAllowWatcherReloadsSessionScope(t *testing.T) {
	dir := t.TempDir()
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(old)
		setGuardAllowSessionScopeID("")
		guardAllowSessionScopeQuarantined = false
	})

	setGuardAllowSessionScopeID("watcher-test")
	t.Setenv(guardAllowOverlayEnv, filepath.Join(dir, "base.allow.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	reload := guardPolicyReloader("")
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	w := newGuardAllowWatcher(time.Millisecond, reload, nil)
	const tool = "session_live_probe"
	verdict := func() abi.VerdictKind {
		return adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, tool, map[string]any{})).Kind
	}
	if got := verdict(); got == abi.VerdictAllow {
		t.Fatalf("precondition verdict = %v, want denied", got)
	}

	if err := saveGuardAllowOverlay(guardAllowSessionOverlayPath(), guardAllowOverlay{Allow: []string{tool}}); err != nil {
		t.Fatal(err)
	}
	if e := w.Reload(context.Background()); !e.Reloaded || e.Rejected {
		t.Fatalf("session add reload = %+v", e)
	}
	if got := verdict(); got != abi.VerdictAllow {
		t.Fatalf("after session add = %v, want ALLOW", got)
	}
}

func TestGuardAllowWatcherRevokesAtTTLExpiryWithoutFileChange(t *testing.T) {
	dir := t.TempDir()
	allowPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, allowPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))

	expiresAt := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	setClock := guardAllowPinClock(t, expiresAt.Add(-time.Second))
	const tool = "live_ttl_probe"
	if err := saveGuardAllowOverlay(allowPath, guardAllowOverlay{
		Allow:  []string{tool},
		Expiry: map[string]string{tool: guardAllowExpiryStamp(expiresAt)},
	}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(allowPath)
	if err != nil {
		t.Fatal(err)
	}

	reload := guardPolicyReloader("")
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	w := newGuardAllowWatcher(time.Hour, reload, nil)
	decision := func() (abi.VerdictKind, abi.ReasonCode) {
		v := adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, tool, map[string]any{}))
		return v.Kind, v.Reason
	}
	if kind, reason := decision(); kind != abi.VerdictAllow {
		t.Fatalf("before expiry verdict=(%v, %v), want ALLOW", kind, reason)
	}

	setClock(expiresAt)
	if e := w.Reload(context.Background()); !e.Reloaded || e.Rejected {
		t.Fatalf("expiry reload = %+v, want successful reload", e)
	}
	if kind, reason := decision(); kind != abi.VerdictDeny || reason != abi.ReasonDefaultDeny {
		t.Fatalf("at expiry verdict=(%v, %v), want (DENY, DEFAULT_DENY)", kind, reason)
	}
	after, err := os.ReadFile(allowPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("watcher rewrote overlay bytes across expiry:\nbefore: %s\nafter:  %s", before, after)
	}
}

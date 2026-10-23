package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// guardAllowWatcherFailClosed isolates the deny expectations from the live posture
// that the real built-in policy reloader intentionally preserves.
func guardAllowWatcherFailClosed(t *testing.T) {
	t.Helper()
	prior := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prior) })
	adjudicator.Default.SetPolicy(adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
}

func TestGuardAllowWatcherLiveAddRemoveAndMalformedLastGood(t *testing.T) {
	guardAllowWatcherFailClosed(t)
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

func TestGuardAllowWatcherRenderSuppressesSuccessfulReloads(t *testing.T) {
	var output bytes.Buffer
	w := &guardAllowWatcher{onEvent: func(e guardAllowWatchEvent) {
		if !e.Rejected {
			return
		}
		fmt.Fprintf(&output, "fak guard allow watcher: rejected; keeping last-good floor: %s\n", e.Detail)
	}}

	for range 3 {
		w.emit(guardAllowWatchEvent{Reloaded: true, Detail: "operator allow overlay reloaded"})
	}
	if output.Len() != 0 {
		t.Fatalf("successful reload diagnostics = %q, want no attended-terminal output", output.String())
	}

	w.emit(guardAllowWatchEvent{Rejected: true, Detail: "invalid overlay version"})
	got := output.String()
	if strings.Count(got, "\n") != 1 {
		t.Fatalf("rejected reload diagnostics = %q, want one line", got)
	}
	for _, want := range []string{"rejected", "keeping last-good floor", "invalid overlay version"} {
		if !strings.Contains(got, want) {
			t.Fatalf("rejected reload diagnostics = %q, want %q", got, want)
		}
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
	guardAllowWatcherFailClosed(t)
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
	guardAllowWatcherFailClosed(t)
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

// fak-test:runtime fast est=20ms
func TestGuardAllowWatcherStopJoinsReloadBeforeScopeDrop(t *testing.T) {
	scopeTestRepo(t, "watcher-stop")
	sessionPath := armGuardAllowSessionScopeTeardown()
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })
	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"session_tool"}}); err != nil {
		t.Fatal(err)
	}
	// A quarantined overlay can become removable while the session is running;
	// teardown must not clear that quarantine while a reload still reads it.
	guardAllowSessionScopeQuarantined = true
	repoPath := guardAllowOverlayPath()
	userPath, err := guardAllowWritePathForScope("user")
	if err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(userPath, guardAllowOverlay{Allow: []string{"user_tool"}}); err != nil {
		t.Fatal(err)
	}

	synctest.Test(t, func(t *testing.T) {
		entered := make(chan struct{})
		release := make(chan struct{})
		observedQuarantine := make(chan bool, 1)
		stop := startGuardAllowWatcher(context.Background(), func(context.Context) (gateway.PolicyReloadResponse, error) {
			close(entered)
			<-release
			quarantined := guardAllowSessionScopeQuarantined
			for _, layer := range guardAllowEffectiveReadLayers() {
				if layer.Name == guardAllowScopeSession {
					quarantined = false
				}
			}
			observedQuarantine <- quarantined
			return gateway.PolicyReloadResponse{}, nil
		}, false)
		released := false
		defer func() {
			if !released {
				close(release)
			}
			stop()
		}()

		// Changing a durable layer enters Reload even while session scope is quarantined.
		if err := saveGuardAllowOverlay(repoPath, guardAllowOverlay{Allow: []string{"repo_tool"}}); err != nil {
			t.Fatal(err)
		}
		durable := map[string][]byte{}
		for _, path := range []string{repoPath, userPath} {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			durable[path] = raw
		}
		<-entered
		stopped := make(chan struct{})
		go func() {
			stop()
			dropGuardAllowSessionScopeAtSessionEnd(true, nil)
			close(stopped)
		}()
		// Unlike a sleep or a select/default alone, Wait proves the stop caller has
		// reached its join while the callback remains blocked on our release channel.
		synctest.Wait()
		select {
		case <-stopped:
			t.Fatal("stop returned and dropped scope while Reload was still in flight")
		default:
		}
		if !guardAllowSessionScopeQuarantined || guardAllowSessionScopeTeardownPath != sessionPath {
			t.Fatal("scope state changed before Reload completed")
		}
		if _, err := os.Stat(sessionPath); err != nil {
			t.Fatalf("session overlay removed before Reload completed: %v", err)
		}

		close(release)
		released = true
		<-stopped
		if quarantined := <-observedQuarantine; !quarantined {
			t.Fatal("in-flight Reload observed scope after quarantine was cleared")
		}
		if guardAllowSessionScopeQuarantined || guardAllowSessionScopeTeardownPath != "" {
			t.Fatal("session scope was not disarmed after the watcher joined")
		}
		if _, err := os.Stat(sessionPath); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("session overlay survived teardown: %v", err)
		}
		for path, before := range durable {
			after, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("teardown changed durable overlay %q: %v", path, err)
			}
		}
		stop()
		stop()
	})
}

// fak-test:runtime fast est=10ms
func TestGuardAllowWatcherStopAfterParentCancellation(t *testing.T) {
	scopeTestRepo(t, "watcher-already-canceled")
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		stop := startGuardAllowWatcher(ctx, nil, false)
		stopped := make(chan struct{}, 2)
		for range 2 {
			go func() {
				stop()
				stopped <- struct{}{}
			}()
		}
		synctest.Wait()
		if len(stopped) != 2 {
			t.Fatal("concurrent stops did not join an already-canceled watcher")
		}
		stop()
	})
}

func TestGuardAllowWatcherStopIsWiredIntoTerminalCancel(t *testing.T) {
	boot, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	wiring := "\tstopAllowWatcher := startGuardAllowWatcher("
	cancelJoin := "\tcancel = func() {\n\t\tcancelContext()\n\t\tstopAllowWatcher()\n\t}"
	startAt := strings.Index(string(boot), wiring)
	joinAt := strings.Index(string(boot), cancelJoin)
	serveAt := strings.Index(string(boot), "go func() { serveErr <- srv.Serve(")
	if startAt < 0 || joinAt <= startAt || serveAt <= joinAt {
		t.Fatal("watcher stop/join must be in the shared cancel chain before gateway startup")
	}
	exit, err := os.ReadFile("guard_child_supervision.go")
	if err != nil {
		t.Fatal(err)
	}
	finishAt := bytes.Index(exit, []byte("func finishGuardChildAndReport("))
	if finishAt < 0 {
		t.Fatal("missing child terminal funnel")
	}
	terminal := string(exit[finishAt:])
	cancelAt := strings.Index(terminal, "\tcancel()")
	dropAt := strings.Index(terminal, "\tdropGuardAllowSessionScopeAtSessionEnd(")
	exitAt := strings.Index(terminal, "\t\t\tos.Exit(")
	if cancelAt < 0 || dropAt <= cancelAt || exitAt <= dropAt {
		t.Fatal("terminal funnel must synchronously cancel before dropping scope and os.Exit")
	}
}

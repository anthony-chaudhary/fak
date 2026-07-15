package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func newGuardLifecycleTestServer(t testing.TB) (*gateway.Server, *guardLifecycleServer) {
	t.Helper()
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	ipc, err := startGuardLifecycleServer(srv)
	if err != nil {
		t.Fatalf("start lifecycle IPC: %v", err)
	}
	t.Cleanup(ipc.Close)
	return srv, ipc
}

func TestGuardLifecycleIPCAuthenticatedSnapshot(t *testing.T) {
	srv, ipc := newGuardLifecycleTestServer(t)
	got, err := fetchGuardLifecycleSignals(ipc.path, ipc.token, time.Second)
	if err != nil {
		t.Fatalf("fetch lifecycle IPC: %v", err)
	}
	want := srv.LifecycleSignalsSnapshot()
	if got != want {
		t.Fatalf("snapshot mismatch: got %+v want %+v", got, want)
	}
	if _, err := fetchGuardLifecycleSignals(ipc.path, "wrong-token", time.Second); err == nil {
		t.Fatal("invalid token unexpectedly succeeded")
	}
}

func TestGuardLifecyclePreferredIPCDoesNotScrapeHTTP(t *testing.T) {
	_, ipc := newGuardLifecycleTestServer(t)
	t.Setenv(guardLifecycleSocketEnv, ipc.path)
	t.Setenv(guardLifecycleTokenEnv, ipc.token)
	var hits atomic.Int32
	httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		fmt.Fprintln(w, "fak_guard_deny_all_consecutive 99")
		fmt.Fprintln(w, "fak_harness_coherence_posture 0")
	}))
	defer httpSrv.Close()

	stop, source, err := fetchGuardStopHookSignalsPreferred(context.Background(), httpSrv.URL, time.Second)
	if err != nil || source != "ipc" {
		t.Fatalf("Stop preferred fetch = (%+v, %q, %v), want IPC success", stop, source, err)
	}
	compact, source, err := fetchGuardPreCompactSignalsPreferred(context.Background(), httpSrv.URL, time.Second)
	if err != nil || source != "ipc" || compact.posture != compactcohere.PostureBlock {
		t.Fatalf("PreCompact preferred fetch = (%+v, %q, %v), want block IPC success", compact, source, err)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("supervisor lifecycle path scraped HTTP %d times", got)
	}
}

func TestGuardLifecycleHTTPFallbackWithoutIPC(t *testing.T) {
	t.Setenv(guardLifecycleSocketEnv, "")
	t.Setenv(guardLifecycleTokenEnv, "")
	httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "fak_guard_deny_all_consecutive 2")
		fmt.Fprintln(w, "fak_guard_deny_all_same_consecutive 1")
		fmt.Fprintln(w, "fak_guard_tool_feedback_consecutive 3")
		fmt.Fprintln(w, "fak_guard_fak_verb_calls 4")
		fmt.Fprintln(w, "fak_harness_coherence_posture 0")
	}))
	defer httpSrv.Close()
	if got, source, err := fetchGuardStopHookSignalsPreferred(context.Background(), httpSrv.URL, time.Second); err != nil || source != "http" || got.DenyAllConsecutive != 2 {
		t.Fatalf("Stop HTTP fallback = (%+v, %q, %v)", got, source, err)
	}
	if got, source, err := fetchGuardPreCompactSignalsPreferred(context.Background(), httpSrv.URL, time.Second); err != nil || source != "http" || got.posture != compactcohere.PostureAllow {
		t.Fatalf("PreCompact HTTP fallback = (%+v, %q, %v)", got, source, err)
	}
}

func TestGuardLifecycleIPCFailureIsAuthoritative(t *testing.T) {
	t.Setenv(guardLifecycleSocketEnv, t.TempDir()+"/missing.sock")
	t.Setenv(guardLifecycleTokenEnv, "present")
	var hits atomic.Int32
	httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		fmt.Fprintln(w, "fak_guard_deny_all_consecutive 0")
		fmt.Fprintln(w, "fak_harness_coherence_posture 0")
	}))
	defer httpSrv.Close()
	if _, source, err := fetchGuardStopHookSignalsPreferred(context.Background(), httpSrv.URL, 50*time.Millisecond); err == nil || source != "ipc" {
		t.Fatalf("Stop forced IPC failure = source %q err %v", source, err)
	}
	if _, source, err := fetchGuardPreCompactSignalsPreferred(context.Background(), httpSrv.URL, 50*time.Millisecond); err == nil || source != "ipc" {
		t.Fatalf("PreCompact forced IPC failure = source %q err %v", source, err)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("authoritative IPC failure fell back to HTTP %d times", got)
	}
}

func BenchmarkGuardLifecycleSignalRead(b *testing.B) {
	srv, ipc := newGuardLifecycleTestServer(b)
	httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "fak_guard_deny_all_consecutive 0")
		fmt.Fprintln(w, "fak_guard_deny_all_same_consecutive 0")
		fmt.Fprintln(w, "fak_guard_tool_feedback_consecutive 0")
		fmt.Fprintln(w, "fak_guard_fak_verb_calls 0")
	}))
	defer httpSrv.Close()
	b.Run("direct-snapshot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = srv.LifecycleSignalsSnapshot()
		}
	})
	b.Run("local-ipc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := fetchGuardLifecycleSignals(ipc.path, ipc.token, time.Second); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("http-metrics", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := fetchGuardStopHookSignals(context.Background(), httpSrv.URL, time.Second); err != nil {
				b.Fatal(err)
			}
		}
	})
}

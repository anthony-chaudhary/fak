package main

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
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

func TestGuardSessionStartClearCreatesFakSessionBoundary(t *testing.T) {
	oldSessions, oldDurability := serveSessions, serveSessionDurability
	serveSessions, serveSessionDurability = session.NewTable(), nil
	t.Cleanup(func() {
		serveSessions, serveSessionDurability = oldSessions, oldDurability
	})
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	journalPath := filepath.Join(t.TempDir(), "session-journal.jsonl")
	t.Setenv(sessionjournal.EnvPath, journalPath)
	t.Setenv(guardSessionJournalEnvMode, "")
	stageGuardSessionStartWitness(t, 1, nil)

	const oldTrace = "guard-old"
	budget := session.Budget{
		TurnsLeft: 5, TokensLeft: 800, ContextTokensLeft: 100, ContextTokensCap: 1000,
		SpendMicroCentsLeft: 40, SpendMicroCentsCap: 100,
	}
	if _, ok := serveSessions.SetBudget(oldTrace, budget); !ok {
		t.Fatal("stage old session budget")
	}
	recordGuardSessionStartIdentityFor(oldTrace, "provider-thread-1", "claude", "startup")
	recordGuardSessionStartJournalFor(oldTrace, "provider-thread-1", "claude", 0)
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless", DefaultTraceID: "launch-placeholder"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	ipc, err := startGuardLifecycleServer(srv)
	if err != nil {
		t.Fatalf("start lifecycle IPC: %v", err)
	}
	t.Cleanup(ipc.Close)
	// The gateway default can advance after the lifecycle server starts (for
	// example after a context-budget recontinue); clear must close the live trace.
	srv.SetDefaultTraceID(oldTrace)
	t.Setenv(guardLifecycleSocketEnv, ipc.path)
	t.Setenv(guardLifecycleTokenEnv, ipc.token)

	payload := bytes.NewBufferString(`{"hook_event_name":"SessionStart","source":"clear","session_id":"provider-thread-2"}`)
	var stdout, stderr bytes.Buffer
	if code := runGuardSessionStartHook(&stdout, &stderr, payload, []string{"--mode", "off", "--provider", "claude", "--trace", oldTrace}); code != 0 {
		t.Fatalf("hook exit=%d stderr=%s", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("off-mode hook emitted stdout: %s", stdout.String())
	}
	newTrace := providerBoundaryTrace("claude", "provider-thread-2")
	old := serveSessions.Get(oldTrace)
	child := serveSessions.Get(newTrace)
	if old.Run != session.Stopped || old.Reason != session.ReasonProviderSessionClear {
		t.Fatalf("old session = run %s reason %q", old.Run, old.Reason)
	}
	if child.Run != session.Running || child.ProviderBoundary.PreviousTrace != oldTrace {
		t.Fatalf("child session = %+v", child)
	}
	if child.Budget.ContextTokensLeft != 1000 || child.Budget.TurnsLeft != 5 || child.Budget.SpendMicroCentsLeft != 40 {
		t.Fatalf("child budget = %+v", child.Budget)
	}
	if got := srv.DefaultTraceID(); got != newTrace {
		t.Fatalf("gateway default trace=%q, want %q", got, newTrace)
	}
	wire := toGatewaySessionState(child)
	if wire.ProviderBoundary.Schema != session.ProviderSessionBoundarySchema ||
		wire.ProviderBoundary.PreviousTrace != oldTrace {
		t.Fatalf("gateway provider boundary = %+v", wire.ProviderBoundary)
	}

	// A duplicate hook delivery for the same provider session is idempotent.
	result, err := notifyGuardProviderSessionStart(ipc.path, ipc.token, "claude", "clear", "provider-thread-2", time.Second)
	if err != nil {
		t.Fatalf("duplicate boundary: %v", err)
	}
	if result.Applied || serveSessions.Get(newTrace).Rev != child.Rev {
		t.Fatalf("duplicate mutated child: result=%+v child rev=%d want=%d", result, serveSessions.Get(newTrace).Rev, child.Rev)
	}

	// The new provider id is durably joined to the new fak trace.
	byID, _ := resume.LoadIdentity(regDir)
	if got := byID["provider-thread-2"]; got != newTrace {
		t.Fatalf("identity join=%q, want %q", got, newTrace)
	}
	journal := sessionjournal.FoldEvents(sessionjournal.LoadFile(journalPath))
	bySession := make(map[string]sessionjournal.Session, len(journal))
	for _, rec := range journal {
		bySession[rec.ID] = rec
	}
	if len(journal) != 2 || !bySession["provider-thread-1"].Closed || bySession["provider-thread-2"].Closed {
		t.Fatalf("provider journal boundary = %+v", journal)
	}
}

func TestGuardSessionStartCodexClearAndNewCreateOneBoundaryEach(t *testing.T) {
	oldSessions, oldDurability := serveSessions, serveSessionDurability
	serveSessions, serveSessionDurability = session.NewTable(), nil
	t.Cleanup(func() {
		serveSessions, serveSessionDurability = oldSessions, oldDurability
	})
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	journalPath := filepath.Join(t.TempDir(), "session-journal.jsonl")
	t.Setenv(sessionjournal.EnvPath, journalPath)
	t.Setenv(guardSessionJournalEnvMode, "")
	stageGuardSessionStartWitness(t, 1, nil)

	const oldTrace = "guard-codex-launch"
	budget := session.Budget{
		TurnsLeft: 5, TokensLeft: 800, ContextTokensLeft: 100, ContextTokensCap: 1000,
		SpendMicroCentsLeft: 40, SpendMicroCentsCap: 100, ToolCallsLeft: 7, ToolCallsCap: 9,
	}
	if _, ok := serveSessions.SetBudget(oldTrace, budget); !ok {
		t.Fatal("stage old session budget")
	}
	srv, ipc := newGuardLifecycleTestServer(t)
	srv.SetDefaultTraceID(oldTrace)
	t.Setenv(guardLifecycleSocketEnv, ipc.path)
	t.Setenv(guardLifecycleTokenEnv, ipc.token)
	statePath := filepath.Join(t.TempDir(), "codex-sessionstart-state")

	emit := func(source, sessionID string) {
		t.Helper()
		payload := bytes.NewBufferString(fmt.Sprintf(`{"hook_event_name":"SessionStart","source":%q,"session_id":%q}`, source, sessionID))
		var stdout, stderr bytes.Buffer
		args := []string{"--mode", "off", "--provider", "codex", "--trace", oldTrace, "--state", statePath}
		if code := runGuardSessionStartHook(&stdout, &stderr, payload, args); code != 0 {
			t.Fatalf("%s hook exit=%d stderr=%s", source, code, stderr.String())
		}
		if stdout.Len() != 0 || stderr.Len() != 0 {
			t.Fatalf("%s hook output stdout=%q stderr=%q", source, stdout.String(), stderr.String())
		}
	}
	assertCumulative := func(trace string) {
		t.Helper()
		got := serveSessions.Get(trace).Budget
		if got.TurnsLeft != budget.TurnsLeft || got.TokensLeft != budget.TokensLeft ||
			got.SpendMicroCentsLeft != budget.SpendMicroCentsLeft || got.ToolCallsLeft != budget.ToolCallsLeft {
			t.Fatalf("trace %q reset cumulative envelope: got %+v want cumulative %+v", trace, got, budget)
		}
		if got.ContextTokensLeft != budget.ContextTokensCap {
			t.Fatalf("trace %q context budget=%d, want fresh cap %d", trace, got.ContextTokensLeft, budget.ContextTokensCap)
		}
	}

	// First launch startup only establishes the ordered Codex binding.
	emit("startup", "codex-thread-1")
	if got := srv.DefaultTraceID(); got != oldTrace {
		t.Fatalf("initial startup changed gateway trace=%q", got)
	}
	if got := serveSessions.Get(oldTrace); got.Run != session.Running {
		t.Fatalf("initial startup changed launch session=%+v", got)
	}
	decision, err := classifyGuardCodexSessionStart(statePath, "startup", "codex-thread-1")
	if err != nil || decision.Trace != oldTrace || decision.Boundary || decision.Bind {
		t.Fatalf("initial binding=(%+v, %v)", decision, err)
	}

	// Codex emits startup again for /new. The changed id is the missing discriminator.
	emit("startup", "codex-thread-2")
	newTrace := providerBoundaryTrace("codex", "codex-thread-2")
	if got := srv.DefaultTraceID(); got != newTrace {
		t.Fatalf("/new gateway trace=%q, want %q", got, newTrace)
	}
	if got := serveSessions.Get(oldTrace); got.Run != session.Stopped || got.Reason != session.ReasonProviderSessionClear {
		t.Fatalf("/new old session=%+v", got)
	}
	if got := serveSessions.Get(newTrace); got.Run != session.Running || got.ProviderBoundary.PreviousTrace != oldTrace || got.ProviderBoundary.Source != "clear" {
		t.Fatalf("/new child session=%+v", got)
	}
	assertCumulative(newTrace)

	// Redelivery of the same startup neither rekeys again nor corrupts its identity join.
	newRev := serveSessions.Get(newTrace).Rev
	emit("startup", "codex-thread-2")
	if got := serveSessions.Get(newTrace).Rev; got != newRev {
		t.Fatalf("duplicate /new mutated child rev=%d, want %d", got, newRev)
	}
	byID, _ := resume.LoadIdentity(regDir)
	if got := byID["codex-thread-2"]; got != newTrace {
		t.Fatalf("duplicate /new identity=%q, want %q", got, newTrace)
	}

	// /clear has an explicit source and advances from the /new child.
	emit("clear", "codex-thread-3")
	clearTrace := providerBoundaryTrace("codex", "codex-thread-3")
	if got := srv.DefaultTraceID(); got != clearTrace {
		t.Fatalf("/clear gateway trace=%q, want %q", got, clearTrace)
	}
	if got := serveSessions.Get(newTrace); got.Run != session.Stopped || got.Reason != session.ReasonProviderSessionClear {
		t.Fatalf("/clear previous session=%+v", got)
	}
	if got := serveSessions.Get(clearTrace); got.Run != session.Running || got.ProviderBoundary.PreviousTrace != newTrace {
		t.Fatalf("/clear child session=%+v", got)
	}
	assertCumulative(clearTrace)
	clearRev := serveSessions.Get(clearTrace).Rev
	emit("clear", "codex-thread-3")
	if got := serveSessions.Get(clearTrace).Rev; got != clearRev {
		t.Fatalf("duplicate /clear mutated child rev=%d, want %d", got, clearRev)
	}

	decision, err = classifyGuardCodexSessionStart(statePath, "startup", "codex-thread-3")
	if err != nil || decision.Trace != clearTrace || decision.Boundary || decision.Bind {
		t.Fatalf("final binding=(%+v, %v)", decision, err)
	}
	journal := sessionjournal.FoldEvents(sessionjournal.LoadFile(journalPath))
	bySessionID := make(map[string]sessionjournal.Session, len(journal))
	for _, row := range journal {
		bySessionID[row.ID] = row
	}
	if len(journal) != 3 || !bySessionID["codex-thread-1"].Closed ||
		!bySessionID["codex-thread-2"].Closed || bySessionID["codex-thread-3"].Closed {
		t.Fatalf("Codex provider journal boundaries=%+v", journal)
	}
}

func TestParseGuardProviderSessionStartAcceptsProviderIDShapes(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		wantID  string
	}{
		{name: "session", payload: `{"source":" CLEAR ","session_id":"session-1"}`, wantID: "session-1"},
		{name: "thread", payload: `{"source":"clear","thread_id":"thread-1"}`, wantID: "thread-1"},
		{name: "conversation", payload: `{"source":"clear","conversation_id":"conversation-1"}`, wantID: "conversation-1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseGuardProviderSessionStart([]byte(tt.payload))
			if got.Source != "clear" || got.SessionID != tt.wantID {
				t.Fatalf("provider start = %+v, want clear/%s", got, tt.wantID)
			}
		})
	}
	if got := parseGuardProviderSessionStart([]byte(`{"source":`)); got != (guardProviderSessionStart{}) {
		t.Fatalf("malformed payload = %+v, want zero", got)
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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// turnkeyGatedPlanner blocks every in-flight completion until release is
// closed, so a test can hold K requests concurrently inside the server and
// observe the admission decision while they are live.
type turnkeyGatedPlanner struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func newTurnkeyGatedPlanner() *turnkeyGatedPlanner {
	return &turnkeyGatedPlanner{started: make(chan struct{}), release: make(chan struct{})}
}

func (p *turnkeyGatedPlanner) Model() string { return "local" }

func (p *turnkeyGatedPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.once.Do(func() { close(p.started) })
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

// newTurnkeyCapacityServer starts a real listener so concurrent HTTP clients can
// hold sessions open at the same time.
func newTurnkeyCapacityServer(t *testing.T, srv *turnkeyServer) (baseURL string, shutdown func()) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/v1/completions", srv.handleCompletions)
	mux.HandleFunc("/healthz", srv.handleHealthz)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv.httpServer = &http.Server{Handler: mux}
	go func() { _ = srv.httpServer.Serve(listener) }()
	return "http://" + listener.Addr().String(), func() { _ = srv.httpServer.Close() }
}

// postTurnkeyChat admits one buffered chat request and returns the raw status and
// body. Unlike postTurnkeyToolRequest it does NOT require a 200 — a shed/queued
// decision is the behavior under test.
func postTurnkeyChat(t *testing.T, url string, body any, client *http.Client) (int, http.Header, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, resp.Header, got
}

// TestTurnkeySessionAdmissionShedsAtCapacity is the #13075 reproduction: with a
// per-session KV budget that admits exactly N sessions, a K > N concurrent fan-out
// must produce a typed shed/queue decision (HTTP 429 or 503) — never an unbounded
// silent stall where every client waits. Before the fix, every request is admitted
// and blocks, so no non-200 decision is observed and the test fails.
func TestTurnkeySessionAdmissionShedsAtCapacity(t *testing.T) {
	const maxSessions = 2
	const concurrent = 4

	planner := newTurnkeyGatedPlanner()
	srv := &turnkeyServer{
		planner: planner,
		plan: macfit.TurnkeyProfile{
			ContextBudgetTokens: 8192,
			KVPoolBytes:         2 * 8192 * 2, // two 8192-token sessions at 2 bytes/token
			KVBytesPerToken:     2,
			Tier:                macfit.ModelTier{ModelID: "local"},
		},
	}
	if got := srv.capacity().MaxSessions; got != maxSessions {
		t.Fatalf("derived MaxSessions = %d, want %d", got, maxSessions)
	}

	baseURL, shutdown := newTurnkeyCapacityServer(t, srv)
	defer shutdown()

	client := &http.Client{Timeout: 10 * time.Second}
	type result struct {
		status int
		hdr    http.Header
	}
	results := make(chan result, concurrent)
	for i := 0; i < concurrent; i++ {
		go func() {
			status, hdr, _ := postTurnkeyChat(t, baseURL+"/v1/chat/completions",
				map[string]any{"model": "local", "messages": []map[string]any{{"role": "user", "content": "hold"}}}, client)
			results <- result{status: status, hdr: hdr}
		}()
	}

	// The first admitted request proves the gate opened; the remaining capacity is
	// filled once it is observed inside Complete.
	select {
	case <-planner.started:
	case <-time.After(5 * time.Second):
		t.Fatal("no request was admitted (planner never started)")
	}
	// The admitted sessions are held inside Complete, so only the SHED requests can
	// return a decision now. Collect exactly `concurrent-maxSessions` shed/queued
	// decisions; a request that blocks instead is the unbounded stall under test.
	deadline := time.After(8 * time.Second)
	sheds := 0
	shedsByStatus := map[int]int{}
	retryAfterSeen := false
	for sheds < concurrent-maxSessions {
		select {
		case res := <-results:
			shedsByStatus[res.status]++
			if res.status == http.StatusTooManyRequests || res.status == http.StatusServiceUnavailable {
				sheds++
				if ra := res.hdr.Get("Retry-After"); ra != "" {
					if secs, err := strconv.Atoi(ra); err != nil || secs < 1 {
						t.Fatalf("Retry-After = %q, want >= 1s", ra)
					}
					retryAfterSeen = true
				}
				continue
			}
			if res.status != http.StatusOK {
				t.Fatalf("unexpected status %d (not a shed 429/503 nor an admitted 200)", res.status)
			}
			// An admitted 200 returned while the planner is still gated: the server
			// admitted more than maxSessions. Fail loudly rather than hang.
			t.Fatalf("more than %d sessions admitted concurrently (got a 200 while gate held)", maxSessions)
		case <-deadline:
			t.Fatalf("only %d/%d shed decisions at capacity before deadline; shed statuses=%v (remaining clients stalled unbounded)", sheds, concurrent-maxSessions, shedsByStatus)
		}
	}
	if sheds != concurrent-maxSessions {
		t.Fatalf("sheds = %d, want %d at capacity", sheds, concurrent-maxSessions)
	}
	if !retryAfterSeen {
		t.Fatal("shed responses omitted the bounded Retry-After hint")
	}
	// A shed must carry a bounded retry hint (Retry-After) so a client can back off.
	close(planner.release)
	// Drain the admitted requests and check at least one shed carried Retry-After.
	remaining := concurrent - sheds
	for i := 0; i < remaining; i++ {
		select {
		case res := <-results:
			if res.status != http.StatusOK {
				t.Fatalf("admitted request finished with status %d, want 200", res.status)
			}
		case <-time.After(8 * time.Second):
			t.Fatal("admitted requests did not drain after release")
		}
	}
	if stats := srv.capacityStats(); stats.ShedTotal != int64(sheds) || stats.LiveSessions != 0 {
		t.Fatalf("capacity stats = %+v, want ShedTotal=%d LiveSessions=0 after drain", stats, sheds)
	}
}

// TestTurnkeyCapacityStatsSurface pins the read-only telemetry contract: live
// session count, admitted/shed totals, and KV budget utilization.
func TestTurnkeyCapacityStatsSurface(t *testing.T) {
	srv := &turnkeyServer{
		plan: macfit.TurnkeyProfile{
			ContextBudgetTokens: 8192,
			KVPoolBytes:         4 * 8192 * 2,
			KVBytesPerToken:     2,
			Tier:                macfit.ModelTier{ModelID: "local"},
		},
	}
	rec := httptest.NewRecorder()
	srv.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	var got struct {
		Sessions struct {
			MaxSessions       int     `json:"max_sessions"`
			LiveSessions      int     `json:"live_sessions"`
			AdmittedTotal     int64   `json:"admitted_total"`
			ShedTotal         int64   `json:"shed_total"`
			KVBudgetBytes     uint64  `json:"kv_budget_bytes"`
			KVInUseBytes      uint64  `json:"kv_in_use_bytes"`
			KVUtilization     float64 `json:"kv_utilization"`
			PerSessionKVBytes uint64  `json:"per_session_kv_bytes"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode healthz: %v (%s)", err, rec.Body.String())
	}
	if got.Sessions.MaxSessions != 4 {
		t.Fatalf("max_sessions = %d, want 4", got.Sessions.MaxSessions)
	}
	if got.Sessions.PerSessionKVBytes != 8192*2 {
		t.Fatalf("per_session_kv_bytes = %d, want %d", got.Sessions.PerSessionKVBytes, 8192*2)
	}
	if got.Sessions.KVUtilization != 0 {
		t.Fatalf("kv_utilization = %v, want 0 when idle", got.Sessions.KVUtilization)
	}

	// Admit one session directly to observe non-zero live/utilization.
	if !srv.beginChatRequest() {
		t.Fatal("beginChatRequest refused on an idle server")
	}
	rec = httptest.NewRecorder()
	srv.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Sessions.LiveSessions != 1 || got.Sessions.AdmittedTotal != 1 {
		t.Fatalf("after admit: %+v, want LiveSessions=1 AdmittedTotal=1", got.Sessions)
	}
	if got.Sessions.KVUtilization != 0.25 {
		t.Fatalf("kv_utilization = %v, want 0.25 for 1/4 sessions", got.Sessions.KVUtilization)
	}
	srv.endChatRequest()
}

// TestTurnkeyCapacityRaceClean exercises concurrent admit/release/stats under
// -race to prove the capacity accounting has no data race.
func TestTurnkeyCapacityRaceClean(t *testing.T) {
	srv := &turnkeyServer{
		plan: macfit.TurnkeyProfile{
			ContextBudgetTokens: 8192,
			KVPoolBytes:         8 * 8192 * 2,
			KVBytesPerToken:     2,
			Tier:                macfit.ModelTier{ModelID: "local"},
		},
	}
	const workers = 16
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				if srv.beginChatRequest() {
					_ = srv.capacityStats()
					srv.endChatRequest()
				} else {
					_ = srv.capacityStats()
				}
			}
		}()
	}
	wg.Wait()
	if got := srv.capacityStats(); got.LiveSessions != 0 {
		t.Fatalf("live sessions leaked: %+v", got)
	}
}

// TestTurnkeyCapacityDisabledKeepsLegacyBehavior pins the backward-compatibility
// boundary: when the plan carries no KV pool (mock/legacy paths), admission is
// unbounded and existing tests stay byte-for-byte unaffected.
func TestTurnkeyCapacityDisabledKeepsLegacyBehavior(t *testing.T) {
	srv := &turnkeyServer{plan: macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: "local"}}}
	if cap := srv.capacity(); cap.MaxSessions != 0 {
		t.Fatalf("MaxSessions = %d, want 0 (unbounded) when no KV pool is declared", cap.MaxSessions)
	}
	for i := 0; i < 100; i++ {
		if !srv.beginChatRequest() {
			t.Fatalf("admission refused at i=%d with capacity disabled", i)
		}
	}
	if got := srv.capacityStats(); got.LiveSessions != 100 {
		t.Fatalf("live sessions = %d, want 100", got.LiveSessions)
	}
}

var _ = gateway.ChatRequest{}

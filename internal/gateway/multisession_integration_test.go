package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// multisession_integration_test.go — the multi-session multiplexing integration witness
// for Issue #10857:
//  1. fak serve multiplexes 100 concurrent session streams over a single loopback port
//     without port proliferation or process sprawl.
//  2. Scheduling across active sessions satisfies weighted fairness without starving
//     high-priority sessions.
//  3. 100 concurrent simulated sessions through admission.go and session.Table, verifying
//     <= 50 MB resident footprint delta and zero dropped turns.
//  4. Shared token pool enforcement and session lifecycle endpoints (list, pause, resume,
//     throttle, stop, SSE/NDJSON subscribe).

func setupMultiSessionTestServer(t *testing.T, tbl *session.Table, sched *session.Scheduler, pool *session.Pool, admPolicy AdmissionPolicy) (*Server, *AdmissionController, *httptest.Server) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{
		EngineID:  "test",
		Model:     "test-model",
		VDSO:      true,
		Table:     tbl,
		Scheduler: sched,
		Pool:      pool,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}}

	admCtl := NewAdmissionController(admPolicy)
	if sched != nil {
		admCtl.SetOrder(sched.Policy())
	}
	srv.SetAdmissionController(admCtl)

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	return srv, admCtl, ts
}

func postChatTurn(client *http.Client, url, trace string, maxTokens int) (*http.Response, []byte, error) {
	bodyMap := map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}
	if maxTokens > 0 {
		bodyMap["max_tokens"] = maxTokens
	}
	bodyBytes, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, nil, err
	}
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if trace != "" {
		req.Header.Set("X-Trace-Id", trace)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, nil, err
	}
	return resp, respBytes, nil
}

// TestMultiSessionMultiplexing100ConcurrentWitness proves that fak serve multiplexes 100
// concurrent session streams over a single loopback port with continuous batching admission,
// achieving zero dropped turns, table state recorded for all sessions, and <= 50 MB resident
// memory delta.
func TestMultiSessionMultiplexing100ConcurrentWitness(t *testing.T) {
	tbl := session.NewTable()
	sched := session.NewScheduler(session.WeightedFair)
	admPolicy := AdmissionPolicy{
		MaxNumSeqs:  16,
		TokenBudget: 1000000,
		MaxWaiting:  1000,
		AgingRounds: 1,
	}

	srv, admCtl, ts := setupMultiSessionTestServer(t, tbl, sched, nil, admPolicy)
	_ = srv
	_ = admCtl

	transport := &http.Transport{
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 200,
		MaxConnsPerHost:     200,
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second,
	}
	defer transport.CloseIdleConnections()

	// 1. Record baseline resident memory
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	const numSessions = 100
	const turnsPerSession = 5
	const expectedTurns = numSessions * turnsPerSession

	var wg sync.WaitGroup
	startGate := make(chan struct{})
	errCh := make(chan error, expectedTurns)
	var successCount int64

	// Launch 100 concurrent session streams
	for i := 1; i <= numSessions; i++ {
		wg.Add(1)
		go func(sessIdx int) {
			defer wg.Done()
			<-startGate
			traceID := fmt.Sprintf("session-%d", sessIdx)
			for turn := 0; turn < turnsPerSession; turn++ {
				resp, bodyBytes, err := postChatTurn(client, ts.URL, traceID, 10)
				if err != nil {
					errCh <- fmt.Errorf("session %s turn %d failed: %w", traceID, turn, err)
					return
				}
				if resp.StatusCode != http.StatusOK {
					errCh <- fmt.Errorf("session %s turn %d status %d: %s", traceID, turn, resp.StatusCode, string(bodyBytes))
					return
				}
				atomic.AddInt64(&successCount, 1)
			}
		}(i)
	}

	close(startGate)
	wg.Wait()

	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	// 2. Verify zero dropped turns: all 500 requests returned HTTP 200 OK
	if successCount != expectedTurns {
		t.Fatalf("successful turns = %d, want %d (zero dropped turns)", successCount, expectedTurns)
	}

	// 3. Verify session.Table records state for all 100 sessions
	snap := tbl.Snapshot()
	if len(snap) != numSessions {
		t.Fatalf("table snapshot len = %d, want %d", len(snap), numSessions)
	}
	for i := 1; i <= numSessions; i++ {
		traceID := fmt.Sprintf("session-%d", i)
		st := tbl.Get(traceID)
		if st.TraceID != traceID || st.Rev == 0 {
			t.Fatalf("session %s not properly recorded in table: %+v", traceID, st)
		}
	}

	// 4. Verify resident footprint delta <= 50 MB
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	delta := int64(m2.Alloc) - int64(m1.Alloc)
	t.Logf("Resident memory Alloc delta: %d bytes (%.2f MB)", delta, float64(delta)/(1024*1024))
	const maxAllowedDelta = 50 * 1024 * 1024
	if delta > maxAllowedDelta {
		t.Fatalf("resident footprint delta %d bytes exceeds limit %d bytes (50 MB)", delta, maxAllowedDelta)
	}
}

// TestMultiSessionWeightedFairScheduling verifies that scheduling across competing sessions
// with different priorities satisfies weighted fairness (high priority gets proportionally
// more turns) without starving low priority sessions.
func TestMultiSessionWeightedFairScheduling(t *testing.T) {
	tbl := session.NewTable()
	sched := session.NewScheduler(session.WeightedFair)
	admPolicy := AdmissionPolicy{
		MaxNumSeqs:  1,
		TokenBudget: 10000,
		MaxWaiting:  100,
		AgingRounds: 100,
	}

	srv, admCtl, ts := setupMultiSessionTestServer(t, tbl, sched, nil, admPolicy)

	// High Priority (0) vs Low Priority (2)
	// Weight(0) = (2-0)+1 = 3
	// Weight(2) = (2-2)+1 = 1
	tbl.SetPriority("sess-high", 0)
	tbl.SetPriority("sess-low", 2)

	// Level 1: Deterministic schedule order via AdmissionController with session.Scheduler
	admCtl.Offer(SeqRequest{TraceID: "blocker-init", Tokens: 100})
	for i := 1; i <= 6; i++ {
		admCtl.Offer(SeqRequest{TraceID: fmt.Sprintf("high-%d", i), SessionID: "sess-high", Tokens: 10})
	}
	for i := 1; i <= 2; i++ {
		admCtl.Offer(SeqRequest{TraceID: fmt.Sprintf("low-%d", i), SessionID: "sess-low", Tokens: 10})
	}
	admCtl.Complete("blocker-init")

	var admitted []SeqRequest
	for i := 0; i < 8; i++ {
		batch := admCtl.Schedule()
		if len(batch) != 1 {
			t.Fatalf("round %d: expected 1 admitted request, got %d", i, len(batch))
		}
		admitted = append(admitted, batch[0])
		admCtl.Complete(batch[0].TraceID)
	}
	wantOrder := []string{
		"high-1", "high-2", "low-1",
		"high-3", "high-4", "high-5",
		"low-2", "high-6",
	}
	for i, want := range wantOrder {
		if admitted[i].TraceID != want {
			t.Fatalf("admitted[%d] = %q, want %q", i, admitted[i].TraceID, want)
		}
	}

	// Level 2: End-to-end HTTP multiplexing with constrained concurrency (MaxNumSeqs = 1)
	var mu sync.Mutex
	var servedTraces []string
	gateUnblock := make(chan struct{})

	srv.planner = plannerFunc(func(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
		trace := ""
		if len(msgs) > 0 {
			trace = msgs[0].Content
		}
		if trace == "gate-blocker" {
			<-gateUnblock
		}
		mu.Lock()
		servedTraces = append(servedTraces, trace)
		mu.Unlock()
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		}, nil
	})

	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        50,
			MaxIdleConnsPerHost: 50,
		},
		Timeout: 30 * time.Second,
	}

	// Start gate blocker
	go func() {
		body, _ := json.Marshal(map[string]any{
			"model":    "test-model",
			"messages": []map[string]string{{"role": "user", "content": "gate-blocker"}},
		})
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Trace-Id", "gate-blocker")
		resp, err := client.Do(req)
		if err == nil {
			resp.Body.Close()
		}
	}()

	// Wait until gate blocker holds the running slot
	for {
		if admCtl.Stats().Running == 1 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}

	// Queue 6 turns for sess-high and 2 turns for sess-low
	var httpWg sync.WaitGroup
	postTurn := func(sessID string) {
		httpWg.Add(1)
		go func() {
			defer httpWg.Done()
			body, _ := json.Marshal(map[string]any{
				"model":    "test-model",
				"messages": []map[string]string{{"role": "user", "content": sessID}},
			})
			req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Trace-Id", sessID)
			resp, err := client.Do(req)
			if err != nil {
				t.Errorf("request %s failed: %v", sessID, err)
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Errorf("request %s status = %d", sessID, resp.StatusCode)
			}
		}()
	}

	for i := 0; i < 6; i++ {
		postTurn("sess-high")
	}
	for i := 0; i < 2; i++ {
		postTurn("sess-low")
	}

	// Wait until all 8 requests are queued in waiting
	for {
		if admCtl.Stats().Waiting == 8 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}

	// Unblock gate and wait for completion
	close(gateUnblock)
	httpWg.Wait()

	mu.Lock()
	results := append([]string(nil), servedTraces...)
	mu.Unlock()

	highCount := 0
	lowCount := 0
	for _, tr := range results {
		if tr == "sess-high" {
			highCount++
		} else if tr == "sess-low" {
			lowCount++
		}
	}
	if highCount != 6 || lowCount != 2 {
		t.Fatalf("expected 6 high and 2 low turns, got high=%d low=%d (all=%v)", highCount, lowCount, results)
	}

	// Verify low priority is never starved (served smoothly amidst high priority turns)
	firstLowIdx := -1
	for i, tr := range results {
		if tr == "sess-low" {
			firstLowIdx = i
			break
		}
	}
	if firstLowIdx == -1 || firstLowIdx >= len(results)-1 {
		t.Fatalf("low priority was starved: first low execution at index %d of %d (results=%v)", firstLowIdx, len(results), results)
	}
}

// TestMultiSessionSharedTokenPool verifies that when a shared token pool is configured across
// sessions, continuous batching sheds requests (HTTP 429) once the pool is exhausted and never
// exceeds the pool ceiling.
func TestMultiSessionSharedTokenPool(t *testing.T) {
	tbl := session.NewTable()
	sched := session.NewScheduler(session.WeightedFair)
	const poolCeiling = 60
	pool := session.NewPool(poolCeiling)

	admPolicy := AdmissionPolicy{
		MaxNumSeqs:  10,
		TokenBudget: 1000,
		MaxWaiting:  5,
	}
	srv, admCtl, ts := setupMultiSessionTestServer(t, tbl, sched, pool, admPolicy)

	// Planner that holds in-flight requests until released
	holdCh := make(chan struct{})
	srv.planner = plannerFunc(func(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
		<-holdCh
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		}, nil
	})

	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        50,
			MaxIdleConnsPerHost: 50,
		},
		Timeout: 30 * time.Second,
	}

	// Each turn asks for max_tokens: 25 -> estimateServedAdmissionTokens = 26 tokens.
	// Pool ceiling is 60 tokens: 2 concurrent requests draw 52 tokens (8 remaining).
	// Subsequent concurrent requests require 26 tokens > 8 remaining and are shed with HTTP 429.
	type result struct {
		traceID string
		status  int
		body    string
	}
	resCh := make(chan result, 5)
	var wg sync.WaitGroup

	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			trace := fmt.Sprintf("pool-sess-%d", id)
			resp, body, err := postChatTurn(client, ts.URL, trace, 25)
			if err != nil {
				resCh <- result{traceID: trace, status: 0, body: err.Error()}
				return
			}
			resCh <- result{traceID: trace, status: resp.StatusCode, body: string(body)}
		}(i)
	}

	// Wait until at least 2 requests are admitted into running
	for {
		if admCtl.Stats().Running >= 2 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}

	close(holdCh)
	wg.Wait()
	close(resCh)

	var okCount, shed429Count int
	for res := range resCh {
		if res.status == http.StatusOK {
			okCount++
		} else if res.status == http.StatusTooManyRequests {
			shed429Count++
		} else {
			t.Errorf("unexpected status %d for %s: %s", res.status, res.traceID, res.body)
		}
	}

	if shed429Count == 0 {
		t.Fatalf("expected at least one HTTP 429 shed turn, got ok=%d shed=%d", okCount, shed429Count)
	}
	if admCtl.Stats().Shed == 0 {
		t.Fatalf("expected AdmissionController.Stats().Shed > 0, got %d", admCtl.Stats().Shed)
	}

	// Verify pool ceiling was never exceeded
	if pool.Remaining() < 0 || pool.Remaining() > poolCeiling {
		t.Fatalf("pool remaining = %d, want between 0 and %d", pool.Remaining(), poolCeiling)
	}

	// Verify that when pool is completely drained, new requests are immediately shed with HTTP 429
	pool.Draw(pool.Remaining())
	if pool.Remaining() != 0 {
		t.Fatalf("expected 0 remaining in pool, got %d", pool.Remaining())
	}

	resp, body, err := postChatTurn(client, ts.URL, "drained-sess", 10)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429 Too Many Requests for drained pool, got %d (body: %s)", resp.StatusCode, string(body))
	}
}

// TestMultiSessionLifecycleEndpoints tests the session lifecycle surface:
//   - GET /v1/fak/sessions: returns active sessions, states, and tokens_used
//   - POST /v1/fak/session/{id}/{pause|resume|throttle|stop}: verifies state changes and 409 refusals
//   - GET /v1/fak/session/{id}/subscribe: live state transitions stream via SSE and NDJSON
func TestMultiSessionLifecycleEndpoints(t *testing.T) {
	tbl := session.NewTable()
	sched := session.NewScheduler(session.WeightedFair)
	admPolicy := AdmissionPolicy{MaxNumSeqs: 16, TokenBudget: 10000, MaxWaiting: 100}

	srv, _, ts := setupMultiSessionTestServer(t, tbl, sched, nil, admPolicy)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "lifecycle-response"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 10, CompletionTokens: 25, TotalTokens: 35},
	}}

	client := http.DefaultClient

	// 1. GET /v1/fak/sessions: returns active sessions, states, and token usage (tokens_used)
	resp, body, err := postChatTurn(client, ts.URL, "session-live", 0)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("setup turn failed: status=%d, body=%s, err=%v", resp.StatusCode, string(body), err)
	}

	listResp, err := client.Get(ts.URL + "/v1/fak/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer listResp.Body.Close()
	if listResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/fak/sessions status = %d", listResp.StatusCode)
	}
	listBytes, err := io.ReadAll(listResp.Body)
	if err != nil {
		t.Fatal(err)
	}
	var sessList SessionListResponse
	if err := json.Unmarshal(listBytes, &sessList); err != nil {
		t.Fatalf("failed to decode SessionListResponse: %v, raw: %s", err, string(listBytes))
	}
	if sessList.Count == 0 || len(sessList.Sessions) == 0 {
		t.Fatalf("expected at least 1 session, got count=%d len=%d", sessList.Count, len(sessList.Sessions))
	}
	foundLive := false
	for _, s := range sessList.Sessions {
		if s.TraceID == "session-live" {
			foundLive = true
			if s.Run != "running" {
				t.Fatalf("expected session-live run=running, got %s", s.Run)
			}
			if s.TokensUsed <= 0 {
				t.Fatalf("expected tokens_used > 0, got %d", s.TokensUsed)
			}
		}
	}
	if !foundLive {
		t.Fatal("session-live not found in /v1/fak/sessions")
	}
	if !strings.Contains(string(listBytes), `"tokens_used"`) {
		t.Fatalf("GET /v1/fak/sessions wire JSON missing 'tokens_used': %s", string(listBytes))
	}

	// 2. Control endpoints: POST pause, resume, throttle, stop
	targetSession := "session-lifecycle"
	resp, body, err = postChatTurn(client, ts.URL, targetSession, 0)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("initial turn failed: status=%d, body=%s, err=%v", resp.StatusCode, string(body), err)
	}

	// POST pause
	pauseResp, err := client.Post(ts.URL+"/v1/fak/session/"+targetSession+"/pause", "application/json", strings.NewReader(`{"reason":"operator-paused"}`))
	if err != nil || pauseResp.StatusCode != http.StatusOK {
		t.Fatalf("POST pause failed: status=%v, err=%v", pauseResp.StatusCode, err)
	}
	pauseResp.Body.Close()
	if st := tbl.Get(targetSession); st.Run != session.Paused {
		t.Fatalf("expected state Paused, got %s", st.Run)
	}
	// Subsequent turn must be refused with 409
	resp, body, err = postChatTurn(client, ts.URL, targetSession, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("paused session turn status = %d, want 409 Conflict (body: %s)", resp.StatusCode, string(body))
	}

	// POST resume
	resumeResp, err := client.Post(ts.URL+"/v1/fak/session/"+targetSession+"/resume", "application/json", nil)
	if err != nil || resumeResp.StatusCode != http.StatusOK {
		t.Fatalf("POST resume failed: status=%v, err=%v", resumeResp.StatusCode, err)
	}
	resumeResp.Body.Close()
	if st := tbl.Get(targetSession); st.Run != session.Running {
		t.Fatalf("expected state Running, got %s", st.Run)
	}
	// Subsequent turn must succeed with 200 OK
	resp, body, err = postChatTurn(client, ts.URL, targetSession, 0)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("resumed session turn failed: status=%d, body=%s, err=%v", resp.StatusCode, string(body), err)
	}

	// POST throttle
	throttleResp, err := client.Post(ts.URL+"/v1/fak/session/"+targetSession+"/throttle", "application/json", strings.NewReader(`{"reason":"traffic-pace"}`))
	if err != nil || throttleResp.StatusCode != http.StatusOK {
		t.Fatalf("POST throttle failed: status=%v, err=%v", throttleResp.StatusCode, err)
	}
	throttleResp.Body.Close()
	if st := tbl.Get(targetSession); st.Run != session.Throttled {
		t.Fatalf("expected state Throttled, got %s", st.Run)
	}
	// Subsequent turn still proceeds with 200 OK (throttled is advancing)
	resp, body, err = postChatTurn(client, ts.URL, targetSession, 0)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("throttled session turn failed: status=%d, body=%s, err=%v", resp.StatusCode, string(body), err)
	}

	// POST stop
	stopResp, err := client.Post(ts.URL+"/v1/fak/session/"+targetSession+"/stop", "application/json", strings.NewReader(`{"reason":"done"}`))
	if err != nil || stopResp.StatusCode != http.StatusOK {
		t.Fatalf("POST stop failed: status=%v, err=%v", stopResp.StatusCode, err)
	}
	stopResp.Body.Close()
	if st := tbl.Get(targetSession); st.Run != session.Stopped {
		t.Fatalf("expected state Stopped, got %s", st.Run)
	}
	// Subsequent turn must be refused with 409
	resp, body, err = postChatTurn(client, ts.URL, targetSession, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("stopped session turn status = %d, want 409 Conflict (body: %s)", resp.StatusCode, string(body))
	}

	// 3. GET /v1/fak/session/{id}/subscribe with SSE and NDJSON
	// SSE subscription
	subscribeSSE := "session-sub-sse"
	ctxSSE, cancelSSE := context.WithCancel(context.Background())
	defer cancelSSE()

	reqSSE, err := http.NewRequestWithContext(ctxSSE, http.MethodGet, ts.URL+"/v1/fak/session/"+subscribeSSE+"/subscribe?stream=sse", nil)
	if err != nil {
		t.Fatal(err)
	}
	reqSSE.Header.Set("Accept", "text/event-stream")
	respSSE, err := client.Do(reqSSE)
	if err != nil {
		t.Fatal(err)
	}
	defer respSSE.Body.Close()

	if ct := respSSE.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("SSE Content-Type = %q, want text/event-stream", ct)
	}

	// Trigger live state transition to paused
	go func() {
		time.Sleep(20 * time.Millisecond)
		r, err := client.Post(ts.URL+"/v1/fak/session/"+subscribeSSE+"/pause", "application/json", strings.NewReader(`{"reason":"sse-transition"}`))
		if err == nil {
			r.Body.Close()
		}
	}()

	readerSSE := bufio.NewReader(respSSE.Body)
	foundSSE := false
	for i := 0; i < 20; i++ {
		line, err := readerSSE.ReadString('\n')
		if err != nil {
			break
		}
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if strings.Contains(data, subscribeSSE) && strings.Contains(data, "paused") {
				foundSSE = true
				break
			}
		}
	}
	if !foundSSE {
		t.Fatalf("did not receive live SSE state transition event for %s", subscribeSSE)
	}
	cancelSSE()

	// NDJSON subscription
	subscribeND := "session-sub-ndjson"
	ctxND, cancelND := context.WithCancel(context.Background())
	defer cancelND()

	reqND, err := http.NewRequestWithContext(ctxND, http.MethodGet, ts.URL+"/v1/fak/session/"+subscribeND+"/subscribe?stream=ndjson", nil)
	if err != nil {
		t.Fatal(err)
	}
	reqND.Header.Set("Accept", "application/x-ndjson")
	respND, err := client.Do(reqND)
	if err != nil {
		t.Fatal(err)
	}
	defer respND.Body.Close()

	if ct := respND.Header.Get("Content-Type"); !strings.Contains(ct, "application/x-ndjson") {
		t.Fatalf("NDJSON Content-Type = %q, want application/x-ndjson", ct)
	}

	// Trigger live state transition to paused
	go func() {
		time.Sleep(20 * time.Millisecond)
		r, err := client.Post(ts.URL+"/v1/fak/session/"+subscribeND+"/pause", "application/json", strings.NewReader(`{"reason":"ndjson-transition"}`))
		if err == nil {
			r.Body.Close()
		}
	}()

	readerND := bufio.NewReader(respND.Body)
	foundND := false
	for i := 0; i < 20; i++ {
		line, err := readerND.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line != "" && strings.Contains(line, subscribeND) && strings.Contains(line, "paused") {
			foundND = true
			break
		}
	}
	if !foundND {
		t.Fatalf("did not receive live NDJSON state transition event for %s", subscribeND)
	}
	cancelND()
}

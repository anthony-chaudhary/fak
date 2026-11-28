package gateway

// leak_test.go — goroutine-leak sentinel for the long-lived serving process.
//
// The streaming /v1/messages handler spawns one goroutine per request to run the planner
// while the main handler pumps SSE pings off a time.NewTicker, returning on whichever of
// {planner done, ticker, client disconnect} fires first (messages.go). The correctness of
// that pattern hinges on two things that are easy to regress: the result channel is
// buffered (cap 1) so the planner goroutine's send never blocks after the client has gone,
// and the ticker is defer-stopped. If either breaks, every disconnected stream leaks a
// goroutine (and a ticker) for the life of the server. This test drives many streams —
// half of them cancelled mid-flight — through the reusable leakcheck.Stable sentinel,
// which asserts the goroutine count returns to baseline (it would grow by ~N on a leak).

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/leakcheck"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

func TestStreamingHandlerDoesNotLeakGoroutines(t *testing.T) {
	old := anthropicStreamPingInterval
	anthropicStreamPingInterval = 2 * time.Millisecond // make the ticker fire during each stream
	defer func() { anthropicStreamPingInterval = old }()

	srv := newTestServer(t)
	srv.planner = delayedPlanner{
		delay: 20 * time.Millisecond, // long enough to cancel a stream mid-flight
		comp: &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "pong"},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11},
		},
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Each request gets a fresh connection so lingering keep-alive conn goroutines don't
	// masquerade as leaks.
	transport := &http.Transport{DisableKeepAlives: true}
	client := &http.Client{Transport: transport}
	reqBody := `{"model":"m","stream":true,"messages":[{"role":"user","content":"go"}]}`
	post := func(ctx context.Context) (*http.Response, error) {
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/messages", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		return client.Do(req)
	}

	// One stream per iteration: even = full read (planner-done path), odd = cancel mid-stream
	// (client-disconnect path — the one that must let the per-request goroutine + ticker exit
	// rather than block on the buffered result send).
	body := func(i int) {
		if i%2 == 0 {
			r, err := post(context.Background())
			if err != nil {
				return
			}
			io.Copy(io.Discard, r.Body)
			r.Body.Close()
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r, err := post(ctx)
		if err != nil {
			return // a cancel/err before headers is fine; nothing to leak
		}
		buf := make([]byte, 64)
		_, _ = r.Body.Read(buf) // read a little, then yank the connection
		cancel()
		r.Body.Close()
	}

	base, final := leakcheck.Stable(t, leakcheck.StableOpts{
		Iters:  40,              // ~20 of them cancelled mid-flight
		Slack:  8,               // absorbs scheduler/GC jitter + the odd un-reaped conn (≪ 40)
		Settle: 3 * time.Second, // planner honors ctx, so cancelled goroutines return promptly
	}, body)
	t.Logf("goroutines stable across 40 streams (20 cancelled mid-flight): baseline=%d final=%d", base, final)
}

func TestA2APushConfigsBoundedAcrossEvictions(t *testing.T) {
	store := getA2AStore()
	store.mu.Lock()
	savedTasks := store.tasks
	store.tasks = make(map[string]*a2aTask)
	store.mu.Unlock()

	a2aPushMu.Lock()
	savedPush := a2aPushSingleton
	a2aPushSingleton = newA2APushStore()
	ps := a2aPushSingleton
	a2aPushMu.Unlock()

	t.Cleanup(func() {
		store.mu.Lock()
		store.tasks = savedTasks
		store.mu.Unlock()
		a2aPushMu.Lock()
		a2aPushSingleton = savedPush
		a2aPushMu.Unlock()
	})

	baseTime := time.Date(2026, 11, 27, 0, 0, 0, 0, time.UTC)
	total := a2aMaxTasks + 64
	for i := 0; i < total; i++ {
		id := fmt.Sprintf("task-%05d", i)
		ts := baseTime.Add(time.Duration(i) * time.Millisecond)
		store.mu.Lock()
		store.insertLocked(id, &a2aTask{
			TaskID:    id,
			State:     "completed",
			CreatedAt: ts,
			UpdatedAt: ts,
		})
		store.mu.Unlock()
		ps.setConfig(id, a2aPushConfig{
			URL:       "https://example.com/hook",
			CallerID:  "caller-1",
			CreatedAt: ts,
		})
	}

	store.mu.RLock()
	taskCount := len(store.tasks)
	store.mu.RUnlock()
	if taskCount > a2aMaxTasks {
		t.Fatalf("a2aTaskStore len = %d, want <= %d", taskCount, a2aMaxTasks)
	}

	ps.mu.RLock()
	cfgCount := len(ps.configs)
	_, oldestPresent := ps.configs["task-00000"]
	ps.mu.RUnlock()
	if cfgCount > a2aMaxTasks {
		t.Fatalf("a2aPushStore.configs len = %d, want <= %d (LEAK)", cfgCount, a2aMaxTasks)
	}
	if oldestPresent {
		t.Fatalf("evicted task-00000 still present in a2aPushStore.configs")
	}
}

func TestServerCloseReleasesGlobalMapsAndEvictor(t *testing.T) {
	base, final := leakcheck.Stable(t, leakcheck.StableOpts{
		Iters:  12,
		Slack:  4,
		Settle: 2 * time.Second,
	}, func(i int) {
		srv := newTestServer(t)
		srv.SetToolprocReuse(toolproc.ArmedConfig{}, nil)
		_ = srv.ControlManager()
		_ = srv.clientRuntime()
		ev := NewIdleGatedEvictor(nil, nil)
		srv.SetIdleGatedEvictor(ev)
		ev.Start(context.Background())

		srv.Close()

		if _, ok := reuseSeams.Load(srv); ok {
			t.Fatalf("iter %d: reuseSeams still holds closed Server", i)
		}
		if _, ok := serverControlManagers.Load(srv); ok {
			t.Fatalf("iter %d: serverControlManagers still holds closed Server", i)
		}
		if _, ok := sessionClientRuntimes.Load(srv); ok {
			t.Fatalf("iter %d: sessionClientRuntimes still holds closed Server", i)
		}
		if ev.IsRunning() {
			t.Fatalf("iter %d: kvEvictor still running after Server.Close()", i)
		}
	})
	t.Logf("Server.Close goroutines stable: baseline=%d final=%d", base, final)
}

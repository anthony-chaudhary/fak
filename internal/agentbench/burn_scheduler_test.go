package agentbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBurnSteadyEnforcesCausalConcurrencyAndCutoff(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	addBurnCorpusMaterial(t, repo)
	normal, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{}, repo)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := buildBurnInCorpus(context.Background(), &burnMaterialEncoder{}, normal, repo)
	if err != nil {
		t.Fatal(err)
	}

	server, wire := burnSchedulerEndpoint(t)
	defer server.Close()
	now := time.Now().UTC()
	waitsOutsideSlot := atomic.Bool{}
	waitsOutsideSlot.Store(true)
	waitEntered, releaseWait := make(chan struct{}), make(chan struct{})
	var waitOnce sync.Once
	receiptDone := make(chan burnSteadyReceipt, 1)
	go func() {
		receiptDone <- runBurnSteady(context.Background(), burnSteadyConfig{Client: http.DefaultClient, Endpoint: server.URL, Model: "fixture-model", Corpus: corpus, Concurrency: 4, QueueDepth: 8, AdmissionCutoff: now.Add(time.Hour), DrainDeadline: now.Add(time.Hour + 90*time.Second), Now: time.Now, Wait: func(_ context.Context, session string, turn int, _ time.Duration) error {
			wire.mu.Lock()
			active := wire.active[session]
			wire.mu.Unlock()
			if active {
				waitsOutsideSlot.Store(false)
			}
			if session == "burn-01" && turn == 2 {
				waitOnce.Do(func() { close(waitEntered) })
				<-releaseWait
			}
			return nil
		}})
	}()
	select {
	case <-waitEntered:
	case <-time.After(time.Second):
		close(releaseWait)
		t.Fatal("session burn-01 never reached held tool wait")
	}
	advanced := false
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		wire.mu.Lock()
		for session, turns := range wire.turns {
			if session != "burn-01" && len(turns) >= 3 {
				advanced = true
			}
		}
		wire.mu.Unlock()
		if advanced {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	close(releaseWait)
	if !advanced {
		t.Fatal("one session's tool wait globally stalled other causal sessions")
	}
	receipt := <-receiptDone
	if receipt.PeakInFlight != int(wire.peak.Load()) {
		t.Fatalf("reported HTTP peak=%d want observed request start/finish peak=%d", receipt.PeakInFlight, wire.peak.Load())
	}
	if receipt.PeakQueued != observedBurnQueuePeak(receipt.Requests) {
		t.Fatalf("reported queue peak=%d want lifecycle-observed=%d", receipt.PeakQueued, observedBurnQueuePeak(receipt.Requests))
	}
	if receipt.Accepted != 256 || receipt.Completed != 256 || receipt.RejectedAfterCutoff != 0 || receipt.PeakInFlight != 4 || receipt.PeakQueued > 8 || len(receipt.Errors) != 0 {
		t.Fatalf("full steady schedule=%+v", receipt)
	}
	if !waitsOutsideSlot.Load() {
		t.Fatal("recorded tool wait retained an HTTP inference slot")
	}
	wire.mu.Lock()
	if wire.sameSessionOverlap {
		t.Fatal("two turns from one session overlapped")
	}
	for session, turns := range wire.turns {
		for i, turn := range turns {
			if turn != i+1 {
				t.Fatalf("session %s dispatch order=%v", session, turns)
			}
		}
	}
	wire.mu.Unlock()
	if len(receipt.Requests) != 256 {
		t.Fatalf("lifecycle receipts=%d", len(receipt.Requests))
	}
	for _, request := range receipt.Requests {
		if request.SessionID == "" || request.Turn <= 0 || request.RequestID == "" || request.ReleasedAt.IsZero() || request.EnqueuedAt.IsZero() || request.DequeuedAt.IsZero() || request.DispatchedAt.IsZero() || request.FirstOutputAt.IsZero() || request.EndedAt.IsZero() || request.EnqueuedAt.Before(request.ReleasedAt) || request.DequeuedAt.Before(request.EnqueuedAt) || request.DispatchedAt.Before(request.DequeuedAt) || request.EndedAt.Before(request.DispatchedAt) || request.Status != "completed" {
			t.Fatalf("incomplete request lifecycle: %+v", request)
		}
	}

	t.Run("absolute drain deadline interrupts held wait", func(t *testing.T) {
		started := time.Now()
		deadline := started.Add(150 * time.Millisecond)
		got := runBurnSteady(context.Background(), burnSteadyConfig{Client: http.DefaultClient, Endpoint: server.URL, Model: "fixture-model", Corpus: corpus, Concurrency: 2, QueueDepth: 4, AdmissionCutoff: deadline, DrainDeadline: deadline, Now: time.Now, Wait: func(ctx context.Context, _ string, _ int, _ time.Duration) error { <-ctx.Done(); return ctx.Err() }})
		if time.Since(started) > time.Second || len(got.Errors) == 0 {
			t.Fatalf("held wait ignored absolute drain deadline: elapsed=%v receipt=%+v", time.Since(started), got)
		}
		for _, request := range got.Requests {
			if request.Status == "accepted" || (request.Status != "rejected" && request.EndedAt.IsZero()) {
				t.Fatalf("admitted request lost its terminal receipt at drain deadline: %+v", request)
			}
		}
	})

	t.Run("durable lifecycle failure is surfaced", func(t *testing.T) {
		file, err := os.Create(filepath.Join(t.TempDir(), "closed-events.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		_ = file.Close()
		badSink := &eventSink{file: file, terminal: map[int]lifecycleEvent{}, inFlight: map[string]int{}, peak: map[string]int{}}
		beforeWire := wire.requests.Load()
		got := runBurnSteady(context.Background(), burnSteadyConfig{Client: http.DefaultClient, Endpoint: server.URL, Model: "fixture-model", Corpus: corpus, Sink: badSink, Concurrency: 1, QueueDepth: 2, AdmissionCutoff: now.Add(time.Hour), DrainDeadline: now.Add(time.Hour + 90*time.Second), Now: time.Now, Wait: func(context.Context, string, int, time.Duration) error { return nil }})
		if len(got.Errors) == 0 {
			t.Fatal("closed durable event sink was ignored")
		}
		if got.PeakInFlight != 0 || wire.requests.Load() != beforeWire {
			t.Fatalf("sink failure reported HTTP work that never reached wire: peak=%d wire delta=%d", got.PeakInFlight, wire.requests.Load()-beforeWire)
		}
	})

	t.Run("cutoff rejects queued work but drains admitted", func(t *testing.T) {
		cutServer, cutWire := burnSchedulerEndpoint(t)
		defer cutServer.Close()
		cutoff := time.Date(2026, 9, 10, 12, 0, 0, 0, time.UTC)
		var after atomic.Bool
		cutWire.afterRequest = func(count int) {
			if count >= 12 {
				after.Store(true)
			}
		}
		clock := func() time.Time {
			if after.Load() {
				return cutoff.Add(time.Second)
			}
			return cutoff.Add(-time.Second)
		}
		got := runBurnSteady(context.Background(), burnSteadyConfig{Client: http.DefaultClient, Endpoint: cutServer.URL, Model: "fixture-model", Corpus: corpus, Concurrency: 2, QueueDepth: 4, AdmissionCutoff: cutoff, DrainDeadline: cutoff.Add(90 * time.Second), Now: clock, Wait: func(context.Context, string, int, time.Duration) error { return nil }})
		if got.Accepted == 0 || got.Accepted >= 256 || got.Completed != got.Accepted || got.RejectedAfterCutoff != 256-got.Accepted || cutWire.requests.Load() != int64(got.Accepted) || len(got.Errors) != 0 {
			t.Fatalf("cutoff/drain receipt=%+v wire=%d", got, cutWire.requests.Load())
		}
		for _, request := range got.Requests {
			if request.Status == "rejected" {
				if request.RejectedAt.IsZero() || request.EndedAt.IsZero() {
					t.Fatalf("rejected queued request missing durable terminal timing: %+v", request)
				}
				continue
			}
			if request.DispatchedAt.After(cutoff) {
				t.Fatalf("request dispatched after admission cutoff: %+v", request)
			}
			if request.EndedAt.After(cutoff.Add(90 * time.Second)) {
				t.Fatalf("request exceeded absolute drain grace: %+v", request)
			}
		}
	})
}

func observedBurnQueuePeak(requests []burnSteadyRequestReceipt) int {
	type point struct {
		at    time.Time
		delta int
		order uint64
	}
	points := make([]point, 0, len(requests)*2)
	for _, request := range requests {
		if request.EnqueuedAt.IsZero() || request.DequeuedAt.IsZero() {
			continue
		}
		points = append(points, point{request.EnqueuedAt, 1, request.EnqueueOrder}, point{request.DequeuedAt, -1, request.DequeueOrder})
	}
	sort.Slice(points, func(i, j int) bool {
		if points[i].order != 0 && points[j].order != 0 {
			return points[i].order < points[j].order
		}
		if points[i].at.Equal(points[j].at) {
			return points[i].delta < points[j].delta
		}
		return points[i].at.Before(points[j].at)
	})
	active, peak := 0, 0
	for _, point := range points {
		active += point.delta
		peak = max(peak, active)
	}
	return peak
}

type burnSchedulerWire struct {
	mu                 sync.Mutex
	active             map[string]bool
	turns              map[string][]int
	sameSessionOverlap bool
	inFlight           atomic.Int64
	requests           atomic.Int64
	afterRequest       func(int)
	peak               atomic.Int64
}

func burnSchedulerEndpoint(t *testing.T) (*httptest.Server, *burnSchedulerWire) {
	t.Helper()
	state := &burnSchedulerWire{active: map[string]bool{}, turns: map[string][]int{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		metadata, _ := body["metadata"].(map[string]any)
		session, _ := metadata["session"].(string)
		turn := int(metadata["turn"].(float64))
		state.mu.Lock()
		if state.active[session] {
			state.sameSessionOverlap = true
		}
		state.active[session] = true
		state.turns[session] = append(state.turns[session], turn)
		state.mu.Unlock()
		active := state.inFlight.Add(1)
		for observed := state.peak.Load(); active > observed && !state.peak.CompareAndSwap(observed, active); observed = state.peak.Load() {
		}
		count := int(state.requests.Add(1))
		if state.afterRequest != nil {
			state.afterRequest(count)
		}
		defer func() { state.inFlight.Add(-1); state.mu.Lock(); state.active[session] = false; state.mu.Unlock() }()
		time.Sleep(5 * time.Millisecond)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"progress\"}}]}\n\n")
		w.(http.Flusher).Flush()
		fmt.Fprint(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n")
	}))
	return server, state
}

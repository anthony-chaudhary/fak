package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Candidate-only tests: these new ledger symbols are deliberately excluded from
// the parent-compatible HTTP feature-absence witness.
func outcomeTestRecord(sequence uint64, now time.Time) requestOutcomeRecord {
	return requestOutcomeRecord{
		key: requestOutcomeKey{"owner", fmt.Sprintf("trace-%d", sequence),
			fmt.Sprintf("00-%032x-%016x-00", 1, sequence)},
		sequence: sequence, started: now.Add(-time.Second), finished: now,
		outcome: "succeeded", source: "native_generate_return",
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestRequestOutcomeRetentionAndFirstTerminal(t *testing.T) {
	t.Parallel()
	now := time.Unix(1700000000, 0)
	var store requestOutcomeStore
	first := outcomeTestRecord(1, now)
	store.offer(first, now)
	duplicate := first
	duplicate.finished = now.Add(time.Hour)
	duplicate.outcome = "completion_unknown"
	store.offer(duplicate, duplicate.finished)
	got, ok, accounting := store.lookup(first.key, duplicate.finished)
	if !ok || got != first || accounting.Duplicates != 1 {
		t.Fatalf("first terminal replaced: %+v %+v", got, accounting)
	}
	_, ok, accounting = store.lookup(first.key, now.Add(requestOutcomeMaxAge))
	if ok || accounting.Expired != 1 {
		t.Fatalf("duplicate refreshed retention: %+v", accounting)
	}
	for n := uint64(1); n <= requestOutcomeLimit+1; n++ {
		row := outcomeTestRecord(n, now.Add(time.Duration(n)*time.Nanosecond))
		store.offer(row, row.finished)
	}
	if len(store.rows) != requestOutcomeLimit {
		t.Fatalf("unbounded store: %d", len(store.rows))
	}
	_, ok, accounting = store.lookup(first.key, now.Add(time.Second))
	if ok || accounting.Evicted != 1 {
		t.Fatalf("oldest survived cap: %+v", accounting)
	}
	last := outcomeTestRecord(requestOutcomeLimit+1, now)
	if _, ok, _ := store.lookup(last.key, now.Add(time.Second)); !ok {
		t.Fatal("newest absent")
	}
	var restarted requestOutcomeStore
	if _, ok, _ := restarted.lookup(last.key, now); ok {
		t.Fatal("restart inferred terminal state")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestRequestOutcomeContentionDropsOnlyNewOffer(t *testing.T) {
	t.Parallel()
	now := time.Unix(1700000000, 0)
	var store requestOutcomeStore
	first, second := outcomeTestRecord(1, now), outcomeTestRecord(2, now)
	store.offer(first, now)
	store.mu.Lock()
	done := make(chan struct{})
	go func() { store.offer(second, now); close(done) }()
	select {
	case <-done:
		store.mu.Unlock()
	case <-time.After(time.Second):
		store.mu.Unlock()
		<-done
		t.Fatal("terminal producer waited for store lock")
	}
	if _, ok, _ := store.lookup(first.key, now); !ok {
		t.Fatal("ordinary contention invalidated retained history")
	}
	_, ok, accounting := store.lookup(second.key, now)
	if ok || accounting.Dropped != 1 || accounting.Offers != 2 {
		t.Fatalf("dropped new attempt: %+v", accounting)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestRequestOutcomeRejectsInvalidIdentity(t *testing.T) {
	t.Parallel()
	now := time.Unix(1700000000, 0)
	for _, mutate := range []func(*requestOutcomeRecord){
		func(row *requestOutcomeRecord) { row.key.traceID = strings.Repeat("x", requestOutcomeIdentityLimit+1) },
		func(row *requestOutcomeRecord) { row.key.owner = strings.Repeat("x", requestOutcomeIdentityLimit+1) },
		func(row *requestOutcomeRecord) { row.key.traceID = string([]byte{0xff}) },
		func(row *requestOutcomeRecord) { row.key.traceID = " trimmed " },
		func(row *requestOutcomeRecord) {
			row.key.traceparent = strings.ToUpper("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-00")
		},
		func(row *requestOutcomeRecord) { row.sequence = 0 },
	} {
		var store requestOutcomeStore
		row := outcomeTestRecord(1, now)
		mutate(&row)
		store.offer(row, now)
		if len(store.rows) != 0 || store.dropped.Load() != 1 {
			t.Fatal("invalid identity retained")
		}
	}
}

type outcomeFailingWriter struct {
	header http.Header
	cancel context.CancelFunc
	writes int
}

func (w *outcomeFailingWriter) Header() http.Header { return w.header }
func (w *outcomeFailingWriter) WriteHeader(int)     {}
func (w *outcomeFailingWriter) Write([]byte) (int, error) {
	w.writes++
	w.cancel()
	return 0, errors.New("fixture response delivery failed")
}

// fak-test:runtime medium est=5s lane=default
func TestRequestOutcomeNativeSingleFinalizerAfterDeliveryFailure(t *testing.T) {
	t.Parallel()
	s := requestOutcomeTestServer(t, requestOutcomeNativePlanner(t))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest("POST", "/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1,"temperature":0}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer outcome-test-key")
	req.Header.Set("X-Trace-Id", "late-cancel-"+randomHex(16))
	w := &outcomeFailingWriter{header: make(http.Header), cancel: cancel}
	s.Handler().ServeHTTP(w, req)
	if w.writes == 0 || ctx.Err() == nil {
		t.Fatal("response failure/cancel fixture did not execute")
	}
	key := requestOutcomeKey{"", w.header.Get("X-Trace-Id"), w.header.Get("traceparent")}
	row, ok, accounting := s.metrics.requestOutcomes.lookup(key, time.Now())
	if !ok || row.outcome != "succeeded" || accounting.Offers != 1 || accounting.Duplicates != 0 {
		t.Fatalf("late cancellation/deferred fallback overwrote native evidence: %+v %+v", row, accounting)
	}
}

type outcomePanicPlanner struct{}

func (outcomePanicPlanner) Model() string { return "outcome-synth" }
func (outcomePanicPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	panic("fixture planner panic")
}

// fak-test:runtime fast est=100ms lane=default
func TestRequestOutcomePanicAndHeaderDisagreement(t *testing.T) {
	t.Parallel()
	s := requestOutcomeTestServer(t, outcomePanicPlanner{})
	req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"prompt":"hi","max_tokens":1}`))
	req.Header.Set("Authorization", "Bearer outcome-test-key")
	req.Header.Set("X-Trace-Id", "panic-"+randomHex(16))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	key := requestOutcomeKey{"", rec.Header().Get("X-Trace-Id"), rec.Header().Get("traceparent")}
	row, ok, accounting := s.metrics.requestOutcomes.lookup(key, time.Now())
	if !ok || row.outcome != "completion_unknown" || accounting.Offers != 1 {
		t.Fatalf("panic fallback: %+v %+v", row, accounting)
	}
	headers := make(http.Header)
	headers.Set("X-Trace-Id", "client-received")
	headers.Set("traceparent", newTraceContext("", 0).String())
	finish := s.beginRequestOutcome(withInflightID(context.Background(), 10), "", "admission-reset", headers)
	finish(nil, nil)
	finish(nil, context.Canceled)
	if s.metrics.requestOutcomes.offers.Load() != 2 || s.metrics.requestOutcomes.dropped.Load() != 1 {
		t.Fatal("header disagreement was retained or fallback offered twice")
	}
}

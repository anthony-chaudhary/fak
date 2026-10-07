package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type failingPlanner struct {
	err       error
	emitFirst bool
	block     bool
}

func (p failingPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if p.block {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	return nil, p.err
}

func (failingPlanner) Model() string { return "test-model" }

func (failingPlanner) StreamingSupported() bool { return true }

func (p failingPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, m []agent.Message, t []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if p.emitFirst {
		// Windows' coarse wall clock can read the same instant across a fast call.
		time.Sleep(20 * time.Millisecond)
		if err := sink("partial"); err != nil {
			return nil, err
		}
		time.Sleep(5 * time.Millisecond)
	}
	return p.Complete(ctx, m, t, opts...)
}

type perfDelayedPlanner struct {
	inner agent.Planner
	d     time.Duration
}

func (p perfDelayedPlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(p.d)
	return p.inner.Complete(ctx, m, t, opts...)
}

func (p perfDelayedPlanner) Model() string { return p.inner.Model() }

func postPerfChat(t *testing.T, ctx context.Context, url string, stream bool) (*http.Response, error) {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
		"stream":   stream,
	})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(req)
}

func waitPerfRows(t *testing.T, srv *Server, n int) []perfledger.Record {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		recs, _, _ := srv.metrics.perfRecordsSnapshot()
		if len(recs) >= n || time.Now().After(deadline) {
			return recs
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestPerfLedgerFailedTurnLeavesOneRow(t *testing.T) {
	cases := []struct {
		name       string
		planner    failingPlanner
		stream     bool
		wantClass  string
		wantStatus int
		wantTTFT   bool
	}{
		{"first token timeout", failingPlanner{err: agent.NewFirstTokenStalledError(time.Second)}, false, perfledger.ErrorFirstTokenTimeout, http.StatusGatewayTimeout, false},
		{"upstream 5xx", failingPlanner{err: &agent.UpstreamStatusError{Status: http.StatusInternalServerError}}, false, perfledger.ErrorUpstreamStatus, http.StatusBadGateway, false},
		{"upstream error before stream", failingPlanner{err: errors.New("parse failure")}, true, perfledger.ErrorUpstream, http.StatusBadGateway, false},
		{"stall before first byte", failingPlanner{err: &agent.UpstreamStalledError{Kind: "idle", Idle: time.Second}}, true, perfledger.ErrorFirstTokenTimeout, http.StatusGatewayTimeout, false},
		{"stall mid stream", failingPlanner{err: &agent.UpstreamStalledError{Kind: "idle", Idle: time.Second}, emitFirst: true}, true, perfledger.ErrorStall, http.StatusOK, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = tc.planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			resp, err := postPerfChat(t, context.Background(), ts.URL, tc.stream)
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			resp.Body.Close()
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("wire status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
			recs := waitPerfRows(t, srv, 1)
			if len(recs) != 1 {
				t.Fatalf("perf rows = %d, want exactly 1: %+v", len(recs), recs)
			}
			got := recs[0]
			if got.Error != tc.wantClass || got.Status != tc.wantStatus || got.FinishReason != perfledger.FinishReasonError {
				t.Fatalf("row error/status/finish = %q/%d/%q, want %q/%d/%q", got.Error, got.Status, got.FinishReason, tc.wantClass, tc.wantStatus, perfledger.FinishReasonError)
			}
			if got.E2EMS <= 0 && tc.wantTTFT {
				t.Fatalf("e2e_ms = %v, want > 0", got.E2EMS)
			}
			if (got.TTFTMS > 0) != tc.wantTTFT {
				t.Fatalf("ttft_ms = %v, want measured=%v", got.TTFTMS, tc.wantTTFT)
			}
			srv.metrics.inferenceMu.Lock()
			served := len(srv.metrics.inferReqs)
			srv.metrics.inferenceMu.Unlock()
			if served != 0 {
				t.Fatalf("failed turn was counted as a served generation (%d finish reasons)", served)
			}
			rep := decodePerfReport(t, getPerfRecent(t, srv, ""))
			if rep.Summary.Errors != 1 || rep.Summary.ByError[tc.wantClass] != 1 {
				t.Fatalf("summary errors = %d %v, want 1 %s", rep.Summary.Errors, rep.Summary.ByError, tc.wantClass)
			}
		})
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestPerfLedgerClientCancelLeavesOneRow(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = failingPlanner{block: true}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	if resp, err := postPerfChat(t, ctx, ts.URL, false); err == nil {
		resp.Body.Close()
		t.Fatalf("request completed with %d, want a client-side cancel", resp.StatusCode)
	}
	recs := waitPerfRows(t, srv, 1)
	if len(recs) != 1 {
		t.Fatalf("perf rows = %d, want exactly 1", len(recs))
	}
	if recs[0].Error != perfledger.ErrorClientCanceled || recs[0].Status != statusClientClosedRequest {
		t.Fatalf("row error/status = %q/%d, want %q/%d", recs[0].Error, recs[0].Status, perfledger.ErrorClientCanceled, statusClientClosedRequest)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestPerfErrorClassClosedVocabulary(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		ctx  context.Context
		err  error
		want string
	}{
		{canceled, errors.New("transport closed"), perfledger.ErrorClientCanceled},
		{context.Background(), fmt.Errorf("wrap: %w", context.Canceled), perfledger.ErrorClientCanceled},
		{context.Background(), agent.NewFirstTokenStalledError(time.Second), perfledger.ErrorFirstTokenTimeout},
		{context.Background(), &agent.UpstreamStalledError{Kind: "idle"}, perfledger.ErrorStall},
		{context.Background(), fmt.Errorf("wrap: %w", context.DeadlineExceeded), perfledger.ErrorDeadline},
		{context.Background(), &agent.UpstreamUnreachableError{}, perfledger.ErrorUpstreamUnreachable},
		{context.Background(), &agent.UpstreamStatusError{Status: 429}, perfledger.ErrorUpstreamStatus},
		{context.Background(), errors.New("boom"), perfledger.ErrorUpstream},
	}
	for _, c := range cases {
		if got := perfErrorClass(c.ctx, c.err); got != c.want {
			t.Errorf("perfErrorClass(%v) = %q, want %q", c.err, got, c.want)
		}
	}
}

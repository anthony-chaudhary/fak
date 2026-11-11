package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	s := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	cases := map[float64]float64{50: 50, 95: 100, 100: 100}
	for p, want := range cases {
		if got := percentile(s, p); got != want {
			t.Errorf("percentile(%v)=%v want %v", p, got, want)
		}
	}
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("empty percentile=%v want 0", got)
	}
	if got := percentile([]float64{42}, 95); got != 42 {
		t.Errorf("single percentile=%v want 42", got)
	}
}

func TestAggregate_TokensPerSecond(t *testing.T) {
	// 4 requests, each 10 completion tokens, wall = 2s => 40 tok / 2s = 20 tok/s.
	results := []reqResult{
		{latency: 100 * time.Millisecond, completionTokens: 10},
		{latency: 200 * time.Millisecond, completionTokens: 10},
		{latency: 300 * time.Millisecond, completionTokens: 10},
		{latency: 400 * time.Millisecond, completionTokens: 10},
	}
	pr := aggregate(2, 4, 2*time.Second, results)
	if pr.CompletionTokens != 40 {
		t.Fatalf("tokens=%d want 40", pr.CompletionTokens)
	}
	if pr.TokensPerSecond != 20 {
		t.Fatalf("tok/s=%v want 20", pr.TokensPerSecond)
	}
	if pr.ErrorRate != 0 {
		t.Fatalf("errrate=%v want 0", pr.ErrorRate)
	}
	if pr.P50LatencyMS != 200 || pr.P95LatencyMS != 400 {
		t.Fatalf("p50=%v p95=%v want 200/400", pr.P50LatencyMS, pr.P95LatencyMS)
	}
}

func TestAggregate_CountsErrors(t *testing.T) {
	results := []reqResult{
		{latency: 100 * time.Millisecond, completionTokens: 5},
		{err: fmt.Errorf("boom"), latency: 50 * time.Millisecond},
		{err: fmt.Errorf("boom"), latency: 50 * time.Millisecond},
		{latency: 100 * time.Millisecond, completionTokens: 5},
	}
	pr := aggregate(4, 4, time.Second, results)
	if pr.Errors != 2 || pr.ErrorRate != 0.5 {
		t.Fatalf("errors=%d rate=%v want 2/0.5", pr.Errors, pr.ErrorRate)
	}
	if pr.CompletionTokens != 10 {
		t.Fatalf("tokens=%d want 10 (errors excluded)", pr.CompletionTokens)
	}
}

func TestRun_AgainstStubServer(t *testing.T) {
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hits, 1)
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("missing content-type")
		}
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := chatResponse{}
		resp.Choices = []struct {
			Message chatMessage `json:"message"`
		}{{Message: chatMessage{Role: "assistant", Content: "ok"}}}
		resp.Usage.CompletionTokens = 7
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Stack: "test",
		Prompt: "hi", MaxTokens: 8, Concurrencies: []int{1, 4}, RequestsPer: 8,
	}
	mr := Run(context.Background(), cfg)
	if len(mr.Points) != 2 {
		t.Fatalf("points=%d want 2", len(mr.Points))
	}
	for _, p := range mr.Points {
		if p.Errors != 0 {
			t.Errorf("c=%d errors=%d", p.Concurrency, p.Errors)
		}
		if p.CompletionTokens != 8*7 {
			t.Errorf("c=%d tokens=%d want 56", p.Concurrency, p.CompletionTokens)
		}
		if p.TokensPerSecond <= 0 {
			t.Errorf("c=%d tok/s=%v want >0", p.Concurrency, p.TokensPerSecond)
		}
	}
	if got := atomic.LoadInt64(&hits); got != 16 {
		t.Fatalf("total hits=%d want 16", got)
	}
}

func TestRun_HandlesServerErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "upstream boom", http.StatusBadGateway)
	}))
	defer srv.Close()
	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "x",
		Prompt: "hi", Concurrencies: []int{2}, RequestsPer: 4,
	}
	mr := Run(context.Background(), cfg)
	if mr.Points[0].Errors != 4 || mr.Points[0].ErrorRate != 1.0 {
		t.Fatalf("expected all 4 errors, got %+v", mr.Points[0])
	}
}

func TestParseInts(t *testing.T) {
	got, err := parseInts("1, 4 ,16,")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 4, 16}
	if len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 16 {
		t.Fatalf("got %v want %v", got, want)
	}
	if _, err := parseInts("abc"); err == nil {
		t.Fatal("expected error")
	}
}

// fak-test:runtime fast est=1s
func TestRun_ReportsObservedHTTPOverlapAndTokens(t *testing.T) {
	const concurrency = 4
	entered := make(chan struct{}, concurrency)
	release := make(chan struct{})
	var releaseOnce sync.Once

	var serverInFlight atomic.Int64
	var serverPeak atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		inFlight := serverInFlight.Add(1)
		defer serverInFlight.Add(-1)
		for {
			peak := serverPeak.Load()
			if inFlight <= peak || serverPeak.CompareAndSwap(peak, inFlight) {
				break
			}
		}
		entered <- struct{}{}
		<-release

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":7}}`)
	}))
	defer srv.Close()
	defer releaseOnce.Do(func() { close(release) })

	cfg := &Config{
		URL: srv.URL, Model: "stub", Prompt: "hi", MaxTokens: 8,
		Concurrencies: []int{concurrency}, RequestsPer: concurrency,
	}
	done := make(chan MatrixResult, 1)
	go func() { done <- Run(context.Background(), cfg) }()

	testCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	for i := 0; i < concurrency; i++ {
		select {
		case <-entered:
		case <-testCtx.Done():
			t.Fatalf("waiting for request %d/%d to overlap: %v", i+1, concurrency, testCtx.Err())
		}
	}
	releaseOnce.Do(func() { close(release) })

	var got MatrixResult
	select {
	case got = <-done:
	case <-testCtx.Done():
		t.Fatalf("waiting for loadgen: %v", testCtx.Err())
	}
	if len(got.Points) != 1 {
		t.Fatalf("points=%d want 1", len(got.Points))
	}
	point := got.Points[0]
	if peak := int(serverPeak.Load()); peak != concurrency {
		t.Fatalf("server peak in-flight=%d want %d", peak, concurrency)
	}
	if point.PeakInFlight != concurrency {
		t.Fatalf("reported peak in-flight=%d want %d", point.PeakInFlight, concurrency)
	}
	if point.CompletionTokens != concurrency*7 || point.Errors != 0 {
		t.Fatalf("tokens/errors=%d/%d want %d/0", point.CompletionTokens, point.Errors, concurrency*7)
	}
}

// fak-test:runtime fast est=1s
func TestValidateConfigRejectsInvalidConcurrencyBeforeNetwork(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want error
	}{
		{name: "zero", cfg: Config{Concurrencies: []int{0}, RequestsPer: 1}, want: errNonPositiveConcurrency},
		{name: "negative", cfg: Config{Concurrencies: []int{-1}, RequestsPer: 1}, want: errNonPositiveConcurrency},
		{name: "requests below maximum", cfg: Config{Concurrencies: []int{1, 4, 2}, RequestsPer: 3}, want: errRequestsBelowConcurrency},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var networkCalls atomic.Int64
			tt.cfg.HTTPClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				networkCalls.Add(1)
				return nil, errors.New("unexpected network call")
			})}
			if err := validateConfig(&tt.cfg); !errors.Is(err, tt.want) {
				t.Fatalf("validateConfig error=%v want category %v", err, tt.want)
			}
			if calls := networkCalls.Load(); calls != 0 {
				t.Fatalf("network calls=%d want 0", calls)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestParseIntsPreservesOrderedDuplicates(t *testing.T) {
	got, err := parseInts("1,2,4,2,1")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2, 4, 2, 1}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

// fak-test:runtime fast est=2s
func TestRun_CanceledContextDoesNotEmitLaterPoints(t *testing.T) {
	requestEntered := make(chan struct{}, 1)
	releaseRequest := make(chan struct{})
	var releaseOnce sync.Once
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		requestEntered <- struct{}{}
		<-releaseRequest
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":1}}`)
	}))
	defer srv.Close()
	defer releaseOnce.Do(func() { close(releaseRequest) })

	ctx, cancel := context.WithCancel(context.Background())
	cfg := &Config{
		URL: srv.URL, Model: "stub", Prompt: "hi",
		Concurrencies: []int{1, 2, 4}, RequestsPer: 4,
	}
	done := make(chan MatrixResult, 1)
	go func() { done <- Run(ctx, cfg) }()

	testCtx, testCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer testCancel()
	select {
	case <-requestEntered:
		cancel()
		releaseOnce.Do(func() { close(releaseRequest) })
	case <-testCtx.Done():
		t.Fatalf("waiting for first request: %v", testCtx.Err())
	}

	var got MatrixResult
	select {
	case got = <-done:
	case <-testCtx.Done():
		t.Fatalf("waiting for canceled run: %v", testCtx.Err())
	}
	if len(got.Points) > 1 {
		t.Fatalf("canceled run emitted %d points, want at most current point", len(got.Points))
	}
	if !got.Canceled {
		t.Fatal("canceled run did not report canceled=true")
	}
	if gotHits := hits.Load(); gotHits != 1 {
		t.Fatalf("requests after cancellation=%d want only the in-flight request", gotHits)
	}
}

// fak-test:runtime fast est=1s
func TestRun_BodyReadErrorCannotCountTokens(t *testing.T) {
	var closed atomic.Bool
	body := &validJSONThenErrorBody{
		data:   []byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":37}}`),
		closed: &closed,
	}
	cfg := &Config{
		URL: "http://loadgen.test/v1/chat/completions", Model: "stub", Prompt: "hi",
		Concurrencies: []int{1}, RequestsPer: 1,
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       body,
			}, nil
		})},
	}

	got := Run(context.Background(), cfg)
	if len(got.Points) != 1 {
		t.Fatalf("points=%d want 1", len(got.Points))
	}
	point := got.Points[0]
	if point.Errors != 1 || point.CompletionTokens != 0 {
		t.Fatalf("errors/tokens=%d/%d want 1/0", point.Errors, point.CompletionTokens)
	}
	if point.PeakInFlight != 1 {
		t.Fatalf("peak in-flight=%d want 1", point.PeakInFlight)
	}
	if !closed.Load() {
		t.Fatal("response body was not closed")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type validJSONThenErrorBody struct {
	data   []byte
	offset int
	closed *atomic.Bool
}

func (b *validJSONThenErrorBody) Read(p []byte) (int, error) {
	if b.offset < len(b.data) {
		n := copy(p, b.data[b.offset:])
		b.offset += n
		return n, nil
	}
	return 0, io.ErrUnexpectedEOF
}

func (b *validJSONThenErrorBody) Close() error {
	b.closed.Store(true)
	return nil
}

// sleepStep scripts one scripted sleep completion: the simulated clock lands on
// `advance`; ok=false means the context finishes during the sleep.
type sleepStep struct {
	advance time.Time
	ok      bool
}

// pacingSim is a deterministic fake clock + cancellable context driving pacer.run.
type pacingSim struct {
	now               time.Time
	steps             []sleepStep
	step              int
	issueCost         time.Duration
	cancelAfterIssues int
	rejectAfter       int // once this many issues are accepted, feeds are refused (0 = never)
	slept             []time.Duration
	issuedAt          []time.Time
	ctx               context.Context
	cancel            context.CancelFunc
}

func newPacingSim(start time.Time) *pacingSim {
	ctx, cancel := context.WithCancel(context.Background())
	return &pacingSim{now: start, ctx: ctx, cancel: cancel}
}

func (s *pacingSim) nowFn() time.Time { return s.now }

func (s *pacingSim) sleepFn(_ context.Context, d time.Duration) bool {
	s.slept = append(s.slept, d)
	if s.step < len(s.steps) {
		st := s.steps[s.step]
		s.step++
		if !st.ok {
			s.cancel()
			return false
		}
		s.now = st.advance
		return true
	}
	s.now = s.now.Add(d)
	return true
}

func (s *pacingSim) issueFn() bool {
	if s.rejectAfter > 0 && len(s.issuedAt) >= s.rejectAfter {
		return false // handoff lost the race with ctx: job never reaches a worker
	}
	s.issuedAt = append(s.issuedAt, s.now)
	s.now = s.now.Add(s.issueCost)
	if s.cancelAfterIssues > 0 && len(s.issuedAt) >= s.cancelAfterIssues {
		s.cancel()
	}
	return true
}

func assertTimes(t *testing.T, what string, got []time.Time, want []time.Time) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s=%v want %v", what, got, want)
	}
	for i := range want {
		if !got[i].Equal(want[i]) {
			t.Errorf("%s[%d]=%s want %s", what, i, got[i], want[i])
		}
	}
}

func assertDurations(t *testing.T, what string, got []time.Duration, want []time.Duration) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s=%v want %v", what, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s[%d]=%s want %s", what, i, got[i], want[i])
		}
	}
}

// fak-test:runtime fast est=1s
func TestPacingCatchUp(t *testing.T) {
	// Validation sentinels: closed contracts via errors.Is, never prose.
	if err := validateConfig(&Config{RatePerSec: -1}); err == nil || !errors.Is(err, errNegativeRate) {
		t.Fatalf("validateConfig(RatePerSec:-1) error=%v want errNegativeRate", err)
	}
	if err := validateConfig(&Config{MaxCatchUp: -time.Millisecond}); err == nil || !errors.Is(err, errNegativeMaxCatchUp) {
		t.Fatalf("validateConfig(MaxCatchUp:-1ms) error=%v want errNegativeMaxCatchUp", err)
	}
	if err := validateConfig(&Config{RatePerSec: math.NaN()}); err == nil || !errors.Is(err, errNonFiniteRate) {
		t.Fatalf("validateConfig(RatePerSec:NaN) error=%v want errNonFiniteRate", err)
	}
	if err := validateConfig(&Config{RatePerSec: math.Inf(1)}); err == nil || !errors.Is(err, errNonFiniteRate) {
		t.Fatalf("validateConfig(RatePerSec:+Inf) error=%v want errNonFiniteRate", err)
	}
	if err := validateConfig(&Config{RatePerSec: 10, MaxCatchUp: 20 * time.Millisecond}); err != nil {
		t.Fatalf("validateConfig(valid pacing) error=%v", err)
	}

	const (
		interval = 100 * time.Millisecond
		catchUp  = 20 * time.Millisecond
	)
	start := time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)

	run := func(name string, sim *pacingSim, wantIssued int, wantSlept []time.Duration, wantIssuedAt []time.Time) {
		t.Run(name, func(t *testing.T) {
			p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
			got := p.run(sim.ctx, 3, sim.issueFn)
			if got != wantIssued {
				t.Fatalf("issued=%d want %d", got, wantIssued)
			}
			assertDurations(t, "slept", sim.slept, wantSlept)
			assertTimes(t, "issuedAt", sim.issuedAt, wantIssuedAt)
		})
	}

	// Sub-window oversleeps stay on the original absolute schedule (no re-anchor).
	sim := newPacingSim(start)
	sim.steps = []sleepStep{
		{advance: start.Add(105 * time.Millisecond), ok: true},
		{advance: start.Add(210 * time.Millisecond), ok: true},
		{advance: start.Add(305 * time.Millisecond), ok: true},
	}
	run("subWindowOversleepStaysOnSchedule", sim, 3,
		[]time.Duration{100 * time.Millisecond, 95 * time.Millisecond, 90 * time.Millisecond},
		[]time.Time{start.Add(105 * time.Millisecond), start.Add(210 * time.Millisecond), start.Add(305 * time.Millisecond)})

	// A stall past the catch-up window re-anchors to now: the missed tick is
	// dropped (no burst catch-up), the next tick issues immediately at now, and
	// the schedule resumes from there.
	sim = newPacingSim(start)
	sim.steps = []sleepStep{
		{advance: start.Add(100 * time.Millisecond), ok: true},
		{advance: start.Add(350 * time.Millisecond), ok: true},
	}
	run("pastWindowReanchorsWithoutBurst", sim, 3,
		[]time.Duration{100 * time.Millisecond, 100 * time.Millisecond},
		[]time.Time{start.Add(100 * time.Millisecond), start.Add(350 * time.Millisecond), start.Add(350 * time.Millisecond)})

	// The deadline advances BEFORE issuing, so issue latency cannot shift
	// subsequent deadlines (exact sleeps, 80ms issue cost).
	sim = newPacingSim(start)
	sim.issueCost = 80 * time.Millisecond
	run("advanceBeforeIssueDoesNotCompound", sim, 3,
		[]time.Duration{100 * time.Millisecond, 20 * time.Millisecond, 20 * time.Millisecond},
		[]time.Time{start.Add(100 * time.Millisecond), start.Add(200 * time.Millisecond), start.Add(300 * time.Millisecond)})

	// Cancellation during the first sleep stops feeding with zero issues.
	sim = newPacingSim(start)
	sim.steps = []sleepStep{{advance: start.Add(100 * time.Millisecond), ok: false}}
	run("cancelInFirstSleepIssuesNothing", sim, 0,
		[]time.Duration{100 * time.Millisecond}, nil)

	// Cancellation between ticks stops right after the last issued request.
	sim = newPacingSim(start)
	sim.cancelAfterIssues = 1
	run("cancelBetweenTicks", sim, 1,
		[]time.Duration{100 * time.Millisecond},
		[]time.Time{start.Add(100 * time.Millisecond)})

	// A feed that loses the handoff race with ctx is not counted as issued:
	// callers slice results to the returned count, so counting a rejected feed
	// would report a phantom request no worker ever consumed.
	sim = newPacingSim(start)
	sim.rejectAfter = 2
	run("rejectedFeedNotCounted", sim, 2,
		[]time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 100 * time.Millisecond},
		[]time.Time{start.Add(100 * time.Millisecond), start.Add(200 * time.Millisecond)})
}

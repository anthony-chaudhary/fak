package routerfront

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func backend(t *testing.T, name string) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz", "/readyz":
			w.WriteHeader(http.StatusOK)
		default:
			io.WriteString(w, name)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func get(t *testing.T, url string) (int, string, http.Header) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), resp.Header
}

func errType(t *testing.T, body string) string {
	t.Helper()
	var v struct {
		Error struct{ Type string } `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &v); err != nil {
		t.Fatalf("decode %q: %v", body, err)
	}
	return v.Error.Type
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition not reached")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func (f *Front) waitingCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.waiting
}

// fak-test:runtime fast est=10ms
func TestCheckLoopbackAddr(t *testing.T) {
	for _, ok := range []string{"127.0.0.1:8080", "127.9.9.9:1", "[::1]:80", "localhost:80"} {
		if err := CheckLoopbackAddr(ok); err != nil {
			t.Errorf("CheckLoopbackAddr(%q) = %v, want nil", ok, err)
		}
	}
	for _, bad := range []string{":8080", "0.0.0.0:80", "10.0.0.1:80", "[::]:80", "example.com:80"} {
		if err := CheckLoopbackAddr(bad); !errors.Is(err, ErrNonLoopbackFront) {
			t.Errorf("CheckLoopbackAddr(%q) = %v, want ErrNonLoopbackFront", bad, err)
		}
	}
}

// fak-test:runtime fast est=100ms
func TestSSEFlushesBeforeBodyEnds(t *testing.T) {
	release := make(chan struct{})
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		<-release
		io.WriteString(w, "data: last\n\n")
	}))
	defer b.Close()
	defer close(release)
	f := New(Options{})
	if err := f.Promote(b.URL); err != nil {
		t.Fatal(err)
	}
	fs := httptest.NewServer(f)
	defer fs.Close()
	resp, err := http.Get(fs.URL + "/v1/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || line != "data: first\n" {
		t.Fatalf("first line = %q, %v", line, err)
	}
	if n := f.InFlight(b.URL); n != 1 {
		t.Fatalf("InFlight during stream = %d, want 1", n)
	}
}

// fak-test:runtime fast est=50ms
func TestHoldThenPromoteDeliversToNewBackend(t *testing.T) {
	a, b := backend(t, "A"), backend(t, "B")
	f := New(Options{})
	f.Promote(a.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	f.Hold()
	got := make(chan string, 1)
	go func() { _, body, _ := get(t, fs.URL+"/x"); got <- body }()
	waitFor(t, func() bool { return f.waitingCount() == 1 })
	f.Promote(b.URL)
	if body := <-got; body != "B" {
		t.Fatalf("held request served by %q, want B", body)
	}
}

// fak-test:runtime fast est=100ms
func TestUnpromotedHoldTimesOut503(t *testing.T) {
	f := New(Options{HoldTimeout: 30 * time.Millisecond})
	if _, err := f.acquire(context.Background()); !errors.Is(err, ErrHoldTimeout) {
		t.Fatalf("acquire = %v, want ErrHoldTimeout", err)
	}
	fs := httptest.NewServer(f)
	defer fs.Close()
	code, body, h := get(t, fs.URL+"/x")
	if code != http.StatusServiceUnavailable || h.Get("Retry-After") == "" {
		t.Fatalf("code=%d retry-after=%q", code, h.Get("Retry-After"))
	}
	if typ := errType(t, body); typ != "router_unavailable" {
		t.Fatalf("type = %q", typ)
	}
	a := backend(t, "A")
	f.Promote(a.URL)
	f.Hold()
	code, body, _ = get(t, fs.URL+"/x")
	if code != http.StatusServiceUnavailable || errType(t, body) != "router_swapping" {
		t.Fatalf("held: code=%d body=%s", code, body)
	}
}

// fak-test:runtime fast est=100ms
func TestHeldClientDisconnectDoesNotLeak(t *testing.T) {
	f := New(Options{})
	fs := httptest.NewServer(f)
	defer fs.Close()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, fs.URL+"/x", nil)
	done := make(chan struct{})
	go func() { http.DefaultClient.Do(req); close(done) }()
	waitFor(t, func() bool { return f.waitingCount() == 1 })
	cancel()
	<-done
	waitFor(t, func() bool { return f.waitingCount() == 0 })
}

// fak-test:runtime fast est=50ms
func TestForwardedHeadersStripped(t *testing.T) {
	seen := make(chan http.Header, 1)
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { seen <- r.Header.Clone() }))
	defer b.Close()
	f := New(Options{})
	f.Promote(b.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	req, _ := http.NewRequest(http.MethodGet, fs.URL+"/x", nil)
	req.Header.Set("X-Forwarded-For", "6.6.6.6")
	req.Header.Set("Forwarded", "for=6.6.6.6")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	h := <-seen
	for _, k := range []string{"X-Forwarded-For", "Forwarded", "X-Forwarded-Host", "X-Forwarded-Proto"} {
		if _, ok := h[k]; ok {
			t.Errorf("backend saw %s = %q", k, h.Get(k))
		}
	}
}

// fak-test:runtime fast est=100ms
func TestDeadBackendHoldsThenTyped503(t *testing.T) {
	dead := httptest.NewServer(http.NotFoundHandler())
	url := dead.URL
	dead.Close()
	f := New(Options{HoldTimeout: 50 * time.Millisecond, ProbeInterval: 5 * time.Millisecond})
	f.Promote(url)
	fs := httptest.NewServer(f)
	defer fs.Close()
	code, body, _ := get(t, fs.URL+"/x")
	if code != http.StatusServiceUnavailable || errType(t, body) != "router_unavailable" {
		t.Fatalf("code=%d body=%s", code, body)
	}
	if f.InFlight(url) != 0 {
		t.Fatal("in-flight leaked after a held dial failure")
	}
}

type countingBackend struct {
	hits   atomic.Int64
	bodies chan string
}

func newCountingBackend(t *testing.T) (*httptest.Server, *countingBackend) {
	t.Helper()
	c := &countingBackend{bodies: make(chan string, 16)}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			return
		}
		b, _ := io.ReadAll(r.Body)
		c.hits.Add(1)
		c.bodies <- string(b)
		io.WriteString(w, "NEW")
	}))
	t.Cleanup(s.Close)
	return s, c
}

// fak-test:runtime fast est=100ms
func TestDialRefusedHeldAndDeliveredOnceAfterPromote(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		dead := httptest.NewServer(http.NotFoundHandler())
		deadURL := dead.URL
		dead.Close()
		f := New(Options{HoldTimeout: 5 * time.Second})
		f.Promote(deadURL)
		fs := httptest.NewServer(f)
		var body io.Reader = strings.NewReader("payload")
		if chunked {
			body = io.MultiReader(strings.NewReader("pay"), strings.NewReader("load"))
		}
		type result struct {
			code int
			body string
			err  error
		}
		done := make(chan result, 1)
		go func() {
			resp, err := http.Post(fs.URL+"/v1/x", "text/plain", body)
			if err != nil {
				done <- result{err: err}
				return
			}
			defer resp.Body.Close()
			b, _ := io.ReadAll(resp.Body)
			done <- result{code: resp.StatusCode, body: string(b)}
		}()
		waitFor(t, func() bool { return f.waitingCount() == 1 })
		nw, c := newCountingBackend(t)
		if err := f.Promote(nw.URL); err != nil {
			t.Fatal(err)
		}
		res := <-done
		if res.err != nil || res.code != http.StatusOK || res.body != "NEW" {
			t.Fatalf("chunked=%t: %+v", chunked, res)
		}
		if got := <-c.bodies; got != "payload" || c.hits.Load() != 1 {
			t.Fatalf("chunked=%t: backend saw body %q hits %d", chunked, got, c.hits.Load())
		}
		fs.Close()
	}
}

// fak-test:runtime fast est=50ms
func TestErrorAfterWriteIs502NotReplayed(t *testing.T) {
	var hits atomic.Int64
	be := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The dropped request makes the front probe /healthz before it
		// releases the hold; that probe is not a replay of the request.
		if r.URL.Path == "/healthz" {
			return
		}
		_, _ = io.ReadAll(r.Body)
		hits.Add(1)
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close()
		}
	}))
	defer be.Close()
	f := New(Options{HoldTimeout: 5 * time.Second})
	f.Promote(be.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	resp, err := http.Post(fs.URL+"/v1/x", "text/plain", strings.NewReader("payload"))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway || errType(t, string(b)) != "router_backend_unreachable" {
		t.Fatalf("code=%d body=%s", resp.StatusCode, b)
	}
	if hits.Load() != 1 {
		t.Fatalf("backend executed the request %d times, want 1", hits.Load())
	}
	if f.unavailable() {
		t.Fatal("an error after write must not hold the front")
	}
}

// fak-test:runtime fast est=200ms
func TestOversizeBodyNotReplayed(t *testing.T) {
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close()
	f := New(Options{HoldTimeout: 5 * time.Second})
	f.Promote(deadURL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	big := strings.Repeat("x", MaxReplayBody+1)
	for _, chunked := range []bool{false, true} {
		var body io.Reader = strings.NewReader(big)
		if chunked {
			body = io.MultiReader(strings.NewReader(big))
		}
		start := time.Now()
		resp, err := http.Post(fs.URL+"/v1/x", "text/plain", body)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadGateway || time.Since(start) > 3*time.Second {
			t.Fatalf("chunked=%t: code=%d after %s body=%s", chunked, resp.StatusCode, time.Since(start), b)
		}
	}
	if f.unavailable() {
		t.Fatal("an unreplayable request must not hold the front")
	}
}

// fak-test:runtime fast est=300ms
func TestSwapNeverPromotesMismatchedGeneration(t *testing.T) {
	old := backend(t, "OLD")
	var healthGen, readyGen atomic.Value
	healthGen.Store("other")
	readyGen.Store("other")
	nw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g := &healthGen
		if r.URL.Path == "/readyz" {
			g = &readyGen
		}
		w.Header().Set(GenerationHeader, g.Load().(string))
		io.WriteString(w, "NEW")
	}))
	defer nw.Close()
	f := New(Options{ProbeInterval: 5 * time.Millisecond})
	f.Promote(old.URL)
	plan := SwapPlan{New: nw.URL, Generation: "want", HealthTimeout: 100 * time.Millisecond,
		ReadyTimeout: 100 * time.Millisecond, OldAlive: func() bool { return true }}
	rec, err := f.Swap(context.Background(), plan)
	if !errors.Is(err, ErrNotReady) || rec.Outcome != OutcomeAbortedResumed || f.Active() != old.URL {
		t.Fatalf("mismatched health: %+v %v active=%s", rec, err, f.Active())
	}
	healthGen.Store("want")
	rec, err = f.Swap(context.Background(), plan)
	if !errors.Is(err, ErrNotReady) || rec.Outcome != OutcomeAbortedResumed || f.Active() != old.URL {
		t.Fatalf("mismatched readiness: %+v %v active=%s", rec, err, f.Active())
	}
	readyGen.Store("want")
	rec, err = f.Swap(context.Background(), plan)
	if err != nil || rec.Outcome != OutcomePromoted || f.Active() != nw.URL {
		t.Fatalf("matching generation: %+v %v", rec, err)
	}
}

// fak-test:runtime fast est=50ms
func TestPanicRecovered(t *testing.T) {
	f := New(Options{})
	f.Promote("http://127.0.0.1:1")
	f.proxies["http://127.0.0.1:1"] = nil
	rec := httptest.NewRecorder()
	f.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("code = %d", rec.Code)
	}
	if f.InFlight("http://127.0.0.1:1") != 0 {
		t.Fatal("in-flight leaked after panic")
	}
}

// fak-test:runtime fast est=300ms
func TestSwapHappyPath(t *testing.T) {
	slowStarted, slowRelease := make(chan struct{}), make(chan struct{})
	old := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/slow" {
			close(slowStarted)
			<-slowRelease
		}
		io.WriteString(w, "OLD")
	}))
	defer old.Close()
	var ready atomic.Bool
	nw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" && !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		io.WriteString(w, "NEW")
	}))
	defer nw.Close()

	f := New(Options{ProbeInterval: 5 * time.Millisecond})
	f.Promote(old.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()

	slowBody := make(chan string, 1)
	go func() { _, b, _ := get(t, fs.URL+"/slow"); slowBody <- b }()
	<-slowStarted

	heldBody := make(chan string, 1)
	stopped := make(chan struct{})
	plan := SwapPlan{
		New: nw.URL,
		StopOld: func(context.Context) error {
			close(stopped)
			ready.Store(true)
			return nil
		},
		OldAlive: func() bool { return false },
	}
	recCh := make(chan SwapReceipt, 1)
	errCh := make(chan error, 1)
	go func() { r, err := f.Swap(context.Background(), plan); recCh <- r; errCh <- err }()

	waitFor(t, func() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.held })
	go func() { _, b, _ := get(t, fs.URL+"/during"); heldBody <- b }()
	waitFor(t, func() bool { return f.waitingCount() == 1 })
	select {
	case <-stopped:
		t.Fatal("StopOld ran before old in-flight drained")
	default:
	}
	close(slowRelease)

	if b := <-slowBody; b != "OLD" {
		t.Fatalf("in-flight request finished on %q, want OLD", b)
	}
	if b := <-heldBody; b != "NEW" {
		t.Fatalf("request during swap served by %q, want NEW", b)
	}
	rec, err := <-recCh, <-errCh
	if err != nil {
		t.Fatal(err)
	}
	if rec.Outcome != OutcomePromoted || rec.Old != old.URL || rec.New != nw.URL ||
		rec.InFlightAtHold != 1 || rec.HeldRequests != 1 || !rec.DrainedClean {
		t.Fatalf("receipt = %+v", rec)
	}
	if f.Active() != nw.URL {
		t.Fatalf("Active = %s", f.Active())
	}
}

// fak-test:runtime fast est=200ms
func TestSwapReadinessFailureResumesOld(t *testing.T) {
	old := backend(t, "OLD")
	nw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer nw.Close()
	f := New(Options{ProbeInterval: 5 * time.Millisecond})
	f.Promote(old.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	rec, err := f.Swap(context.Background(), SwapPlan{
		New: nw.URL, ReadyTimeout: 50 * time.Millisecond,
		OldAlive: func() bool { return true },
	})
	if !errors.Is(err, ErrNotReady) || rec.Outcome != OutcomeAbortedResumed {
		t.Fatalf("Swap = %+v, %v", rec, err)
	}
	if _, body, _ := get(t, fs.URL+"/x"); body != "OLD" || f.Active() != old.URL {
		t.Fatalf("after rollback body=%q active=%s", body, f.Active())
	}
}

// fak-test:runtime fast est=200ms
func TestSwapReadinessFailureOldDeadFails503(t *testing.T) {
	old := backend(t, "OLD")
	nw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer nw.Close()
	f := New(Options{ProbeInterval: 5 * time.Millisecond, HoldTimeout: 30 * time.Millisecond})
	f.Promote(old.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	rec, err := f.Swap(context.Background(), SwapPlan{
		New: nw.URL, ReadyTimeout: 50 * time.Millisecond,
		OldAlive: func() bool { return false },
	})
	if !errors.Is(err, ErrNotReady) || rec.Outcome != OutcomeAbortedHeldFailed {
		t.Fatalf("Swap = %+v, %v", rec, err)
	}
	code, body, _ := get(t, fs.URL+"/x")
	if code != http.StatusServiceUnavailable || errType(t, body) != "router_unavailable" {
		t.Fatalf("code=%d body=%s", code, body)
	}
}

// fak-test:runtime fast est=100ms
func TestSwapHealthFailureNeverHolds(t *testing.T) {
	old := backend(t, "OLD")
	sick := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer sick.Close()
	f := New(Options{ProbeInterval: 5 * time.Millisecond})
	f.Promote(old.URL)
	var stopCalled atomic.Bool
	_, err := f.Swap(context.Background(), SwapPlan{
		New: sick.URL, HealthTimeout: 30 * time.Millisecond,
		StopOld: func(context.Context) error { stopCalled.Store(true); return nil },
	})
	if !errors.Is(err, ErrNotReady) {
		t.Fatalf("Swap err = %v, want ErrNotReady", err)
	}
	f.mu.Lock()
	held := f.held
	f.mu.Unlock()
	if held || stopCalled.Load() || f.Active() != old.URL {
		t.Fatalf("held=%v stopCalled=%v active=%s", held, stopCalled.Load(), f.Active())
	}
}

// fak-test:runtime fast est=300ms
func TestConcurrentTrafficAcrossPromotes(t *testing.T) {
	a, b := backend(t, "A"), backend(t, "B")
	f := New(Options{})
	f.Promote(a.URL)
	fs := httptest.NewServer(f)
	defer fs.Close()
	errs := make(chan error, 64)
	for i := 0; i < 64; i++ {
		go func(i int) {
			resp, err := http.Get(fs.URL + "/x")
			if err != nil {
				errs <- err
				return
			}
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if s := string(body); s != "A" && s != "B" {
				err = fmt.Errorf("request %d got %q", i, s)
			}
			errs <- err
		}(i)
		if i%8 == 0 {
			f.Hold()
			if i%16 == 0 {
				f.Promote(b.URL)
			} else {
				f.Resume()
			}
		}
	}
	for i := 0; i < 64; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if err := f.WaitIdle(context.Background(), a.URL); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=1ms
func TestGenerationMatchesAcceptsLegacyRefusesForeign(t *testing.T) {
	h := http.Header{}
	if !GenerationMatches(h, "g1") {
		t.Fatal("backend without the header must be accepted")
	}
	h.Set(GenerationHeader, "g2")
	if GenerationMatches(h, "g1") {
		t.Fatal("backend echoing another generation must be refused")
	}
	h.Set(GenerationHeader, "g1")
	if !GenerationMatches(h, "g1") {
		t.Fatal("matching generation must be accepted")
	}
}

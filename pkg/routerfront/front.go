// Package routerfront is the router hot-failover front door: a stable loopback
// reverse proxy to the active router generation that can hold, swap and drain
// generations without dropping client requests (docs/ROUTER-HOT-FAILOVER.md).
package routerfront

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrNonLoopbackFront   = errors.New("routerfront: front door must listen on a loopback address")
	ErrNotReady           = errors.New("routerfront: backend not ready")
	ErrHoldTimeout        = errors.New("routerfront: request hold timed out")
	errBadBackend         = errors.New("routerfront: invalid backend URL")
	errGenerationMismatch = errors.New("routerfront: backend generation mismatch")
)

const (
	defaultHoldTimeout   = 30 * time.Second
	defaultProbeInterval = 100 * time.Millisecond
	retryAfterSeconds    = 1
	// MaxReplayBody bounds the request body the front buffers so a request
	// that never reached a crashed backend can be re-dispatched.
	MaxReplayBody = 8 << 20
	// GenerationHeader carries the per-child generation token a backend
	// echoes on its health and readiness surfaces.
	GenerationHeader = "X-Fak-Generation"
)

type Options struct {
	HoldTimeout   time.Duration
	ProbeInterval time.Duration
	Client        *http.Client
	Logf          func(string, ...any)
}

type Front struct {
	opts      Options
	transport http.RoundTripper
	swapMu    sync.Mutex

	mu        sync.Mutex
	active    string
	activeGen string
	held      bool
	crashHeld bool
	failed    bool
	release   chan struct{}
	waiting   int
	heldTotal int
	inflight  map[string]int
	changed   chan struct{}
	proxies   map[string]*httputil.ReverseProxy
}

// CheckLoopbackAddr refuses any listen address that is not loopback; an empty
// host (":8080") binds every interface and is refused too.
func CheckLoopbackAddr(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	if host == "localhost" {
		return nil
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return nil
	}
	return fmt.Errorf("%w: %q", ErrNonLoopbackFront, addr)
}

func New(opts Options) *Front {
	if opts.HoldTimeout <= 0 {
		opts.HoldTimeout = defaultHoldTimeout
	}
	if opts.ProbeInterval <= 0 {
		opts.ProbeInterval = defaultProbeInterval
	}
	if opts.Client == nil {
		opts.Client = &http.Client{Transport: &http.Transport{Proxy: nil}, Timeout: 2 * time.Second}
	}
	if opts.Logf == nil {
		opts.Logf = func(string, ...any) {}
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil
	dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
	tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		c, err := dialer.DialContext(ctx, network, addr)
		if err != nil {
			if a, ok := ctx.Value(attemptKey{}).(*attempt); ok {
				a.dialFailed.Store(true)
			}
		}
		return c, err
	}
	return &Front{
		opts:      opts,
		transport: tr,
		release:   make(chan struct{}),
		changed:   make(chan struct{}),
		inflight:  map[string]int{},
		proxies:   map[string]*httputil.ReverseProxy{},
	}
}

func (f *Front) Active() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *Front) Promote(backend string) error {
	_, err := f.promote(backend, "")
	return err
}

// PromoteGeneration promotes backend as the child carrying generation; the
// front's own crash self-heal probe then accepts only that generation.
func (f *Front) PromoteGeneration(backend, generation string) error {
	_, err := f.promote(backend, generation)
	return err
}

func (f *Front) promote(backend, generation string) (int, error) {
	if _, err := parseBackend(backend); err != nil {
		return 0, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active = backend
	f.activeGen = generation
	f.failed = false
	f.crashHeld = false
	return f.releaseLocked(), nil
}

func (f *Front) Hold() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.held {
		f.held = true
		f.heldTotal = 0
	}
}

func (f *Front) Resume() {
	f.resume()
}

func (f *Front) resume() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.releaseLocked()
}

func (f *Front) releaseLocked() int {
	n := f.heldTotal
	f.held = false
	f.crashHeld = false
	f.heldTotal = 0
	close(f.release)
	f.release = make(chan struct{})
	return n
}

func (f *Front) InFlight(backend string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.inflight[backend]
}

func (f *Front) WaitIdle(ctx context.Context, backend string) error {
	for {
		f.mu.Lock()
		n, ch := f.inflight[backend], f.changed
		f.mu.Unlock()
		if n == 0 {
			return nil
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (f *Front) ProbeReady(ctx context.Context, backend, path string) error {
	return f.ProbeReadyGeneration(ctx, backend, path, "")
}

// ProbeReadyGeneration polls path until it answers 200 and, when generation
// is set, echoes it in GenerationHeader; another child that happens to answer
// on the same port is never taken for the expected one.
func (f *Front) ProbeReadyGeneration(ctx context.Context, backend, path, generation string) error {
	u, err := parseBackend(backend)
	if err != nil {
		return err
	}
	target := u.JoinPath(path).String()
	var last error
	for {
		last = f.probeOnce(ctx, target, generation)
		if last == nil {
			return nil
		}
		t := time.NewTimer(f.opts.ProbeInterval)
		select {
		case <-ctx.Done():
			t.Stop()
			return fmt.Errorf("%w: %s: %v", ErrNotReady, target, last)
		case <-t.C:
		}
	}
}

func (f *Front) probeOnce(ctx context.Context, target, generation string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	resp, err := f.opts.Client.Do(req)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	if !GenerationMatches(resp.Header, generation) {
		return fmt.Errorf("%w: %s", errGenerationMismatch, target)
	}
	return nil
}

// GenerationMatches refuses a backend echoing a different generation. A
// backend that sends no header predates the token and is accepted, so a newer
// supervisor never strands an older router behind a hold.
func GenerationMatches(h http.Header, generation string) bool {
	got := h.Get(GenerationHeader)
	return generation == "" || got == "" || got == generation
}

func parseBackend(backend string) (*url.URL, error) {
	u, err := url.Parse(backend)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("%w: %q", errBadBackend, backend)
	}
	return u, nil
}

func (f *Front) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	sw := &statusWriter{ResponseWriter: w}
	defer func() {
		v := recover()
		if v == nil {
			return
		}
		if v == http.ErrAbortHandler {
			panic(v)
		}
		f.opts.Logf("routerfront: panic serving %s %s: %v", r.Method, r.URL.Path, v)
		if !sw.wrote {
			writeJSONError(sw, http.StatusInternalServerError, "router_internal", "internal front door error")
		}
	}()
	replay := prepareReplay(r)
	var deadline time.Time
	for {
		backend, err := f.acquire(r.Context())
		if err == nil && !deadline.IsZero() && time.Now().After(deadline) {
			f.done(backend)
			err = ErrHoldTimeout
		}
		if err != nil {
			if errors.Is(err, ErrHoldTimeout) {
				typ, msg := "router_swapping", "router generation swap in progress; retry shortly"
				if f.holdExpiredUnavailable() {
					typ, msg = "router_unavailable", "no router generation is ready"
				}
				sw.Header().Set("Retry-After", strconv.Itoa(retryAfterSeconds))
				writeJSONError(sw, http.StatusServiceUnavailable, typ, msg)
			}
			return
		}
		a := &attempt{replayable: replay != nil}
		if replay != nil {
			replay(r)
		}
		f.dispatch(sw, r.WithContext(context.WithValue(r.Context(), attemptKey{}, a)), backend)
		if !a.redispatch {
			return
		}
		f.opts.Logf("routerfront: backend %s refused the connection; holding request for the next generation", backend)
		f.crashHold(backend)
		if deadline.IsZero() {
			deadline = time.Now().Add(f.opts.HoldTimeout)
		}
	}
}

func (f *Front) dispatch(w http.ResponseWriter, r *http.Request, backend string) {
	defer f.done(backend)
	f.proxyFor(backend).ServeHTTP(w, r)
}

type attemptKey struct{}

// attempt tracks one dispatch. dialFailed is set by the transport's dialer, so
// a redispatch is decided from the dial itself, never from error text.
type attempt struct {
	replayable bool
	dialFailed atomic.Bool
	redispatch bool
}

// neverSent reports whether err proves the backend never received a byte of
// the request: our own dialer failed and the transport surfaced that dial
// error.
func (a *attempt) neverSent(err error) bool {
	var op *net.OpError
	return a.dialFailed.Load() && errors.As(err, &op) && op.Op == "dial"
}

// prepareReplay buffers a body of at most MaxReplayBody and returns a func
// that rewinds r before each dispatch; nil means r cannot be replayed.
func prepareReplay(r *http.Request) func(*http.Request) {
	if r.Body == nil || r.Body == http.NoBody || r.ContentLength == 0 && len(r.TransferEncoding) == 0 {
		return func(*http.Request) {}
	}
	if r.ContentLength > MaxReplayBody {
		return nil
	}
	buf, err := io.ReadAll(io.LimitReader(r.Body, MaxReplayBody+1))
	if err != nil || len(buf) > MaxReplayBody {
		r.Body = readCloser{io.MultiReader(bytes.NewReader(buf), errReader{err}, r.Body), r.Body}
		return nil
	}
	_ = r.Body.Close()
	r.TransferEncoding = nil
	r.ContentLength = int64(len(buf))
	getBody := func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(buf)), nil }
	return func(r *http.Request) {
		r.Body, _ = getBody()
		r.GetBody = getBody
	}
}

type readCloser struct {
	io.Reader
	io.Closer
}

type errReader struct{ err error }

func (e errReader) Read([]byte) (int, error) {
	if e.err != nil {
		return 0, e.err
	}
	return 0, io.EOF
}

// crashHold holds new requests when backend is still the active generation
// and refused a connection; the next Promote releases them. A probe resumes
// the same backend if it turns out to be alive, so a transient refusal cannot
// stall the front until the supervisor promotes again.
func (f *Front) crashHold(backend string) {
	f.holdForProbe(backend, true)
}

// suspectHold holds new requests after backend, still the active generation,
// failed a request the front had already sent (reset, EOF). That request is
// not replayed, but a killed child keeps resetting connections, pooled and
// fresh ones its dying listener still accepts, until the supervisor reaps it
// and promotes the restart; holding here keeps those later requests off the
// dying child so they are delivered to the restarted one, bounded by
// HoldTimeout. The backend is not marked failed: the same probe as crashHold
// releases the hold as soon as a live backend answers its health check.
func (f *Front) suspectHold(backend string) {
	f.holdForProbe(backend, false)
}

func (f *Front) holdForProbe(backend string, refused bool) {
	f.mu.Lock()
	if f.active != backend || f.crashHeld {
		if refused && f.active == backend {
			f.failed = true
		}
		f.mu.Unlock()
		return
	}
	if refused {
		f.failed = true
	}
	if !f.held {
		f.held = true
		f.heldTotal = 0
	}
	f.crashHeld = true
	gen := f.activeGen
	f.mu.Unlock()
	go f.selfHeal(backend, gen)
}

func (f *Front) selfHeal(backend, gen string) {
	ctx, cancel := context.WithTimeout(context.Background(), f.opts.HoldTimeout)
	defer cancel()
	if f.ProbeReadyGeneration(ctx, backend, "/healthz", gen) != nil {
		return
	}
	if !f.swapMu.TryLock() {
		return
	}
	defer f.swapMu.Unlock()
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.crashHeld && f.active == backend && f.activeGen == gen {
		f.failed = false
		f.releaseLocked()
	}
}

func (f *Front) unavailable() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active == "" || f.failed
}

// holdExpiredUnavailable reports whether a request whose hold expired should
// be told no generation is ready: none is promoted, the active one is marked
// failed, or it is still crash-held because it never answered its probe.
func (f *Front) holdExpiredUnavailable() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active == "" || f.failed || f.crashHeld
}

// acquire returns the backend to dispatch to with its in-flight count already
// taken, waiting (bounded by HoldTimeout and ctx) while held or unpromoted.
func (f *Front) acquire(ctx context.Context) (string, error) {
	var timer *time.Timer
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()
	for {
		f.mu.Lock()
		if !f.held && f.active != "" {
			b := f.active
			f.inflight[b]++
			f.mu.Unlock()
			return b, nil
		}
		ch := f.release
		f.waiting++
		if timer == nil {
			f.heldTotal++
			timer = time.NewTimer(f.opts.HoldTimeout)
		}
		f.mu.Unlock()

		var err error
		select {
		case <-ch:
		case <-timer.C:
			err = ErrHoldTimeout
		case <-ctx.Done():
			err = ctx.Err()
		}
		f.mu.Lock()
		f.waiting--
		f.mu.Unlock()
		if err != nil {
			return "", err
		}
	}
}

func (f *Front) done(backend string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inflight[backend]--
	if f.inflight[backend] <= 0 {
		delete(f.inflight, backend)
	}
	close(f.changed)
	f.changed = make(chan struct{})
}

func (f *Front) proxyFor(backend string) *httputil.ReverseProxy {
	f.mu.Lock()
	defer f.mu.Unlock()
	if p, ok := f.proxies[backend]; ok {
		return p
	}
	target, _ := parseBackend(backend)
	p := &httputil.ReverseProxy{
		Transport:     f.transport,
		FlushInterval: -1,
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			for _, h := range []string{"X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "Forwarded"} {
				pr.Out.Header.Del(h)
			}
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if r.Context().Err() != nil {
				return
			}
			a, _ := r.Context().Value(attemptKey{}).(*attempt)
			if a != nil && a.replayable && a.neverSent(err) {
				if sw, ok := w.(*statusWriter); !ok || !sw.wrote {
					a.redispatch = true
					return
				}
			}
			f.opts.Logf("routerfront: backend %s: %v", backend, err)
			if sw, ok := w.(*statusWriter); ok && sw.wrote {
				return
			}
			// A request the backend received and then dropped: hold before
			// answering, so the client's next request already waits for the
			// restarted generation instead of failing on the dying one too.
			// An unreplayable request refused at dial holds nothing.
			if a != nil && !a.neverSent(err) {
				f.suspectHold(backend)
			}
			writeJSONError(w, http.StatusBadGateway, "router_backend_unreachable", "active router generation is unreachable")
		},
	}
	f.proxies[backend] = p
	return p
}

func writeJSONError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"type": typ, "message": msg}})
}

type statusWriter struct {
	http.ResponseWriter
	wrote bool
}

func (s *statusWriter) WriteHeader(code int) {
	if code >= 200 {
		s.wrote = true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusWriter) Write(b []byte) (int, error) {
	s.wrote = true
	return s.ResponseWriter.Write(b)
}

func (s *statusWriter) Unwrap() http.ResponseWriter { return s.ResponseWriter }

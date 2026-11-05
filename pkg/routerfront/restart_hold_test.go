package routerfront

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// dyingGen is a fake router generation that is killed under load: requests
// that reached it before kill() are cut without a response, and its listener
// keeps accepting afterwards (as a SIGKILLed child's does until the kernel
// closes it) but resets every connection, health probes included.
type dyingGen struct {
	ln      net.Listener
	arrived chan struct{}
	killed  chan struct{}
	closing chan struct{}
	once    sync.Once
	reqs    atomic.Int64
	wg      sync.WaitGroup
}

func newDyingGen(t *testing.T) *dyingGen {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	g := &dyingGen{ln: ln, arrived: make(chan struct{}, 64), killed: make(chan struct{}), closing: make(chan struct{})}
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			g.wg.Add(1)
			// Cleanup also resets connections that never sent a request
			// (pooled keep-alives), so it cannot wait on a blocked read.
			stop := make(chan struct{})
			g.wg.Add(1)
			go func() {
				defer g.wg.Done()
				select {
				case <-g.closing:
					c.Close()
				case <-stop:
				}
			}()
			go func() {
				defer g.wg.Done()
				defer close(stop)
				defer c.Close()
				req, err := http.ReadRequest(bufio.NewReader(c))
				if err != nil {
					return
				}
				if req.URL.Path != "/healthz" && req.URL.Path != "/readyz" {
					g.reqs.Add(1)
					select {
					case g.arrived <- struct{}{}:
					default:
					}
				}
				<-g.killed
			}()
		}
	}()
	t.Cleanup(func() {
		g.kill()
		ln.Close()
		close(g.closing)
		g.wg.Wait()
	})
	return g
}

func (g *dyingGen) url() string { return "http://" + g.ln.Addr().String() }
func (g *dyingGen) kill()       { g.once.Do(func() { close(g.killed) }) }

type frontResult struct {
	code int
	body string
	err  error
}

func postAsync(url, payload string) <-chan frontResult {
	done := make(chan frontResult, 1)
	go func() {
		client := &http.Client{Transport: &http.Transport{Proxy: nil, DisableKeepAlives: true}}
		resp, err := client.Post(url, "application/json", strings.NewReader(payload))
		if err != nil {
			done <- frontResult{err: err}
			return
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		done <- frontResult{code: resp.StatusCode, body: string(b)}
	}()
	return done
}

// TestRestartHoldDeliversToRestartedGeneration covers a kill -9 of the active
// generation under load: the requests in flight on it fail with the typed
// 502, while a request arriving before the supervisor promotes the restart is
// held (not answered 502) and delivered exactly once to the restarted child.
//
// fak-test:runtime fast est=150ms
func TestRestartHoldDeliversToRestartedGeneration(t *testing.T) {
	dying := newDyingGen(t)
	f := New(Options{HoldTimeout: 5 * time.Second, ProbeInterval: 5 * time.Millisecond})
	if err := f.PromoteGeneration(dying.url(), "g1"); err != nil {
		t.Fatal(err)
	}
	fs := httptest.NewServer(f)
	defer fs.Close()

	const inFlight = 3
	cut := make([]<-chan frontResult, inFlight)
	for i := range cut {
		cut[i] = postAsync(fs.URL+"/v1/chat/completions", `{"id":"inflight-`+strconv.Itoa(i)+`"}`)
	}
	for i := 0; i < inFlight; i++ {
		select {
		case <-dying.arrived:
		case <-time.After(5 * time.Second):
			t.Fatal("in-flight requests never reached the generation")
		}
	}
	dying.kill()
	for i, ch := range cut {
		res := <-ch
		if res.err != nil || res.code != http.StatusBadGateway || errType(t, res.body) != "router_backend_unreachable" {
			t.Fatalf("in-flight request %d on the killed child: %+v, want typed 502", i, res)
		}
	}
	if got := dying.reqs.Load(); got != inFlight {
		t.Fatalf("killed child saw %d requests, want the %d in flight (none replayed)", got, inFlight)
	}

	held := postAsync(fs.URL+"/v1/chat/completions", `{"id":"after-kill"}`)
	waitFor(t, func() bool { return f.waitingCount() == 1 })
	select {
	case res := <-held:
		t.Fatalf("request during the restart answered before promote: %+v", res)
	default:
	}

	restarted, c := newCountingBackend(t)
	n, err := f.promote(restarted.URL, "g2")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("promote released %d held requests, want 1", n)
	}
	res := <-held
	if res.err != nil || res.code != http.StatusOK || res.body != "NEW" {
		t.Fatalf("held request after restart: %+v", res)
	}
	if got := <-c.bodies; got != `{"id":"after-kill"}` || c.hits.Load() != 1 {
		t.Fatalf("restarted child saw body %q hits %d", got, c.hits.Load())
	}
	if got := dying.reqs.Load(); got != inFlight {
		t.Fatalf("held request reached the killed child (%d requests)", got)
	}
	if f.unavailable() {
		t.Fatal("front still unavailable after the restart was promoted")
	}
}

// TestRestartHoldExpiresTyped503 covers a restart that never becomes ready:
// the held request fails with the typed hold-timeout error and Retry-After,
// never reaching the dead child.
//
// fak-test:runtime fast est=150ms
func TestRestartHoldExpiresTyped503(t *testing.T) {
	dying := newDyingGen(t)
	dying.kill()
	f := New(Options{HoldTimeout: 80 * time.Millisecond, ProbeInterval: 5 * time.Millisecond})
	if err := f.Promote(dying.url()); err != nil {
		t.Fatal(err)
	}
	fs := httptest.NewServer(f)
	defer fs.Close()

	first := <-postAsync(fs.URL+"/v1/x", "a")
	if first.err != nil || first.code != http.StatusBadGateway || errType(t, first.body) != "router_backend_unreachable" {
		t.Fatalf("request on the dead child: %+v, want typed 502", first)
	}
	resp, err := http.Post(fs.URL+"/v1/x", "text/plain", strings.NewReader("b"))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable || errType(t, string(b)) != "router_unavailable" {
		t.Fatalf("held request after the deadline: code=%d body=%s, want typed 503", resp.StatusCode, b)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatal("hold timeout must carry Retry-After")
	}
	if got := dying.reqs.Load(); got != 1 {
		t.Fatalf("dead child saw %d requests, want only the first", got)
	}
	if f.InFlight(dying.url()) != 0 {
		t.Fatal("in-flight leaked after a hold timeout")
	}
}

// TestDroppedConnectionOnLiveBackendResumes keeps a live generation that
// drops one connection from stalling the front: its health probe answers, so
// the suspect hold releases and the next request is served by it.
//
// fak-test:runtime fast est=50ms
func TestDroppedConnectionOnLiveBackendResumes(t *testing.T) {
	var drops atomic.Int64
	be := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			return
		}
		_, _ = io.ReadAll(r.Body)
		if drops.Add(1) == 1 {
			if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
				conn.Close()
			}
			return
		}
		io.WriteString(w, "LIVE")
	}))
	defer be.Close()
	f := New(Options{HoldTimeout: 5 * time.Second, ProbeInterval: 5 * time.Millisecond})
	if err := f.Promote(be.URL); err != nil {
		t.Fatal(err)
	}
	fs := httptest.NewServer(f)
	defer fs.Close()
	if res := <-postAsync(fs.URL+"/v1/x", "a"); res.code != http.StatusBadGateway {
		t.Fatalf("dropped request: %+v, want 502", res)
	}
	res := <-postAsync(fs.URL+"/v1/x", "b")
	if res.err != nil || res.code != http.StatusOK || res.body != "LIVE" {
		t.Fatalf("request after a dropped connection on a live backend: %+v", res)
	}
	if f.unavailable() || f.Active() != be.URL {
		t.Fatal("a live backend that dropped one connection must stay active")
	}
}

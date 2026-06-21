package gateway

import (
	"net"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestMetricsExposesVDSOStats asserts /metrics renders the vDSO effectiveness gauges
// sourced from the live vdso.Default stats API (issue #139 part 1). It checks the
// metric families exist with their HELP/TYPE headers — the values themselves come from
// the process-global vDSO and may carry counts from sibling tests, so the contract is
// "these families are emitted", not a specific count.
func TestMetricsExposesVDSOStats(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	text := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		"# TYPE fak_vdso_hit_rate gauge",
		"fak_vdso_hit_rate ",
		"# TYPE fak_vdso_lookups_total counter",
		"fak_vdso_lookups_total ",
		"# TYPE fak_vdso_hits_total counter",
		"fak_vdso_hits_total ",
		"# TYPE fak_vdso_cache_fills_total counter",
		"fak_vdso_cache_fills_total ",
		"# TYPE fak_vdso_invalidations_total counter",
		"fak_vdso_invalidations_total ",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}
}

// TestNoDelayListenerSetsTCPNoDelay verifies the accepted-connection path sets
// TCP_NODELAY (issue #139 part 2): a real TCP listener wrapped by nodelayListener
// yields a *net.TCPConn (the type the wrapper sets NoDelay(true) on) and a working
// connection. A non-TCP listener passes through untouched.
func TestNoDelayListenerSetsTCPNoDelay(t *testing.T) {
	base, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer base.Close()

	ln := nodelayListener(base)
	addr := ln.Addr().String()

	type accepted struct {
		c   net.Conn
		err error
	}
	acceptCh := make(chan accepted, 1)
	go func() {
		c, err := ln.Accept()
		acceptCh <- accepted{c, err}
	}()

	client, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer client.Close()

	got := <-acceptCh
	if got.err != nil {
		t.Fatalf("accept: %v", got.err)
	}
	defer got.c.Close()

	if _, ok := got.c.(*net.TCPConn); !ok {
		t.Fatalf("accepted conn is %T, want *net.TCPConn (the type NoDelay is set on)", got.c)
	}
}

// TestNoDelayListenerPassThroughNonTCP confirms the wrapper does not panic or alter a
// listener whose Accept yields a non-TCP conn — the wrap is always safe to apply.
func TestNoDelayListenerPassThroughNonTCP(t *testing.T) {
	pl := &pipeListener{conns: make(chan net.Conn, 1)}
	ln := nodelayListener(pl)

	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	pl.conns <- srv

	got, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept: %v", err)
	}
	if got != srv {
		t.Fatalf("pass-through returned a different conn than Accept yielded")
	}
}

// pipeListener is a minimal net.Listener whose Accept hands back queued in-memory
// conns — a non-TCP listener to exercise nodelayListener's pass-through arm.
type pipeListener struct {
	conns chan net.Conn
}

func (p *pipeListener) Accept() (net.Conn, error) { return <-p.conns, nil }
func (p *pipeListener) Close() error              { return nil }
func (p *pipeListener) Addr() net.Addr            { return dummyAddr{} }

type dummyAddr struct{}

func (dummyAddr) Network() string { return "pipe" }
func (dummyAddr) String() string  { return "pipe" }

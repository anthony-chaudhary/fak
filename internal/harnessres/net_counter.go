package harnessres

import (
	"io"
	"net"
	"net/http"
	"sync/atomic"
)

// CountingRoundTripper is an http.RoundTripper that tallies the request- and
// response-body bytes crossing to/from the upstream provider. `fak guard` installs one
// on the gateway's upstream client so the harness can report its actual external
// NETWORK workload — the agent↔LLM traffic the proxy exists to carry — WITNESSED
// in-process and cross-platform (epic #2044 / #2049), rather than approximating
// per-process socket accounting the OS does not cleanly expose.
//
// It counts the KERNEL half's network (the gateway is in-process with guard). The
// wrapped child's own direct network (a tool's WebFetch, a git clone) is not seen here
// and stays n/a until the per-PID sampler (#2048) lands. Streaming responses are
// counted as their body is read; a request retried via GetBody may slightly undercount
// (the re-created body is not re-wrapped) — a bounded, documented approximation.
type CountingRoundTripper struct {
	next http.RoundTripper
	tx   atomic.Uint64 // request bytes sent upstream
	rx   atomic.Uint64 // response bytes read from upstream
}

// NewCountingRoundTripper wraps next (use http.DefaultTransport for the default path).
func NewCountingRoundTripper(next http.RoundTripper) *CountingRoundTripper {
	if next == nil {
		next = http.DefaultTransport
	}
	return &CountingRoundTripper{next: next}
}

func (c *CountingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if c == nil {
		return http.DefaultTransport.RoundTrip(req)
	}
	if req != nil && req.Body != nil {
		req.Body = &countingReadCloser{rc: req.Body, n: &c.tx}
	}
	resp, err := c.next.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		return resp, err
	}
	resp.Body = &countingReadCloser{rc: resp.Body, n: &c.rx}
	return resp, err
}

// Bytes returns the cumulative (rx, tx) upstream body bytes observed so far.
func (c *CountingRoundTripper) Bytes() (rx, tx uint64) {
	if c == nil {
		return 0, 0
	}
	return c.rx.Load(), c.tx.Load()
}

type countingReadCloser struct {
	rc io.ReadCloser
	n  *atomic.Uint64
}

func (c *countingReadCloser) Read(p []byte) (int, error) {
	n, err := c.rc.Read(p)
	if n > 0 {
		c.n.Add(uint64(n))
	}
	return n, err
}

func (c *countingReadCloser) Close() error { return c.rc.Close() }

// CountingListener wraps a net.Listener so every byte read from / written to an
// accepted connection is tallied. `fak guard` wraps its OWN gateway listener with one,
// so the harness reports the NETWORK bytes it served on the wire — the child↔gateway
// traffic the proxy carries (which, on the Anthropic passthrough, mirrors the upstream
// request/response payload volume the gateway forwards verbatim) plus any /metrics
// scrape. This is WITNESSED in-process and cross-platform, with no privileged per-process
// socket accounting the OS does not cleanly expose (epic #2044 / #2049). It counts the
// KERNEL half; the wrapped child's own direct network (a tool's WebFetch, a git clone)
// is not served through this listener and stays n/a until the per-PID sampler (#2048).
//
// rx = bytes the harness READ from clients (inbound requests); tx = bytes the harness
// WROTE to clients (outbound responses).
type CountingListener struct {
	net.Listener
	rx atomic.Uint64
	tx atomic.Uint64
}

// NewCountingListener wraps inner. Addr and Close delegate to inner via embedding.
func NewCountingListener(inner net.Listener) *CountingListener {
	return &CountingListener{Listener: inner}
}

func (l *CountingListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return c, err
	}
	return &countingConn{Conn: c, rx: &l.rx, tx: &l.tx}, nil
}

// Bytes returns the cumulative (rx, tx) wire bytes served so far. Safe on nil.
func (l *CountingListener) Bytes() (rx, tx uint64) {
	if l == nil {
		return 0, 0
	}
	return l.rx.Load(), l.tx.Load()
}

type countingConn struct {
	net.Conn
	rx *atomic.Uint64
	tx *atomic.Uint64
}

func (c *countingConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if n > 0 {
		c.rx.Add(uint64(n))
	}
	return n, err
}

func (c *countingConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if n > 0 {
		c.tx.Add(uint64(n))
	}
	return n, err
}

package harnessres

import (
	"io"
	"net"
	"net/http"
	"testing"
)

func TestCountingListenerTallies(t *testing.T) {
	inner, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	cl := NewCountingListener(inner)
	defer cl.Close()

	// Server: read the request bytes, then write a fixed response.
	const reply = "PONG-RESPONSE-BODY"
	done := make(chan struct{})
	go func() {
		defer close(done)
		conn, err := cl.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		buf := make([]byte, 64)
		n, _ := conn.Read(buf) // one inbound read is enough for the tally
		_ = n
		_, _ = conn.Write([]byte(reply))
	}()

	c, err := net.Dial("tcp", cl.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	const req = "PING-REQUEST"
	if _, err := c.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(reply))
	if _, err := io.ReadFull(c, got); err != nil {
		t.Fatal(err)
	}
	<-done

	rx, tx := cl.Bytes()
	// rx = bytes the harness READ from the client (the request); tx = bytes WRITTEN back.
	if rx < uint64(len(req)) {
		t.Errorf("rx = %d, want >= %d (inbound request bytes)", rx, len(req))
	}
	if tx < uint64(len(reply)) {
		t.Errorf("tx = %d, want >= %d (outbound response bytes)", tx, len(reply))
	}
}

func TestCountingListenerNilBytes(t *testing.T) {
	var l *CountingListener
	if rx, tx := l.Bytes(); rx != 0 || tx != 0 {
		t.Fatalf("nil listener Bytes() = (%d,%d), want (0,0)", rx, tx)
	}
}

// TestCountingRoundTripperTalliesBodies keeps the sibling upstream counter honest too.
func TestCountingRoundTripperTallies(t *testing.T) {
	crt := NewCountingRoundTripper(rtFunc(func(req *http.Request) (*http.Response, error) {
		// Drain the request body so the tx counter (wrapping it) advances.
		if req.Body != nil {
			_, _ = io.Copy(io.Discard, req.Body)
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(newStringReader("abcdef"))}, nil
	}))
	req, _ := http.NewRequest("POST", "http://x", newStringReadCloser("REQUESTBODY"))
	resp, err := crt.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body) // reading the response advances rx
	rx, tx := crt.Bytes()
	if tx == 0 {
		t.Error("tx (request body) not counted")
	}
	if rx == 0 {
		t.Error("rx (response body) not counted")
	}
}

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newStringReader(s string) io.Reader         { return &strReader{s: s} }
func newStringReadCloser(s string) io.ReadCloser { return io.NopCloser(&strReader{s: s}) }

type strReader struct {
	s string
	i int
}

func (r *strReader) Read(p []byte) (int, error) {
	if r.i >= len(r.s) {
		return 0, io.EOF
	}
	n := copy(p, r.s[r.i:])
	r.i += n
	return n, nil
}

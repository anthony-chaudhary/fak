package main

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"
)

// TestProbePiBackendLoopbackFallbackIPv4Listener is the mirror of
// TestProbePiBackendLoopbackFallback: an IPv4-only loopback listener refused on
// the [::1] literal must be reached through the "localhost" alias. The IPv6
// direction (the Windows-host-to-WSL2 gateway case from 1a514163a5) needs a
// resolver that maps localhost to ::1, which a stock WSL/Debian /etc/hosts does
// not, so this direction is what witnesses the fallback there.
func TestProbePiBackendLoopbackFallbackIPv4Listener(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("no IPv4 loopback on this host: %v", err)
	}
	if !localhostResolvesToFamily(false) {
		_ = ln.Close()
		t.Skip("localhost does not resolve to an IPv4 loopback here; fallback unobservable")
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"qwen38:27b-q4"}`))
			return
		}
		http.NotFound(w, r)
	})}
	go srv.Serve(ln)
	defer srv.Close()
	_, port, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatalf("split listener addr: %v", err)
	}
	literal := "http://[::1]:" + port + "/v1"
	// Probe the SINGLE origin so this dual-stack guard cannot succeed via the
	// very fallback under test.
	if _, ok := probePiBackendOnce(&http.Client{Timeout: 500 * time.Millisecond}, literal); ok {
		t.Skip("loopback is dual-stacked on this host; fallback unobservable")
	}
	model, ok, resolved := probePiBackend(literal, 2*time.Second)
	if !ok || model != "qwen38:27b-q4" {
		t.Fatalf("probePiBackend fallback = (%q, %v), want the served model via localhost", model, ok)
	}
	if resolved != "http://localhost:"+port+"/v1" {
		t.Fatalf("probePiBackend resolved = %q, want the localhost alias", resolved)
	}
}

// localhostResolvesToFamily reports whether the host resolver maps "localhost"
// to a loopback address of the given family (IPv6 when v6, else IPv4).
func localhostResolvesToFamily(v6 bool) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, "localhost")
	if err != nil {
		return false
	}
	for _, a := range addrs {
		if a.IP.IsLoopback() && (a.IP.To4() == nil) == v6 {
			return true
		}
	}
	return false
}

package modelsrc

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDownloadClientBoundsPreBodyPhasesWithoutWholeRequestTimeout(t *testing.T) {
	client := downloadClient()
	if client.Timeout != 0 {
		t.Fatalf("whole-request timeout = %v, want 0", client.Timeout)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", client.Transport)
	}
	if transport.TLSHandshakeTimeout != defaultTLSHandshakeTimeout {
		t.Fatalf("TLS timeout = %v, want %v", transport.TLSHandshakeTimeout, defaultTLSHandshakeTimeout)
	}
	if transport.ResponseHeaderTimeout != defaultResponseHeaderTimeout {
		t.Fatalf("header timeout = %v, want %v", transport.ResponseHeaderTimeout, defaultResponseHeaderTimeout)
	}
}

func TestDownloadClientRefusesStalledResponseHeaders(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	defer func() {
		close(release)
		server.Close()
	}()

	client := downloadClientWithTimeouts(time.Second, time.Second, 20*time.Millisecond)
	start := time.Now()
	resp, err := client.Get(server.URL)
	if resp != nil {
		resp.Body.Close()
	}
	if err == nil {
		t.Fatal("stalled response headers unexpectedly succeeded")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("header refusal took %v", elapsed)
	}
}

func TestDownloadClientDoesNotTruncateHealthyBodyStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "2")
		w.(http.Flusher).Flush()
		_, _ = w.Write([]byte("a"))
		w.(http.Flusher).Flush()
		time.Sleep(60 * time.Millisecond)
		_, _ = w.Write([]byte("b"))
	}))
	defer server.Close()

	client := downloadClientWithTimeouts(time.Second, time.Second, 20*time.Millisecond)
	resp, err := client.Get(server.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(got) != "ab" {
		t.Fatalf("body = %q, want ab", got)
	}
}

func TestWithHTTPClientOverridesDownloadDefault(t *testing.T) {
	transport := roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Length": []string{"2"}},
			Body:       io.NopCloser(strings.NewReader("ok")),
		}, nil
	})
	registry := New(WithHTTPClient(&http.Client{Transport: transport}))
	assertObject(t, registry.Open, "https://override.invalid/model", []byte("ok"))
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

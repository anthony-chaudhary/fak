package fakclient

import (
	"net/http"
	"testing"
	"time"
)

// A Client built with no WithHTTPClient must NOT use http.DefaultClient (zero timeout, hangs
// forever on a dead gateway) — it must carry a bounded default timeout. This is the
// MISSING_HTTP_TIMEOUT failure mode the repo's boundarylint flags.
func TestNew_DefaultClientHasBoundedTimeout(t *testing.T) {
	c := New("http://127.0.0.1:8080")
	if c.httpClient == http.DefaultClient {
		t.Fatal("default client is http.DefaultClient (no timeout) — an SDK caller would hang forever on a dead gateway")
	}
	if c.httpClient.Timeout <= 0 {
		t.Fatalf("default client timeout = %s, want a positive bound", c.httpClient.Timeout)
	}
	if c.httpClient.Timeout > 5*time.Minute {
		t.Fatalf("default client timeout = %s is implausibly long for the verdict surface", c.httpClient.Timeout)
	}
}

// WithHTTPClient must still let a caller override the default (e.g. for streaming / long bodies).
func TestWithHTTPClient_Overrides(t *testing.T) {
	custom := &http.Client{Timeout: 90 * time.Second}
	c := New("http://127.0.0.1:8080", WithHTTPClient(custom))
	if c.httpClient != custom {
		t.Fatal("WithHTTPClient did not override the default client")
	}
	// A nil override must be ignored (keep the safe default), not panic or zero the client.
	c2 := New("http://127.0.0.1:8080", WithHTTPClient(nil))
	if c2.httpClient == nil || c2.httpClient.Timeout <= 0 {
		t.Fatal("a nil WithHTTPClient override must leave the bounded default in place")
	}
}

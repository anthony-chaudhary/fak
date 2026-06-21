package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestServeSmokeE2E boots the REAL gateway over an actual TCP socket — the full
// New() -> Serve(listener) lifecycle, not httptest with a stubbed Handler — and
// drives a first-time client's three core live-API surfaces end to end: the
// unauthenticated health check, the OpenAI-compatible chat-completions proxy, and
// the native Anthropic /v1/messages server. It then cancels the context and asserts
// the server drains cleanly.
//
// This is the automated analogue of the manual scripts/dogfood-claude.sh --smoke
// witness. CI runs `go test ./...` on Linux, so this gates every commit to the
// serve wire: a regression in real startup, listener-bind, graceful shutdown, or a
// full client request/response roundtrip now fails CI rather than slipping past the
// httptest-only unit tests (which never bind a port).
func TestServeSmokeE2E(t *testing.T) {
	srv := newTestServer(t) // offline mock planner + a tool-adjudicating kernel, no auth

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("bind ephemeral port: %v", err)
	}
	base := "http://" + ln.Addr().String()

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx, ln) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-served:
			if err != nil && err != http.ErrServerClosed {
				t.Errorf("Serve returned unexpected error on shutdown: %v", err)
			}
		case <-time.After(6 * time.Second):
			t.Error("Serve did not return within 6s of ctx cancel (graceful shutdown wedged)")
		}
	})

	client := &http.Client{Timeout: 5 * time.Second}
	waitServing(t, client, base+"/healthz")

	// 1) /healthz — unauthenticated liveness over the real socket.
	if code, body := smokeGet(t, client, base+"/healthz"); code != http.StatusOK || body["ok"] != true {
		t.Fatalf("/healthz = %d %v, want 200 ok:true", code, body)
	}

	// 2) OpenAI /v1/chat/completions — the proxy surface. The offline mock proposes
	// a tool call; the kernel adjudicates it before the body crosses the boundary.
	code, chat := smokePost(t, client, base+"/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"allow_x","parameters":{"type":"object"}}}]}`)
	if code != http.StatusOK {
		t.Fatalf("/v1/chat/completions = %d, want 200 (body %v)", code, chat)
	}
	if _, ok := chat["choices"]; !ok {
		t.Fatalf("/v1/chat/completions missing choices: %v", chat)
	}

	// 3) Anthropic /v1/messages — the native Claude-Code-facing surface.
	code, msg := smokePost(t, client, base+"/v1/messages",
		`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusOK {
		t.Fatalf("/v1/messages = %d, want 200 (body %v)", code, msg)
	}
	if msg["type"] != "message" || msg["role"] != "assistant" {
		t.Fatalf("/v1/messages not a well-formed Anthropic message: %v", msg)
	}
	if blocks, ok := msg["content"].([]any); !ok || len(blocks) == 0 {
		t.Fatalf("/v1/messages content not a non-empty block array: %v", msg["content"])
	}
}

// waitServing polls url until the bound listener accepts and answers (the Serve
// goroutine reaches hs.Serve a beat after MarkReady), failing the test if the
// gateway never comes up within the budget.
func waitServing(t *testing.T, client *http.Client, url string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("gateway did not start serving within 3s")
}

func smokeGet(t *testing.T, client *http.Client, url string) (int, map[string]any) {
	t.Helper()
	resp, err := client.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	return resp.StatusCode, decodeBody(t, resp.Body)
}

func smokePost(t *testing.T, client *http.Client, url, body string) (int, map[string]any) {
	t.Helper()
	resp, err := client.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp.StatusCode, decodeBody(t, resp.Body)
}

func decodeBody(t *testing.T, r io.ReadCloser) map[string]any {
	t.Helper()
	defer r.Close()
	var m map[string]any
	if err := json.NewDecoder(r).Decode(&m); err != nil {
		t.Fatalf("decode response body: %v", err)
	}
	return m
}

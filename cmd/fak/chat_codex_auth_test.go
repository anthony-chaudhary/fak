package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type chatCodexLoopbackTransport struct {
	base   http.RoundTripper
	target *url.URL
}

func (tr chatCodexLoopbackTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.String() != guardCodexChatGPTBackendBaseURL+"/responses" {
		return nil, fmt.Errorf("unexpected subscription endpoint")
	}
	local := r.Clone(r.Context())
	u := *r.URL
	u.Scheme, u.Host = tr.target.Scheme, tr.target.Host
	local.URL = &u
	return tr.base.RoundTrip(local)
}

func TestChatCodexSubscriptionLoopback(t *testing.T) {
	home := t.TempDir()
	writeCred := func(suffix string) {
		t.Helper()
		body := fmt.Sprintf(`{"auth_mode":"chatgpt","tokens":{"access_token":"fixture-token-%s","account_id":"fixture-account-%s"}}`, suffix, suffix)
		if err := os.WriteFile(filepath.Join(home, "auth.json"), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeCred("one")
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if r.Header.Get("Authorization") != "Bearer fixture-token-two" || r.Header.Get(guardCodexChatGPTAccountHeader) != "fixture-account-two" {
			t.Error("request did not carry the refreshed matched credential pair")
		}
		reasoning, _ := body["reasoning"].(map[string]any)
		if body["model"] != "gpt-5.6-luna" || reasoning["effort"] != "low" || body["stream"] != true {
			t.Error("request lost model, effort, or subscription streaming")
		}
		for _, key := range []string{"temperature", "max_output_tokens"} {
			if _, ok := body[key]; ok {
				t.Errorf("unsupported field %s", key)
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"model\":\"gpt-5.6-luna\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"OK\"}]}]}}\n\n")
	}))
	defer ts.Close()
	// Exercise the actual chat flag path in a subprocess: a supplied login must
	// never be sent to an arbitrary endpoint, even when a valid auth file exists.
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	child := exec.Command(bin, "chat", "--codex-auth", "--codex-home", home, "--provider", "openai-responses", "--base-url", ts.URL, "--task", "do not send")
	child.Env = append(os.Environ(), "FAK_OPS_NATIVE_TEST_CHILD=1")
	if output, err := child.CombinedOutput(); err == nil || !strings.Contains(string(output), "--codex-auth requires") || requests != 0 {
		t.Fatalf("noncanonical CLI endpoint was not refused before HTTP: err=%v output=%s requests=%d", err, output, requests)
	}
	p, err := agent.NewProviderHTTPPlanner("openai-responses", guardCodexChatGPTBackendBaseURL, "gpt-5.6-luna", "")
	if err != nil {
		t.Fatal(err)
	}
	target, _ := url.Parse(ts.URL)
	p.Client = &http.Client{Transport: chatCodexLoopbackTransport{base: ts.Client().Transport, target: target}}
	if err := configureChatCodexSubscription(p, home); err != nil {
		t.Fatal(err)
	}
	writeCred("two")
	result, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "Reply OK"}}, nil, agent.WithReasoningEffort("low"))
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 || result.Message.Content != "OK" {
		t.Fatalf("requests=%d response=%q", requests, result.Message.Content)
	}
	if p.Client.CheckRedirect(&http.Request{}, nil) != http.ErrUseLastResponse {
		t.Fatal("subscription redirects must be refused")
	}
	for _, endpoint := range []string{"http://127.0.0.1", "https://chatgpt.com.evil.invalid/backend-api/codex", guardCodexChatGPTBackendBaseURL + "?redirect=1"} {
		p.BaseURL = endpoint
		if err := configureChatCodexSubscription(p, home); err == nil {
			t.Error("accepted noncanonical subscription endpoint")
		}
	}
}

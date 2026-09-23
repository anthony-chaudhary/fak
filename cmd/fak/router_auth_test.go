package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

const testAutoRouterURL = "http://127.0.0.1:8080/v1"

var (
	agentAuthGuidanceHelper = flag.Bool("agent-auth-guidance-helper", false, "test helper")
	chatAuthFailureURL      = flag.String("chat-auth-failure-url", "", "test helper URL")
)

func TestResolvePlannerAPIKeyAutoRouterRequiresProof(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "gateway-key")
	t.Setenv("GEMINI_API_KEY", "provider-key")
	if got := resolveRouterAPIKey("GEMINI_API_KEY", false, true, testAutoRouterURL); got != "" {
		t.Fatalf("unproved auto-router key = %q, want empty", got)
	}
}

func TestResolvePlannerAPIKeyAutoRouterConfigFallback(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	t.Setenv("GEMINI_API_KEY", "provider-key-must-not-leak-to-router")
	writeRouterConfig(t, testAutoRouterURL, "configured-local-key")
	if got := resolveRouterAPIKey("GEMINI_API_KEY", false, true, testAutoRouterURL); got != "configured-local-key" {
		t.Fatalf("auto-router config key = %q, want matching router config key", got)
	}
}

func TestResolvePlannerAPIKeyAutoRouterWithoutMatchingConfig(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	t.Setenv("GEMINI_API_KEY", "provider-key-must-not-leak-to-router")
	if got := resolveRouterAPIKey("GEMINI_API_KEY", false, true, testAutoRouterURL); got != "" {
		t.Fatalf("auto-router key without config = %q, want empty", got)
	}

	writeRouterConfig(t, "http://127.0.0.1:9090/v1", "wrong-router-key")
	if got := resolveRouterAPIKey("GEMINI_API_KEY", false, true, testAutoRouterURL); got != "" {
		t.Fatalf("auto-router key for mismatched config = %q, want empty", got)
	}
}

func TestResolvePlannerAPIKeyExplicitEnvironmentWins(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "gateway-key")
	t.Setenv("EXPLICIT_ROUTER_KEY", "explicit-key")
	if got := resolveRouterAPIKey("EXPLICIT_ROUTER_KEY", true, true, testAutoRouterURL); got != "explicit-key" {
		t.Fatalf("explicit key = %q, want explicitly selected environment", got)
	}
}

func TestResolvePlannerAPIKeyNonAutoUsesProviderEnvironment(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "gateway-key")
	t.Setenv("GEMINI_API_KEY", "provider-key")
	if got := resolveRouterAPIKey("GEMINI_API_KEY", false, false, "https://provider.example/v1"); got != "provider-key" {
		t.Fatalf("provider key = %q, want provider environment key", got)
	}
}

func TestAgentAuthenticationFailureGuidance(t *testing.T) {
	if *agentAuthGuidanceHelper {
		mustAgentRun(errors.New("provider status 401: missing_credentials inert-secret"))
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestAgentAuthenticationFailureGuidance$", "-agent-auth-guidance-helper=true")
	out, err := cmd.CombinedOutput()
	if exitErr, ok := err.(*exec.ExitError); !ok || exitErr.ExitCode() != 1 {
		t.Fatalf("agent auth helper exit = %v, want code 1; output=%s", err, out)
	}
	got := string(out)
	for _, want := range []string{"model authentication failed [authentication]", "Set FAK_GATEWAY_KEY", "--api-key-env VAR"} {
		if !strings.Contains(got, want) {
			t.Fatalf("agent auth guidance missing %q:\n%s", want, got)
		}
	}
	for _, leaked := range []string{"missing_credentials", "inert-secret"} {
		if strings.Contains(got, leaked) {
			t.Fatalf("agent auth guidance leaked provider detail %q:\n%s", leaked, got)
		}
	}
}

func TestChatHeadlessUnauthorizedExitsNonzero(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"type":"authentication_error","message":"missing_credentials inert-secret"}}`))
	}))
	defer srv.Close()

	cmd := exec.Command(os.Args[0], "-test.run=^TestChatHeadlessUnauthorizedHelper$", "-chat-auth-failure-url="+srv.URL)
	out, err := cmd.CombinedOutput()
	if exitErr, ok := err.(*exec.ExitError); !ok || exitErr.ExitCode() != 1 {
		t.Fatalf("chat unauthorized exit = %v, want code 1; output=%s", err, out)
	}
	got := string(out)
	if !strings.Contains(got, "turn terminated [authentication]") {
		t.Fatalf("chat unauthorized output missing authentication classification:\n%s", got)
	}
	for _, leaked := range []string{"missing_credentials", "inert-secret", "configured-local-key"} {
		if strings.Contains(got, leaked) {
			t.Fatalf("chat unauthorized output leaked detail %q:\n%s", leaked, got)
		}
	}
}

func TestChatHeadlessUnauthorizedHelper(t *testing.T) {
	if *chatAuthFailureURL == "" {
		return
	}
	cmdChat([]string{
		"--task", "reply briefly", "--tools=none", "--memory=false", "--max-turns=2",
		"--base-url", *chatAuthFailureURL + "/v1",
		"--provider", "openai", "--model", "test-model",
		"--api-key-env", "FAK_TEST_NO_ROUTER_KEY",
	})
}

func TestSpoofedHealthCannotReleaseImplicitRouterCredential(t *testing.T) {
	const candidate = "configured-local-key"
	var (
		mu       sync.Mutex
		chatAuth string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"ok":true,"model":"test-model"}`)
		case "/v1/chat/completions":
			mu.Lock()
			chatAuth = r.Header.Get("Authorization")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	t.Setenv("FAK_GATEWAY_KEY", candidate)
	key := resolveRouterAPIKey("", false, true, srv.URL+"/v1")
	if key != "" {
		t.Fatalf("spoofed health released implicit credential: %q", key)
	}
	planner := chatPlannerWithResolvedKey(io.Discard, false, srv.URL+"/v1", "openai", "test-model", key, "", "auto", false, false)
	var out strings.Builder
	if err := runChatHeadless(&out, planner, "reply briefly", 1, false, "", ""); err != nil {
		t.Fatalf("headless request through spoof fixture: %v", err)
	}
	mu.Lock()
	gotAuth := chatAuth
	mu.Unlock()
	if gotAuth != "" {
		t.Fatalf("spoof fixture received Authorization: %q", gotAuth)
	}
}

func TestProbeRouterAuthProofFreshSuccessAndReplayFailure(t *testing.T) {
	const candidate = "configured-local-key"
	var (
		mu         sync.Mutex
		firstProof string
		challenges []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encoded := r.Header.Get(routerAuthChallengeHeader)
		nonce, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(nonce) != 32 {
			http.Error(w, "invalid challenge", http.StatusBadRequest)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		challenges = append(challenges, encoded)
		if firstProof == "" {
			mac := hmac.New(sha256.New, []byte(candidate))
			_, _ = mac.Write([]byte(routerAuthProofDomain))
			_, _ = mac.Write(nonce)
			firstProof = base64.StdEncoding.EncodeToString(mac.Sum(nil))
		}
		w.Header().Set(routerAuthProofHeader, firstProof)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"model":"test-model"}`)
	}))
	defer srv.Close()

	client := srv.Client()
	if !probeRouterAuthProof(client, srv.URL, candidate) {
		t.Fatal("fresh gateway proof was rejected")
	}
	if probeRouterAuthProof(client, srv.URL, candidate) {
		t.Fatal("proof replay was accepted for a fresh challenge")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(challenges) != 2 || challenges[0] == challenges[1] {
		t.Fatalf("challenges = %v, want two distinct nonces", challenges)
	}
}

func writeRouterConfig(t *testing.T, baseURL, apiKey string) {
	t.Helper()
	dir := t.TempDir()
	oldDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldDir) })
	body := []byte(`{"provider":{"fak-router":{"options":{"baseURL":"` + baseURL + `","apiKey":"` + apiKey + `"}}}}`)
	if err := os.WriteFile(filepath.Join(dir, "opencode.json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
}

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Reuse the native-chat test binary dispatch to exercise flag parsing, planner
// selection, the real agent loop and HTTP serialization in a separate process.
func chatOutputTokensChild(t *testing.T, providerCap string, args ...string) ([]byte, error) {
	t.Helper()
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	common := []string{"chat", "--task", "Reply OK", "--tools=none", "--memory=false", "--max-turns=1"}
	child := exec.CommandContext(ctx, bin, append(common, args...)...)
	home := t.TempDir()
	child.Dir = home
	overrides := map[string]string{
		"HOME": home, "USERPROFILE": home, "APPDATA": home,
		"XDG_CONFIG_HOME": filepath.Join(home, "config"), "XDG_DATA_HOME": filepath.Join(home, "data"),
		"FAK_OPS_NATIVE_TEST_CHILD": "1", "FAK_PROVIDER_MAX_TOKENS": providerCap,
		"FAK_AGENT_ROUTER_ORIGIN": "", "OPENAI_BASE_URL": "", "FAK_GATEWAY_KEY": "",
		"FAK_TEST_CHAT_OUTPUT_KEY": "fixture-not-a-provider-secret",
	}
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		up := strings.ToUpper(key)
		if _, replaced := overrides[up]; !replaced && !strings.HasPrefix(up, "FAK_OPS_NATIVE_") {
			child.Env = append(child.Env, entry)
		}
	}
	for key, value := range overrides {
		child.Env = append(child.Env, key+"="+value)
	}
	out, err := child.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("native chat exceeded subprocess deadline: %v", ctx.Err())
	}
	return out, err
}

// fak-test:justify why=regression when=changed:cmd/fak/chat.go witness=cmd/fak/chat.go#cmdChat
// fak-test:runtime fast est=3s lane=default
func TestChatOutputTokensHTTPWire(t *testing.T) {
	for _, tc := range []struct {
		name, cap, field string
		flags            []string
		want             int
	}{
		{name: "omitted-default", field: "max_tokens", want: 1024},
		{name: "explicit-zero-default", field: "max_tokens", flags: []string{"--max-output-tokens=0"}, want: 1024},
		{name: "explicit-larger-budget", field: "max_tokens", flags: []string{"--max-output-tokens=12288"}, want: 12288},
		{name: "provider-hard-cap", cap: "4096", field: "max_tokens", flags: []string{"--max-output-tokens=12288"}, want: 4096},
		{name: "completion-field", field: "max_completion_tokens", flags: []string{"--max-output-tokens=12288", "--completion-token-field=max_completion_tokens"}, want: 12288},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requests := make(chan map[string]json.RawMessage, 4)
			var unexpected atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
					unexpected.Add(1)
					http.NotFound(w, r)
					return
				}
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					http.Error(w, "fixture decode failed", http.StatusBadRequest)
					return
				}
				select {
				case requests <- body:
				default:
					t.Error("unexpected repeated native-chat requests")
				}
				if string(body["stream"]) == "true" {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data: {\"id\":\"fixture\",\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"fixture","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}`)
				}
			}))
			defer srv.Close()
			args := []string{"--provider=openai", "--model=fixture-model", "--base-url=" + srv.URL + "/v1", "--api-key-env=FAK_TEST_CHAT_OUTPUT_KEY"}
			out, err := chatOutputTokensChild(t, tc.cap, append(args, tc.flags...)...)
			if err != nil {
				t.Fatalf("native chat exit: %v; output=%s", err, out)
			}
			if unexpected.Load() != 0 || len(requests) != 1 {
				t.Fatalf("HTTP calls: completion=%d unexpected=%d", len(requests), unexpected.Load())
			}
			body := <-requests
			var actual int
			if err := json.Unmarshal(body[tc.field], &actual); err != nil || actual != tc.want {
				t.Fatalf("wire %s=%s want=%d decode=%v", tc.field, body[tc.field], tc.want, err)
			}
			other := "max_completion_tokens"
			if tc.field == other {
				other = "max_tokens"
			}
			if _, exists := body[other]; exists {
				t.Fatalf("wire included conflicting completion field %s", other)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/chat.go witness=cmd/fak/chat.go#cmdChat
// fak-test:runtime fast est=1s lane=default
func TestChatOutputTokensNegativeBeforeNetwork(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "network must not be reached", http.StatusInternalServerError)
	}))
	defer srv.Close()
	// Omit --model deliberately: validation must precede model discovery too.
	out, err := chatOutputTokensChild(t, "", "--provider=openai", "--base-url="+srv.URL+"/v1", "--api-key-env=FAK_TEST_CHAT_OUTPUT_KEY", "--max-output-tokens=-1")
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() == 0 || requests.Load() != 0 {
		t.Fatalf("negative budget exit=%v requests=%d output=%s", err, requests.Load(), out)
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/chat.go witness=cmd/fak/chat.go#cmdChat
// fak-test:runtime fast est=1s lane=default
func TestChatOfflineFlagIsRetired(t *testing.T) {
	for _, budget := range []int{0, 12288} {
		t.Run(fmt.Sprint(budget), func(t *testing.T) {
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				http.Error(w, "retired flag must not reach HTTP", http.StatusInternalServerError)
			}))
			defer srv.Close()
			out, err := chatOutputTokensChild(t, "", "--offline", "--model=fixture-model", "--base-url="+srv.URL+"/v1", fmt.Sprintf("--max-output-tokens=%d", budget))
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() == 0 || requests.Load() != 0 {
				t.Fatalf("retired flag exit=%v requests=%d output=%s", err, requests.Load(), out)
			}
			if !strings.Contains(string(out), "flag provided but not defined: -offline") {
				t.Fatalf("retired flag was not rejected by parsing: %s", out)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/chat.go witness=cmd/fak/chat.go#cmdChat
// fak-test:runtime fast est=1s lane=default
func TestChatOutputTokensWorkflowRefusesBeforeDispatch(t *testing.T) {
	// A missing workflow safely distinguishes normal workflow dispatch (exit 1)
	// from invalid output-budget configuration (usage exit 2), without spawning work.
	for _, tc := range []struct {
		name string
		flag []string
		want int
	}{
		{name: "omitted-retains-workflow-dispatch", want: 1},
		{name: "explicit-zero", flag: []string{"--max-output-tokens=0"}, want: 2},
		{name: "explicit-positive", flag: []string{"--max-output-tokens=12288"}, want: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				http.Error(w, "workflow fixture must not reach HTTP", http.StatusInternalServerError)
			}))
			defer srv.Close()
			checkpoint := filepath.Join(t.TempDir(), "must-not-exist")
			args := []string{"--workflow=output-budget-fixture-does-not-exist", "--workflow-checkpoint-dir=" + checkpoint, "--base-url=" + srv.URL + "/v1"}
			out, err := chatOutputTokensChild(t, "", append(args, tc.flag...)...)
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() != tc.want || requests.Load() != 0 {
				t.Fatalf("workflow config exit=%v want=%d requests=%d output=%s", err, tc.want, requests.Load(), out)
			}
			if _, err := os.Stat(checkpoint); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("workflow fixture created checkpoint side effects: %v", err)
			}
		})
	}
}

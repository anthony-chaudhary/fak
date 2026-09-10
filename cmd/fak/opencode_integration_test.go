package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
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

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// TestOpencodeIntegrationLoopback runs the installed OpenCode process through the
// real guard socket. The upstream is a synthetic protocol fixture: this witnesses
// process, wire, credential-swap, tool-execution, and audit integration only.
func TestOpencodeIntegrationLoopback(t *testing.T) {
	opencode, err := exec.LookPath("opencode")
	if err != nil {
		t.Skip("installed OpenCode is required for the macOS integration witness")
	}

	const (
		model         = "opencode-test-model"
		upstreamKey   = "fixture-upstream-key"
		childKey      = "fixture-child-placeholder"
		fileContent   = "read-through-opencode-and-fak"
		finalResponse = "observed read-through-opencode-and-fak"
	)

	var completions atomic.Int32
	var sawToolResult atomic.Bool
	var sawUpstreamAuth atomic.Bool
	var witnessPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || r.URL.Path == "/v1/models" || r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"context_length":8192,"max_output_tokens":512}]}`, model)
			return
		}
		if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") == "Bearer "+upstreamKey {
			sawUpstreamAuth.Store(true)
		} else {
			http.Error(w, `{"error":{"message":"fixture rejected credential"}}`, http.StatusUnauthorized)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var request struct {
			Model     string `json:"model"`
			MaxTokens int    `json:"max_tokens"`
			Messages  []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if request.Model != model {
			http.Error(w, fmt.Sprintf(`{"error":{"message":"unexpected model %q"}}`, request.Model), http.StatusBadRequest)
			return
		}
		if bytes.Contains(body, []byte("You are a title generator")) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "chatcmpl-title", "object": "chat.completion", "created": 1, "model": model, "choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "Read witness"}, "finish_reason": "stop"}}, "usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
			return
		}
		if request.MaxTokens != 512 {
			http.Error(w, fmt.Sprintf(`{"error":{"message":"unexpected max_tokens %d"}}`, request.MaxTokens), http.StatusBadRequest)
			return
		}
		for _, message := range request.Messages {
			if message.Role == "tool" && strings.Contains(fmt.Sprint(message.Content), fileContent) {
				sawToolResult.Store(true)
			}
		}
		call := completions.Add(1)
		if call == 1 {
			hash := sha256.Sum256(body)
			artifact := filepath.Join(os.TempDir(), fmt.Sprintf("fak-12307-opencode-main-request-%x.json", hash))
			_ = os.WriteFile(artifact, body, 0o600)
			var envelope map[string]json.RawMessage
			_ = json.Unmarshal(body, &envelope)
			var tools []json.RawMessage
			_ = json.Unmarshal(envelope["tools"], &tools)
			t.Logf("OpenCode request envelope: artifact=%s body_bytes=%d messages_bytes=%d tools_bytes=%d tool_count=%d max_tokens=%s top_p=%s temperature=%s", artifact, len(body), len(envelope["messages"]), len(envelope["tools"]), len(tools), envelope["max_tokens"], envelope["top_p"], envelope["temperature"])
		}
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			arguments, _ := json.Marshal(map[string]string{"filePath": witnessPath})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-tool", "object": "chat.completion", "created": 1, "model": model,
				"choices": []any{map[string]any{
					"index": 0,
					"message": map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{
						map[string]any{"id": "call-read-1", "type": "function", "function": map[string]any{"name": "read", "arguments": string(arguments)}},
					}},
					"finish_reason": "tool_calls",
				}},
				"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
			})
		} else {
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "chatcmpl-final", "object": "chat.completion", "created": 2, "model": model, "choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": finalResponse}, "finish_reason": "stop"}}, "usage": map[string]any{"prompt_tokens": 14, "completion_tokens": 4, "total_tokens": 18}})
		}
	}))
	defer upstream.Close()

	workspace := t.TempDir()
	witnessPath = filepath.Join(workspace, "witness.txt")
	gitInit := exec.Command("git", "init", "--quiet")
	gitInit.Dir = workspace
	if out, err := gitInit.CombinedOutput(); err != nil {
		t.Fatalf("initialize isolated workspace: %v: %s", err, out)
	}
	if err := os.WriteFile(filepath.Join(workspace, "dos.toml"), []byte("[lanes]\n[lanes.trees]\ncmd = [\"cmd/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(witnessPath, []byte(fileContent+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	home := filepath.Join(workspace, "home")
	for _, dir := range []string{home, filepath.Join(home, "config"), filepath.Join(home, "data"), filepath.Join(home, "cache")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	auditPath := filepath.Join(workspace, "audit.jsonl")
	launchDir := workspace
	prompt := "Read witness.txt and report its exact content."
	if os.Getenv("FAK_OPENCODE_CAPTURE_REPO_ENVELOPE") == "1" {
		root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
		if err != nil {
			t.Fatalf("resolve repository root: %v", err)
		}
		launchDir = strings.TrimSpace(string(root))
		prompt = "Using parallel subagents, audit the packages under internal/ and report their status"
	}
	t.Chdir(launchDir)
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, "config"))
	t.Setenv("XDG_DATA_HOME", filepath.Join(home, "data"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, "cache"))
	t.Setenv("TEST_OPENCODE_UPSTREAM_KEY", upstreamKey)
	t.Setenv("OPENAI_API_KEY", childKey)
	t.Setenv("PATH", filepath.Dir(opencode)+string(os.PathListSeparator)+os.Getenv("PATH"))
	var output bytes.Buffer
	origRun := opencodeLaunchRun
	opencodeLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		if len(argv) < 3 || argv[1] != "guard" {
			t.Errorf("launcher argv does not enter fak guard: %v", argv)
			return 2
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0])
		cmd.Dir = workspace
		cmd.Env = append(opencodeIntegrationBaseEnv(env),
			guardE2EHelperEnv+"="+strings.Join(argv[2:], " "),
			"FAK_SESSION_REGISTRY="+filepath.Join(workspace, "session-registry.jsonl"),
			`OPENCODE_CONFIG_CONTENT={"provider":{"fak":{"models":{"opencode-test-model":{"limit":{"context":8192,"output":512}}}}}}`,
		)
		cmd.Stdout, cmd.Stderr = stdout, stderr
		cmd.Cancel = func() error {
			if cmd.Process != nil {
				_, _ = procguard.KillPID(cmd.Process.Pid)
			}
			return nil
		}
		if err := cmd.Run(); err != nil {
			t.Errorf("launcher guard child: %v", err)
			return 1
		}
		return 0
	}
	t.Cleanup(func() { opencodeLaunchRun = origRun })
	code := runOpencode(&output, &output, []string{
		"--base-url", upstream.URL + "/v1", "--api-key-env", "TEST_OPENCODE_UPSTREAM_KEY",
		"--model", model, "--audit", auditPath, "--quiet", "--split", "off",
		"--probe", prompt, "--pure", "--auto", "--skip-permissions=false",
		"--", "--dir", launchDir, "--format", "json", "--print-logs", "--log-level", "DEBUG",
	})
	if code != 0 {
		rows, _ := journal.ReadRows(auditPath)
		t.Fatalf("fak opencode launcher failed: code=%d completions=%d saw_tool_result=%t audit_rows=%+v output:\n%s", code, completions.Load(), sawToolResult.Load(), rows, output.String())
	}
	if !strings.Contains(output.String(), finalResponse) {
		t.Fatalf("OpenCode JSON output lacks final fixture response; output:\n%s", output.String())
	}
	if completions.Load() < 2 || !sawToolResult.Load() {
		t.Fatalf("tool round trip incomplete: completions=%d saw_tool_result=%t", completions.Load(), sawToolResult.Load())
	}
	if !sawUpstreamAuth.Load() {
		t.Fatal("guard did not replace the child placeholder with the upstream credential")
	}
	if _, err := journal.Verify(auditPath); err != nil {
		t.Fatalf("guard audit journal failed hash-chain verification: %v", err)
	}
	rows, err := journal.ReadRows(auditPath)
	if err != nil {
		t.Fatalf("read guard audit journal: %v", err)
	}
	var sawReadDecision bool
	for _, row := range rows {
		if row.Kind == "DECIDE" && row.Tool == "read" && (row.Verdict == "ALLOW" || row.Verdict == "TRANSFORM") {
			sawReadDecision = true
			break
		}
	}
	if !sawReadDecision {
		t.Fatalf("audit journal lacks an allowed read DECIDE row: %+v", rows)
	}
}

func opencodeIntegrationBaseEnv(source []string) []string {
	blocked := map[string]bool{
		"HOME": true, "OPENAI_API_KEY": true, "OPENAI_BASE_URL": true,
		"OPENAI_API_BASE": true, "OPENCODE_CONFIG": true,
		"OPENCODE_CONFIG_CONTENT": true, "XDG_CACHE_HOME": true,
		"XDG_CONFIG_HOME": true, "XDG_DATA_HOME": true,
	}
	out := make([]string, 0, len(source))
	for _, entry := range source {
		key, _, _ := strings.Cut(entry, "=")
		if blocked[key] || strings.HasPrefix(key, "FAK_") {
			continue
		}
		out = append(out, entry)
	}
	return out
}

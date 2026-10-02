package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type nativeExactCommandPlanner struct {
	commands        []string
	results         []string
	seenToolResults int
}

func (*nativeExactCommandPlanner) Model() string { return "native-exact-command-test" }

func (p *nativeExactCommandPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	var toolResults []string
	for _, message := range messages {
		if message.Role == agent.RoleTool {
			toolResults = append(toolResults, message.Content)
		}
	}
	if len(toolResults) > p.seenToolResults {
		p.results = append(p.results, toolResults[p.seenToolResults:]...)
		p.seenToolResults = len(toolResults)
	}
	if len(p.results) == len(p.commands) {
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}, FinishReason: "stop"}, nil
	}
	args, err := json.Marshal(codetools.BashArgs{Command: p.commands[len(p.results)]})
	if err != nil {
		return nil, err
	}
	return nativeCallTurn("bash-"+string(rune('a'+len(p.results))), codetools.ToolBash, string(args), 8), nil
}

func runNativeExactCommands(t *testing.T, cfg Config, commands ...string) ([]string, string) {
	t.Helper()
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	cfg.EngineID = "localtools"
	cfg.Model = "test-model"
	cfg.VDSO = true
	cfg.Native = true
	cfg.NativeMaxTurns = len(commands) + 2
	if cfg.NativeCodeWorkspace == "" {
		cfg.NativeCodeWorkspace = t.TempDir()
	}

	planner := &nativeExactCommandPlanner{commands: commands}
	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("New native gateway: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = planner

	healthReq := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	healthRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(healthRec, healthReq)
	if healthRec.Code != http.StatusOK {
		t.Fatalf("health status=%d body=%s", healthRec.Code, healthRec.Body.String())
	}

	body, err := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": "run the requested checks"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		raw, _ := io.ReadAll(rec.Result().Body)
		t.Fatalf("native messages status=%d body=%s", rec.Code, raw)
	}
	if len(planner.results) != len(commands) {
		t.Fatalf("native gateway executed %d/%d commands: %q", len(planner.results), len(commands), planner.results)
	}
	return planner.results, healthRec.Body.String()
}

func nativeBashCalls(t *testing.T, commands ...string) []agent.ToolCall {
	t.Helper()
	calls := make([]agent.ToolCall, 0, len(commands))
	for i, command := range commands {
		args, err := json.Marshal(codetools.BashArgs{Command: command})
		if err != nil {
			t.Fatal(err)
		}
		calls = append(calls, agent.ToolCall{
			ID:       "bash-proposal-" + string(rune('a'+i)),
			Function: agent.Func{Name: codetools.ToolBash, Arguments: string(args)},
		})
	}
	return calls
}

func newNativeExactCommandServer(t *testing.T, grants []string) *Server {
	t.Helper()
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	srv, err := New(Config{
		EngineID:                   "localtools",
		Model:                      "test-model",
		VDSO:                       true,
		Native:                     true,
		NativeCodeWorkspace:        t.TempDir(),
		NativeExactAllowedCommands: grants,
	})
	if err != nil {
		t.Fatalf("New native gateway: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// fak-test:runtime fast est=2s
func TestNativeExactCommandGrantChatAdjudication(t *testing.T) {
	const granted = "go version"
	const focusedDefault = "git status --short"

	defaultSrv := newNativeExactCommandServer(t, nil)
	defaultCtx, cancelDefault := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelDefault()
	kept, _, dropped, served, hits := defaultSrv.adjudicateProposedServed(defaultCtx,
		nativeBashCalls(t, granted, focusedDefault), "default-chat-proposal")
	if len(kept) != 0 || dropped != 2 || served != "" || hits != 0 {
		t.Fatalf("default chat adjudication kept=%+v dropped=%d served=%q hits=%d, want both Bash proposals denied", kept, dropped, served, hits)
	}

	grants := []string{granted}
	exactSrv := newNativeExactCommandServer(t, grants)
	grants[0] = granted + " && echo suffix"
	exactCtx, cancelExact := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelExact()
	kept, adjs, dropped, served, hits := exactSrv.adjudicateProposedServed(exactCtx,
		nativeBashCalls(t, granted, granted+" && echo suffix", "go env GOPATH"), "granted-chat-proposal")
	if len(kept) != 0 || dropped != 3 || served != "" || hits != 0 {
		t.Fatalf("chat proxy must remain fail-closed: kept=%+v dropped=%d served=%q hits=%d adjs=%+v", kept, dropped, served, hits, adjs)
	}
	for _, adj := range adjs {
		if adj.Verdict.Reason != "DEFAULT_DENY" || adj.Verdict.By != "lease-admission" {
			t.Fatalf("chat proxy denial=%+v, want lease-admission DEFAULT_DENY", adj.Verdict)
		}
	}
}

// fak-test:runtime fast est=5s
func TestNativeExactCommandGrantOwnedGateway(t *testing.T) {
	const granted = "go version"

	defaults, _ := runNativeExactCommands(t, Config{}, granted, "git status --short")
	if !strings.Contains(defaults[0], string(codetools.CodeCommandDeny)) {
		t.Fatalf("default server admitted ungranted command: %s", defaults[0])
	}
	if strings.Contains(defaults[1], string(codetools.CodeCommandDeny)) {
		t.Fatalf("grant feature changed focused default: %s", defaults[1])
	}

	grants := []string{granted}
	cfg := Config{NativeExactAllowedCommands: grants}
	// New must retain its own immutable copy. Mutating the caller's Config after New
	// is witnessed behaviorally below: the original stays allowed and the mutation
	// does not become a new grant.
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	cfg.EngineID, cfg.Model, cfg.VDSO, cfg.Native = "localtools", "test-model", true, true
	cfg.NativeMaxTurns, cfg.NativeCodeWorkspace = 5, t.TempDir()
	planner := &nativeExactCommandPlanner{commands: []string{granted, granted + " && echo suffix", "go env GOPATH"}}
	srv, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	srv.planner = planner
	grants[0] = granted + " && echo suffix"

	body := []byte(`{"model":"test-model","max_tokens":256,"messages":[{"role":"user","content":"run checks"}]}`)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK || len(planner.results) != 3 {
		t.Fatalf("native owned loop status=%d results=%q body=%s", rec.Code, planner.results, rec.Body.String())
	}
	if strings.Contains(planner.results[0], string(codetools.CodeCommandDeny)) || !strings.Contains(planner.results[0], "go version") {
		t.Fatalf("exact server grant did not execute: %s", planner.results[0])
	}
	for i, result := range planner.results[1:] {
		if !strings.Contains(result, string(codetools.CodeCommandDeny)) {
			t.Fatalf("ungranted command %d escaped exact matching: %s", i, result)
		}
	}

	health := httptest.NewRecorder()
	srv.Handler().ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if strings.Contains(health.Body.String(), granted) || strings.Contains(rec.Body.String(), granted) {
		t.Fatalf("exact grant leaked onto HTTP: health=%s response=%s", health.Body.String(), rec.Body.String())
	}
}

// fak-test:runtime fast est=3s
func TestNativeBashCommandTimeoutOwnedGateway(t *testing.T) {
	const bound = 20 * time.Millisecond
	command := quoteNativeTimeoutCommand(os.Args[0]) + " -test.run=TestNativeBashCommandTimeoutChild"
	t.Setenv("FAK_NATIVE_BASH_TIMEOUT_CHILD", "1")

	results, _ := runNativeExactCommands(t, Config{
		NativeExactAllowedCommands: []string{command},
		NativeMaxCommandTime:       bound,
	}, command)
	if !strings.Contains(results[0], `"timed_out":true`) {
		t.Fatalf("server timeout did not reach owned Bash tool: %s", results[0])
	}
}

func quoteNativeTimeoutCommand(path string) string {
	if runtime.GOOS == "windows" {
		return `"` + strings.ReplaceAll(path, `"`, `""`) + `"`
	}
	return "'" + strings.ReplaceAll(path, "'", "'\"'\"'") + "'"
}

// fak-test:runtime fast est=1s
func TestNativeBashCommandTimeoutChild(t *testing.T) {
	if os.Getenv("FAK_NATIVE_BASH_TIMEOUT_CHILD") == "1" {
		time.Sleep(250 * time.Millisecond)
	}
}

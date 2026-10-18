package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/zaitask"
)

type fakeZAI struct {
	srv   *httptest.Server
	hits  atomic.Int32
	last  map[string]any
	auth  string
	path  string
	reply func(w http.ResponseWriter, body map[string]any)
}

func newFakeZAI(t *testing.T, reply func(w http.ResponseWriter, body map[string]any)) *fakeZAI {
	t.Helper()
	f := &fakeZAI{reply: reply}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.hits.Add(1)
		f.auth = r.Header.Get("Authorization")
		f.path = r.URL.Path
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		f.last = body
		f.reply(w, body)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func chatReply(content string) func(http.ResponseWriter, map[string]any) {
	return func(w http.ResponseWriter, body map[string]any) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices":    []map[string]any{{"message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
			"model":      body["model"],
			"request_id": "req-1",
			"usage":      map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	}
}

func env(key string) func(string) string {
	return func(k string) string {
		if k == "ZAI_API_KEY" {
			return key
		}
		return ""
	}
}

type runOut struct {
	err            error
	stdout, stderr string
}

func runWith(t *testing.T, stdin io.Reader, getenv func(string) string, args ...string) runOut {
	t.Helper()
	if stdin == nil {
		stdin = strings.NewReader("")
	}
	var stdout, stderr bytes.Buffer
	err := run(args, stdin, &stdout, &stderr, getenv)
	return runOut{err: err, stdout: stdout.String(), stderr: stderr.String()}
}

func receiptOf(t *testing.T, stderr string) map[string]any {
	t.Helper()
	const marker = "[zaitask receipt] "
	i := strings.Index(stderr, marker)
	if i < 0 {
		t.Fatalf("no receipt line on stderr: %q", stderr)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(stderr[i+len(marker):])), &got); err != nil {
		t.Fatalf("receipt is not JSON: %v", err)
	}
	return got
}

// fak-test:runtime fast est=10ms lane=default
func TestRunDefaultModelPrintsContentAndReceipt(t *testing.T) {
	f := newFakeZAI(t, chatReply("hello back"))
	out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light", "--max-tokens", "77", "say", "hi")
	if out.err != nil || exitCode(out.err) != 0 {
		t.Fatalf("run err=%v", out.err)
	}
	if out.stdout != "hello back" {
		t.Fatalf("stdout = %q", out.stdout)
	}
	if f.hits.Load() != 1 || f.path != "/chat/completions" || f.auth != "Bearer test-key" {
		t.Fatalf("hits=%d path=%q auth=%q", f.hits.Load(), f.path, f.auth)
	}
	if f.last["model"] != zaitask.DefaultModel || f.last["max_tokens"] != float64(77) || f.last["stream"] != false {
		t.Fatalf("request = %v", f.last)
	}
	if msgs, _ := f.last["messages"].([]any); len(msgs) != 1 || msgs[0].(map[string]any)["content"] != "say hi" {
		t.Fatalf("messages = %v", f.last["messages"])
	}
	r := receiptOf(t, out.stderr)
	if r["model"] != zaitask.DefaultModel || r["request_id"] != "req-1" || r["usage"].(map[string]any)["total_tokens"] != float64(5) {
		t.Fatalf("receipt = %v", r)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunJSONEmitsResultWithRouting(t *testing.T) {
	f := newFakeZAI(t, chatReply("json body"))
	out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light", "--json", "do", "it")
	if out.err != nil {
		t.Fatalf("run err=%v", out.err)
	}
	var got struct {
		zaitask.Result
		Routing zaitask.Suitability `json:"routing"`
	}
	if err := json.Unmarshal([]byte(out.stdout), &got); err != nil {
		t.Fatalf("stdout not JSON: %v", err)
	}
	if got.Content != "json body" || got.RequestID != "req-1" || !got.Routing.Suitable || got.Routing.TargetTier < 2 {
		t.Fatalf("payload = %+v", got)
	}
	if strings.Contains(out.stderr, "[zaitask receipt]") {
		t.Fatalf("--json must not also write the stderr receipt")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunReadsPromptFromStdinWhenNoArgs(t *testing.T) {
	f := newFakeZAI(t, chatReply("ok"))
	out := runWith(t, strings.NewReader("summarize this note"), env("test-key"), "--base-url", f.srv.URL)
	if out.err != nil {
		t.Fatalf("run err=%v", out.err)
	}
	if msgs, _ := f.last["messages"].([]any); len(msgs) != 1 || msgs[0].(map[string]any)["content"] != "summarize this note" {
		t.Fatalf("messages = %v", f.last["messages"])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunFlashModelSendsReasoningEffort(t *testing.T) {
	f := newFakeZAI(t, chatReply("flash"))
	out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light",
		"--model", zaitask.GLM53FlashModel, "--reasoning-effort", "low", "go")
	if out.err != nil {
		t.Fatalf("run err=%v", out.err)
	}
	if out.stdout != "flash" || f.last["model"] != zaitask.GLM53FlashModel || f.last["reasoning_effort"] != "low" || f.last["stream"] != false {
		t.Fatalf("stdout=%q request=%v", out.stdout, f.last)
	}
	if r := receiptOf(t, out.stderr); r["provider"] != zaitask.HostedProvider || r["engine"] != zaitask.HostedEngine || r["fak_native"] != false {
		t.Fatalf("receipt = %v", r)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunFlashModelAssemblesStream(t *testing.T) {
	f := newFakeZAI(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"role":"assistant","content":"str"}}],"model":"glm-5.3-flash","request_id":"s1"}`+"\n\n"+
			`data: {"choices":[{"delta":{"content":"eamed"},"finish_reason":"stop"}],"model":"glm-5.3-flash","request_id":"s1"}`+"\n\n"+
			"data: [DONE]\n\n")
	})
	out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light",
		"--model", zaitask.GLM53FlashModel, "--stream", "--json", "go")
	if out.err != nil {
		t.Fatalf("run err=%v", out.err)
	}
	var got zaitask.Result
	if err := json.Unmarshal([]byte(out.stdout), &got); err != nil {
		t.Fatalf("stdout not JSON: %v", err)
	}
	if f.last["stream"] != true || got.Content != "streamed" || !got.Streamed || !got.Done {
		t.Fatalf("request=%v result=%+v", f.last, got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunRefusesBeforeAnyRequest(t *testing.T) {
	cases := []struct {
		name   string
		getenv func(string) string
		stdin  string
		args   []string
		want   error
		code   int
	}{
		{"unsuitable frontier class", env("test-key"), "", []string{"--task-class", "hard", "refactor", "the", "kernel"}, errUnsuitable, 1},
		{"unsupported class", env("test-key"), "", []string{"--task-class", "bogus", "x"}, errUnsuitable, 1},
		{"empty stdin prompt", env("test-key"), "  \n", nil, errUnsuitable, 1},
		{"stream on non-flash model", env("test-key"), "", []string{"--task-class", "light", "--stream", "x"}, errFlashOnlyFlags, 1},
		{"effort on non-flash model", env("test-key"), "", []string{"--task-class", "light", "--reasoning-effort", "low", "x"}, errFlashOnlyFlags, 1},
		{"unknown flag", env("test-key"), "", []string{"--nope"}, errUsage, 2},
		{"bad flag value", env("test-key"), "", []string{"--max-tokens", "many"}, errUsage, 2},
		{"help", env("test-key"), "", []string{"-h"}, flag.ErrHelp, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeZAI(t, chatReply("must not be reached"))
			args := append([]string{"--base-url", f.srv.URL}, tc.args...)
			out := runWith(t, strings.NewReader(tc.stdin), tc.getenv, args...)
			if !errors.Is(out.err, tc.want) {
				t.Fatalf("err = %v, want errors.Is %v", out.err, tc.want)
			}
			if got := exitCode(out.err); got != tc.code {
				t.Fatalf("exitCode = %d, want %d", got, tc.code)
			}
			if f.hits.Load() != 0 || out.stdout != "" {
				t.Fatalf("hits=%d stdout=%q, want no request and no output", f.hits.Load(), out.stdout)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunForceOverridesUnsuitable(t *testing.T) {
	f := newFakeZAI(t, chatReply("forced"))
	out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "hard", "--force", "refactor", "kernel")
	if out.err != nil || out.stdout != "forced" || f.hits.Load() != 1 {
		t.Fatalf("err=%v stdout=%q hits=%d", out.err, out.stdout, f.hits.Load())
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunClientFailuresExitOne(t *testing.T) {
	t.Run("missing api key", func(t *testing.T) {
		for _, model := range []string{zaitask.DefaultModel, zaitask.GLM53FlashModel} {
			f := newFakeZAI(t, chatReply("unreached"))
			out := runWith(t, nil, env(""), "--base-url", f.srv.URL, "--task-class", "light", "--model", model, "x")
			if out.err == nil || exitCode(out.err) != 1 || f.hits.Load() != 0 || out.stdout != "" {
				t.Fatalf("model=%s err=%v hits=%d stdout=%q", model, out.err, f.hits.Load(), out.stdout)
			}
		}
	})
	t.Run("provider http error", func(t *testing.T) {
		for _, model := range []string{zaitask.DefaultModel, zaitask.GLM53FlashModel} {
			f := newFakeZAI(t, func(w http.ResponseWriter, _ map[string]any) {
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = io.WriteString(w, `{"error":{"message":"quota","code":1302}}`)
			})
			out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light", "--model", model, "x")
			if out.err == nil || exitCode(out.err) != 1 || f.hits.Load() != 1 || out.stdout != "" {
				t.Fatalf("model=%s err=%v hits=%d stdout=%q", model, out.err, f.hits.Load(), out.stdout)
			}
		}
	})
	t.Run("request timeout", func(t *testing.T) {
		release := make(chan struct{})
		f := newFakeZAI(t, func(http.ResponseWriter, map[string]any) { <-release })
		t.Cleanup(func() { close(release) })
		out := runWith(t, nil, env("test-key"), "--base-url", f.srv.URL, "--task-class", "light", "--timeout", "50ms", "x")
		if !errors.Is(out.err, context.DeadlineExceeded) || exitCode(out.err) != 1 || out.stdout != "" {
			t.Fatalf("err=%v stdout=%q", out.err, out.stdout)
		}
	})
	t.Run("stdin read error", func(t *testing.T) {
		readErr := errors.New("boom")
		out := runWith(t, io.MultiReader(strings.NewReader("part"), errReader{readErr}), env("test-key"))
		if !errors.Is(out.err, readErr) || exitCode(out.err) != 1 {
			t.Fatalf("err = %v", out.err)
		}
	})
}

type errReader struct{ err error }

func (r errReader) Read([]byte) (int, error) { return 0, r.err }

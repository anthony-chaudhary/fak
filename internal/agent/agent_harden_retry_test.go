package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func agentHardenFastRetry(t *testing.T) {
	t.Helper()
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "4")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
}

const agentHardenOK = `{"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`

// agentHardenServer answers each POST with respond(n), n counting from 1.
func agentHardenServer(t *testing.T, respond func(n int32, w http.ResponseWriter)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	posts := new(atomic.Int32)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		respond(posts.Add(1), w)
	}))
	t.Cleanup(srv.Close)
	return srv, posts
}

func agentHardenStatus(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Retry-After", "0")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, body)
}

func agentHardenPlanner(srv *httptest.Server, limit int) *HTTPPlanner {
	p := NewHTTPPlanner(srv.URL, "harden-fixture", "")
	p.Client = srv.Client()
	p.IdenticalRetryLimit = limit
	return p
}

func agentHardenCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return ctx
}

var agentHardenMsgs = []Message{{Role: RoleUser, Content: "hi"}}

func agentHardenCall(t *testing.T, p *HTTPPlanner, stream bool) error {
	var err error
	if stream {
		_, err = p.CompleteStream(agentHardenCtx(t), nil, agentHardenMsgs, nil)
	} else {
		_, err = p.Complete(agentHardenCtx(t), agentHardenMsgs, nil)
	}
	return err
}

func TestAgentHardenIdenticalRetryLimitStops502(t *testing.T) {
	agentHardenFastRetry(t)
	for _, stream := range []bool{false, true} {
		t.Run("stream="+strconv.FormatBool(stream), func(t *testing.T) {
			srv, posts := agentHardenServer(t, func(_ int32, w http.ResponseWriter) {
				agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"decode failed"}}`)
			})
			err := agentHardenCall(t, agentHardenPlanner(srv, 1), stream)
			if !errors.Is(err, ErrIdenticalRetry) {
				t.Fatalf("err = %v, want ErrIdenticalRetry", err)
			}
			var ie *IdenticalRetryError
			if !errors.As(err, &ie) || ie.Status != http.StatusBadGateway || ie.Sends != 2 || ie.BodySHA256 == "" {
				t.Fatalf("identical retry error = %+v", ie)
			}
			if got := posts.Load(); got != 2 {
				t.Fatalf("POSTs = %d, want 2", got)
			}
		})
	}
}

func TestAgentHardenLegacyRetryStillRecovers(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(n int32, w http.ResponseWriter) {
		if n == 1 {
			agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"busy"}}`)
			return
		}
		agentHardenStatus(w, http.StatusOK, agentHardenOK)
	})
	comp, err := agentHardenPlanner(srv, 0).Complete(agentHardenCtx(t), agentHardenMsgs, nil)
	if err != nil || comp.Message.Content != "done" {
		t.Fatalf("comp=%+v err=%v", comp, err)
	}
	if got := posts.Load(); got != 2 {
		t.Fatalf("POSTs = %d, want 2", got)
	}
}

func TestAgentHardenContextOverflowNeverRetried(t *testing.T) {
	agentHardenFastRetry(t)
	for _, status := range []int{http.StatusBadRequest, http.StatusInternalServerError, http.StatusRequestEntityTooLarge} {
		for _, stream := range []bool{false, true} {
			t.Run(strconv.Itoa(status)+"/stream="+strconv.FormatBool(stream), func(t *testing.T) {
				srv, posts := agentHardenServer(t, func(_ int32, w http.ResponseWriter) {
					agentHardenStatus(w, status, `{"error":{"code":"context_length_exceeded"}}`)
				})
				err := agentHardenCall(t, agentHardenPlanner(srv, 0), stream)
				if !errors.Is(err, ErrContextOverflow) {
					t.Fatalf("err = %v, want ErrContextOverflow", err)
				}
				if got := posts.Load(); got != 1 {
					t.Fatalf("POSTs = %d, want 1", got)
				}
			})
		}
	}
}

func TestAgentHardenRunArmContextOverflowEndsArm(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(_ int32, w http.ResponseWriter) {
		agentHardenStatus(w, http.StatusBadRequest, `{"error":{"code":"context_length_exceeded","message":"maximum context length is 4096"}}`)
	})
	m, err := RunArm(agentHardenCtx(t), agentHardenPlanner(srv, 0), "fixture task", false, 4, nil, WithInfraRepromptBudget(3))
	if !errors.Is(err, ErrContextOverflow) {
		t.Fatalf("err = %v, want ErrContextOverflow", err)
	}
	if m.EndedBy != EndedByContextOverflow || m.InfraReprompts != 0 {
		t.Fatalf("metrics = %+v", m)
	}
	if got := posts.Load(); got != 1 {
		t.Fatalf("POSTs = %d, want 1", got)
	}
}

func TestAgentHardenRunArmIdenticalRetryEndsArm(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(_ int32, w http.ResponseWriter) {
		agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"decode failed"}}`)
	})
	m, err := RunArm(agentHardenCtx(t), agentHardenPlanner(srv, 1), "fixture task", false, 4, nil, WithInfraRepromptBudget(3))
	if !errors.Is(err, ErrIdenticalRetry) {
		t.Fatalf("err = %v, want ErrIdenticalRetry", err)
	}
	if m.EndedBy != EndedByIdenticalRetry || m.InfraReprompts != 0 {
		t.Fatalf("metrics = %+v", m)
	}
	if got := posts.Load(); got != 2 {
		t.Fatalf("POSTs = %d, want 2", got)
	}
}

func TestAgentHardenClassifyOverflowAndIdentical(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		reason string
	}{
		{"in-band stream", &UpstreamStreamError{Type: "error", Message: "This model's maximum context length is 8192 tokens"}, "CONTEXT_OVERFLOW"},
		{"stream code", &UpstreamStreamError{Code: "exceed_context_size_error", Message: "x"}, "CONTEXT_OVERFLOW"},
		{"status body", &UpstreamStatusError{Status: 500, Body: `{"error":"prompt is too long"}`}, "CONTEXT_OVERFLOW"},
		{"typed", &ContextOverflowError{Status: 413, Cause: errors.New("too large")}, "CONTEXT_OVERFLOW"},
		{"identical", &IdenticalRetryError{Status: 502, Sends: 2, Cause: &UpstreamStatusError{Status: 502}}, "IDENTICAL_RETRY"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			class, reason := classifyCompleteError(tc.err)
			if class != infraTerminal || reason != tc.reason {
				t.Fatalf("classify = %v/%q, want terminal/%q", class, reason, tc.reason)
			}
		})
	}
	for _, e := range []error{
		&UpstreamStreamError{Type: "error", Message: "overloaded"},
		&UpstreamStreamError{Type: "rate_limit_error", Message: "maximum context length of tokens per minute reached"},
		&UpstreamStatusError{Status: 502, Body: `{"error":"context window cache warming, retry"}`},
	} {
		if class, _ := classifyCompleteError(e); class != infraReprompt {
			t.Fatalf("transient %v became terminal", e)
		}
	}
}

func TestAgentHardenGenericContextWordingStaysRetryable(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(n int32, w http.ResponseWriter) {
		if n == 1 {
			agentHardenStatus(w, http.StatusInternalServerError, `{"error":{"message":"too many tokens in flight; context window busy"}}`)
			return
		}
		agentHardenStatus(w, http.StatusOK, agentHardenOK)
	})
	if _, err := agentHardenPlanner(srv, 0).Complete(agentHardenCtx(t), agentHardenMsgs, nil); err != nil {
		t.Fatalf("err = %v, want recovery after a transient 500", err)
	}
	if got := posts.Load(); got != 2 {
		t.Fatalf("POSTs = %d, want 2", got)
	}
}

type agentHardenRequest struct {
	MaxTokens int       `json:"max_tokens"`
	Messages  []Message `json:"messages"`
}

func TestAgentHardenTruncatedToolCallLoop(t *testing.T) {
	agentHardenFastRetry(t)
	const truncated = `<|message_model|>write_file<|content_invoke_tool_json|>{"name":"write_file","args":{"content":"aaaaaaaa`
	var mu sync.Mutex
	var reqs []agentHardenRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req agentHardenRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		mu.Lock()
		reqs = append(reqs, req)
		mu.Unlock()
		body, _ := json.Marshal(map[string]any{
			"choices": []any{map[string]any{"index": 0, "finish_reason": "length",
				"message": map[string]any{"role": "assistant", "content": truncated}}},
			"usage": map[string]int{"prompt_tokens": 5, "completion_tokens": req.MaxTokens, "total_tokens": 5 + req.MaxTokens},
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	p := agentHardenPlanner(srv, 0)
	p.MaxTokens = 64
	m, err := RunArm(agentHardenCtx(t), p, "fixture task", false, 5, nil)
	if !errors.Is(err, ErrTruncatedToolCall) {
		t.Fatalf("err = %v, want ErrTruncatedToolCall", err)
	}
	if m.EndedBy != EndedByToolCallTruncated || m.FinalAnswer != "" || m.ToolCallTruncations != 2 {
		t.Fatalf("metrics = %+v", m)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reqs) != 2 {
		t.Fatalf("POSTs = %d, want 2", len(reqs))
	}
	last := reqs[1].Messages[len(reqs[1].Messages)-1]
	if !strings.Contains(last.Content, "cut off at the output limit") {
		t.Fatalf("turn 2 lacks truncation feedback: %+v", last)
	}
	if reqs[0].MaxTokens != 64 || reqs[1].MaxTokens != 128 {
		t.Fatalf("max_tokens = %d then %d, want 64 then 128", reqs[0].MaxTokens, reqs[1].MaxTokens)
	}
}

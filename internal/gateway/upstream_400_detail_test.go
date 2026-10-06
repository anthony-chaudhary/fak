package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// llamaOverflowBody is the llama-server context-overflow 400 shape.
const llamaOverflowBody = `{"error":{"code":400,"message":"request (40141 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":40141,"n_ctx":32768}}`

// TestParseUpstream400Classifies pins the typed-first classification: the JSON code/type
// decide an overflow, message text is the fallback, and a non-JSON or truncated body
// still classifies through the raw signature match.
//
// fak-test:runtime fast est=5ms
func TestParseUpstream400Classifies(t *testing.T) {
	cases := []struct {
		name           string
		body           string
		overflow       bool
		prompt, window int
	}{
		{"llama typed", llamaOverflowBody, true, 40141, 32768},
		{"openai code", `{"error":{"message":"too many tokens","type":"invalid_request_error","code":"context_length_exceeded"}}`, true, 0, 0},
		{"llama type only", `{"error":{"message":"no","type":"exceed_context_size_error"}}`, true, 0, 0},
		{"vllm message", `{"object":"error","message":"This model's maximum context length is 32768 tokens","type":"BadRequestError","code":400}`, true, 0, 0},
		{"string error", `{"error":"prompt is too long: 213290 tokens > 200000 maximum"}`, true, 0, 0},
		{"truncated json", `{"error":{"code":400,"message":"request (40141 tokens) exceeds the available context size (32768 tok`, true, 0, 0},
		{"malformed role", `{"error":{"message":"invalid role: tool_result","type":"invalid_request_error","code":"invalid_value"}}`, false, 0, 0},
		{"n_ctx without overflow", `{"error":{"message":"bad grammar","type":"invalid_request_error","n_ctx":4096}}`, false, 0, 4096},
		{"empty", "", false, 0, 0},
	}
	for _, c := range cases {
		d := parseUpstream400(c.body)
		if d.ContextOverflow != c.overflow || d.PromptTokens != c.prompt || d.ContextWindow != c.window {
			t.Errorf("%s: got overflow=%v prompt=%d window=%d, want %v/%d/%d", c.name, d.ContextOverflow, d.PromptTokens, d.ContextWindow, c.overflow, c.prompt, c.window)
		}
	}
}

type errEnvelope struct {
	Error map[string]any `json:"error"`
}

func writeUpstreamErrEnvelope(t *testing.T, s *Server, err error) (int, map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	s.writeUpstreamErr(rec, err)
	var env errEnvelope
	if jerr := json.Unmarshal(rec.Body.Bytes(), &env); jerr != nil {
		t.Fatalf("decode: %v body=%s", jerr, rec.Body.String())
	}
	return rec.Code, env.Error
}

// TestWriteUpstreamErrContextOverflowCarriesTypedWindow: a proxied llama-server overflow
// reaches the client as context_length_exceeded with the upstream's prompt size and
// per-request window as integer fields, and no upstream prose on the default path.
//
// fak-test:runtime fast est=5ms
func TestWriteUpstreamErrContextOverflowCarriesTypedWindow(t *testing.T) {
	status, e := writeUpstreamErrEnvelope(t, &Server{}, &agent.UpstreamStatusError{Status: http.StatusBadRequest, Body: llamaOverflowBody})
	if status != http.StatusBadRequest || e["code"] != "context_length_exceeded" || e["type"] != "invalid_request_error" {
		t.Fatalf("status=%d code=%v type=%v, want 400 context_length_exceeded invalid_request_error", status, e["code"], e["type"])
	}
	if e["prompt_tokens"] != float64(40141) || e["context_window"] != float64(32768) {
		t.Fatalf("prompt_tokens=%v context_window=%v, want 40141 and 32768", e["prompt_tokens"], e["context_window"])
	}
	if _, ok := e["upstream_message"]; ok {
		t.Fatalf("default path forwarded upstream_message: %v", e)
	}
}

// TestWriteUpstreamErrGeneric400UpstreamMessageIsOptIn: a non-overflow 400 keeps
// upstream_invalid_request; upstream_message appears only where detail exposure is on,
// scrubbed of credential-shaped runs and bounded.
//
// fak-test:runtime fast est=5ms
func TestWriteUpstreamErrGeneric400UpstreamMessageIsOptIn(t *testing.T) {
	body := `{"error":{"message":"invalid role: tool_result (key sk-abcdefghijklmnop0123) ` + strings.Repeat("x ", 400) + `","type":"invalid_request_error"}}`
	err := &agent.UpstreamStatusError{Status: http.StatusBadRequest, Body: body}

	_, off := writeUpstreamErrEnvelope(t, &Server{}, err)
	if off["code"] != "upstream_invalid_request" {
		t.Fatalf("code = %v, want upstream_invalid_request", off["code"])
	}
	for _, k := range []string{"upstream_message", "prompt_tokens", "context_window"} {
		if _, ok := off[k]; ok {
			t.Fatalf("default path carries %q: %v", k, off)
		}
	}

	_, on := writeUpstreamErrEnvelope(t, &Server{exposeUpstreamErrorDetail: true}, err)
	if on["code"] != "upstream_invalid_request" {
		t.Fatalf("opt-in code = %v, want upstream_invalid_request", on["code"])
	}
	msg, ok := on["upstream_message"].(string)
	if !ok || msg == "" {
		t.Fatalf("opt-in path missing upstream_message: %v", on)
	}
	if strings.Contains(msg, "sk-abcdefghijklmnop0123") {
		t.Fatalf("upstream_message kept a credential-shaped run")
	}
	if len(msg) > forbiddenDetailMax+len("…") {
		t.Fatalf("upstream_message len %d exceeds bound %d", len(msg), forbiddenDetailMax)
	}
}

// TestWriteUpstreamErrInKernelOverflowCarriesTypedWindow: the in-kernel overflow carries
// the same typed fields, so a client reads one shape whichever side refused.
//
// fak-test:runtime fast est=5ms
func TestWriteUpstreamErrInKernelOverflowCarriesTypedWindow(t *testing.T) {
	_, e := writeUpstreamErrEnvelope(t, &Server{}, &agent.InKernelContextLengthError{PromptTokens: 9000, MaxNewTokens: 100, MaxContext: 8192})
	if e["code"] != "context_length_exceeded" || e["prompt_tokens"] != float64(9000) || e["context_window"] != float64(8192) {
		t.Fatalf("in-kernel envelope = %v", e)
	}
}

// TestErrObjectStandardKeysWin: extra fields cannot overwrite the four standard keys.
//
// fak-test:runtime fast est=1ms
func TestErrObjectStandardKeysWin(t *testing.T) {
	obj := errObject(http.StatusBadRequest, "c", "m", map[string]any{"code": "x", "message": "y", "context_window": 1})
	if obj["code"] != "c" || obj["message"] != "m" || obj["context_window"] != 1 {
		t.Fatalf("errObject = %v", obj)
	}
}

// TestExposeUpstreamErrorDetailEnvOptIn: FAK_EXPOSE_UPSTREAM_ERROR_DETAIL opts a serve in.
//
// fak-test:runtime fast est=20ms
func TestExposeUpstreamErrorDetailEnvOptIn(t *testing.T) {
	t.Setenv("FAK_EXPOSE_UPSTREAM_ERROR_DETAIL", "")
	if newTestServerWithConfig(t, Config{EngineID: "test", Model: "m", VDSO: true}).exposeUpstreamErrorDetail {
		t.Fatal("detail exposure must default off")
	}
	t.Setenv("FAK_EXPOSE_UPSTREAM_ERROR_DETAIL", "1")
	if !newTestServerWithConfig(t, Config{EngineID: "test", Model: "m", VDSO: true}).exposeUpstreamErrorDetail {
		t.Fatal("FAK_EXPOSE_UPSTREAM_ERROR_DETAIL=1 must enable detail exposure")
	}
}

package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// An upstream 400 whose body names a context-window overflow must surface the
// context_length_exceeded code (the same code the in-kernel path mints), not the generic
// upstream_invalid_request — a router keys failover to a longer-window engine on that code.
// The body is only matched: no byte of it may reach the client message (#82/#346).
func TestUpstream4xxStatus_ContextOverflowSurfacesCode(t *testing.T) {
	bodies := []string{
		// llama-server, as journaled on strix3 2026-10-05.
		`{"error":{"code":400,"message":"request (40141 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":40141,"n_ctx":32768}}`,
		`{"error":{"message":"This model's maximum context length is 65536 tokens. However, you requested 213290 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}`,
		`{"object":"error","message":"This model's Maximum Context Length is 32768 tokens","code":400}`,
	}
	for _, body := range bodies {
		status, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: 400, Body: body})
		if status != 400 {
			t.Errorf("status = %d, want 400 (body %q)", status, body)
		}
		if code != "context_length_exceeded" {
			t.Errorf("code = %q, want context_length_exceeded (body %q)", code, body)
		}
		for _, leak := range []string{"40141", "213290", "32768", "65536", "n_ctx"} {
			if strings.Contains(msg, leak) {
				t.Errorf("message %q leaks upstream body detail %q", msg, leak)
			}
		}
	}
}

// A plain malformed 400, and a non-400 status carrying an overflow signature, keep their
// existing codes: the overflow arm is scoped to 400 bodies that actually name the window.
func TestUpstream4xxStatus_ContextOverflowArmIsScoped(t *testing.T) {
	cases := []struct {
		status   int
		body     string
		wantCode string
	}{
		{400, `{"error":{"message":"invalid role: tool_result"}}`, "upstream_invalid_request"},
		{400, "", "upstream_invalid_request"},
		{413, `{"error":{"code":"context_length_exceeded"}}`, "upstream_payload_too_large"},
		{422, `{"error":{"code":"context_length_exceeded"}}`, "upstream_request_rejected"},
	}
	for _, c := range cases {
		_, code, _ := upstreamErrorStatus(&agent.UpstreamStatusError{Status: c.status, Body: c.body})
		if code != c.wantCode {
			t.Errorf("status %d body %q: code = %q, want %q", c.status, c.body, code, c.wantCode)
		}
	}
}

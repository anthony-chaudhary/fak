package agent

import (
	"context"
	"strings"
	"testing"
)

// DeepSeek V4 offline conformance fixtures (issue #3011, parent #3006).
//
// These pin two documented DeepSeek wire shapes that the OpenAI-compatible
// adapter path already handles in production but that carried no DeepSeek
// witness: the provider ERROR ENVELOPE / 429 rate-limit body, and a STREAMING
// turn whose TERMINAL chunk carries usage (cache hit/miss + reasoning
// subcounter). DeepSeek rides the same chat-completions adapter as OpenAI, so
// the whole point of a conformance fixture is to prove fak depends only on the
// OpenAI-compatible semantics — no DeepSeek-specific branch, no live spend.
//
// Everything here is OFFLINE: no DEEPSEEK_API_KEY, no network, no GPU. The
// error fixtures parse a fixed body through ParseResponse; the streaming
// fixture replays a fixed SSE body through an httptest server (sseServer).

// TestDeepSeekConformanceErrorEnvelopesSurfaceAsParseErrors proves DeepSeek's
// OpenAI-shaped `{"error":{...}}` envelopes — 429 rate-limit, 402 insufficient
// balance, 401 auth, 5xx server error — surface as a parse error carrying the
// provider's own message, never a silently-empty Completion a caller would treat
// as a real (empty) turn. This is the offline half of "provider error envelopes
// / 429 handling"; the retry/rehome behavior on the live status code is exercised
// separately by the streaming/status tests.
func TestDeepSeekConformanceErrorEnvelopesSurfaceAsParseErrors(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name    string
		raw     string
		wantMsg string
	}{
		{
			"rate_limit_429",
			`{"error":{"message":"Rate limit reached for deepseek-v4-pro. Please retry after a moment.","type":"rate_limit_reached","code":"429"}}`,
			"Rate limit reached",
		},
		{
			"insufficient_balance_402",
			`{"error":{"message":"Insufficient Balance","type":"insufficient_balance","code":"insufficient_balance"}}`,
			"Insufficient Balance",
		},
		{
			"authentication_401",
			`{"error":{"message":"Authentication Fails, Your api key is invalid","type":"authentication_error","code":"invalid_api_key"}}`,
			"Authentication Fails",
		},
		{
			"server_error_503",
			`{"error":{"message":"The server is overloaded or not ready yet.","type":"server_error"}}`,
			"overloaded",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			comp, err := adapter.ParseResponse([]byte(c.raw))
			if err == nil {
				t.Fatalf("want a parse error for a DeepSeek error envelope, got nil (comp=%+v)", comp)
			}
			if comp != nil {
				t.Fatalf("error envelope must not yield a Completion: %+v", comp)
			}
			if !strings.Contains(err.Error(), c.wantMsg) {
				t.Errorf("error = %q, want it to carry the provider message %q", err.Error(), c.wantMsg)
			}
		})
	}
}

// TestDeepSeekConformanceStreamingTerminalUsage proves a DeepSeek streaming turn
// with delta.reasoning_content + delta.content, terminated by a usage-only chunk
// (empty choices) carrying prompt_cache_hit/miss and the reasoning subcounter,
// assembles correctly: the content is streamed live, the reasoning is accumulated
// but never lifted into executable tool calls, and the TERMINAL usage surfaces
// through the same provider-neutral accessors every cache-value consumer reads.
func TestDeepSeekConformanceStreamingTerminalUsage(t *testing.T) {
	const body = "data: {\"model\":\"deepseek-v4-pro\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Two plus two. \"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"The answer is four.\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":40,\"total_tokens\":1040,\"prompt_cache_hit_tokens\":800,\"prompt_cache_miss_tokens\":200,\"completion_tokens_details\":{\"reasoning_tokens\":25}}}\n\n" +
		"data: [DONE]\n\n"
	srv, _ := sseServer(t, body)

	p := NewHTTPPlanner(srv.URL, "deepseek-v4-pro", "")
	var got []string
	comp, err := p.CompleteStream(context.Background(), func(frag string) error {
		got = append(got, frag)
		return nil
	}, []Message{{Role: RoleUser, Content: "2+2?"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	// Only the final answer reaches the live content sink — reasoning is never streamed.
	if strings.Join(got, "|") != "4" {
		t.Fatalf("sink fragments = %q, want only the final content [4]", got)
	}
	if comp.Message.Content != "4" {
		t.Fatalf("content = %q, want 4", comp.Message.Content)
	}
	if comp.Message.ReasoningContent != "Two plus two. The answer is four." {
		t.Fatalf("reasoning_content = %q, want the accumulated deltas", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 || comp.FinishReason == "tool_calls" {
		t.Fatalf("reasoning must not be lifted into tool calls: finish=%q calls=%+v", comp.FinishReason, comp.Message.ToolCalls)
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("finish = %q, want stop (the terminal usage chunk must not clobber it)", comp.FinishReason)
	}
	// The terminal usage chunk normalizes through the provider-neutral accessors:
	// 800 cache hit + 200 miss = 1000 resident context, 25 reasoning tokens broken
	// out of the 40 completion tokens without deducting from them.
	if got := comp.Usage.CachedPromptTokens(); got != 800 {
		t.Errorf("CachedPromptTokens() = %d, want 800 (prompt_cache_hit_tokens)", got)
	}
	if got := comp.Usage.UncachedPromptTokens(); got != 200 {
		t.Errorf("UncachedPromptTokens() = %d, want 200 (prompt_cache_miss_tokens)", got)
	}
	if got := comp.Usage.ContextWindowTokens(); got != 1000 {
		t.Errorf("ContextWindowTokens() = %d, want 1000 (hit+miss, no double-count)", got)
	}
	if got := comp.Usage.ReasoningTokens(); got != 25 {
		t.Errorf("ReasoningTokens() = %d, want 25 (completion_tokens_details.reasoning_tokens)", got)
	}
	if comp.Usage.CompletionTokens != 40 {
		t.Errorf("CompletionTokens = %d, want the provider's untouched 40", comp.Usage.CompletionTokens)
	}
}

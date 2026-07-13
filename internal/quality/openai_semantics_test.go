package quality

import (
	"fmt"
	"strings"
	"testing"
)

// oaiFaithfulTokens is the witnessed completion token stream every response in
// these tests is audited against: 5 tokens.
func oaiFaithfulTokens() []string {
	return []string{"Throughput", "increased", "12", "%", "."}
}

// oaiFaithfulContent is the assembled assistant text of the faithful decode.
const oaiFaithfulContent = "Throughput increased 12%."

// oaiChatResponse renders a minimal OpenAI-compatible chat-completion response
// with the given message and usage fields.
func oaiChatResponse(role, content, finish string, prompt, completion, total int) string {
	return fmt.Sprintf(`{"object":"chat.completion","choices":[{"index":0,"message":{"role":%q,"content":%q},"finish_reason":%q}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
		role, content, finish, prompt, completion, total)
}

// oaiToolCallResponse is a faithful pure tool-call turn: content null,
// finish_reason "tool_calls", usage consistent with a 5-token trace.
const oaiToolCallResponse = `{"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_throughput","arguments":"{\"week\":\"2026-W28\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":5,"total_tokens":14}}`

// oaiSemanticsCase builds a valid case pinned to maxTokens whose only oracle is
// openai-compat-semantics. The reference trace carries the golden response JSON
// in its Text, per the additive Trace seam.
func oaiSemanticsCase(maxTokens int) QualityCase {
	toks := oaiFaithfulTokens()
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "openai-compat-semantics-chat",
		Version: 1,
		Prompt:  "Summarize this week's throughput as an OpenAI-compatible chat completion.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: maxTokens},
		Reference: Trace{
			Tokens: toks,
			Text:   oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, len(toks), 7+len(toks)),
		},
		Oracles: []string{"openai-compat-semantics"},
	}
}

// oaiEngineTrace pairs the witnessed token stream with a response JSON body.
func oaiEngineTrace(body string) Trace {
	return Trace{Tokens: oaiFaithfulTokens(), Text: body}
}

// TestOAISemanticsRegistered proves the oracle registered under its stable name
// and kind, so cases can reference it by name.
func TestOAISemanticsRegistered(t *testing.T) {
	os, err := Lookup([]string{"openai-compat-semantics"})
	if err != nil {
		t.Fatalf("Lookup(openai-compat-semantics): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestOAISemanticsFaithfulResponsePasses is the happy path: a response whose
// finish_reason, usage arithmetic, trace-witnessed completion count, and
// message shape all hold passes at score 1.
func TestOAISemanticsFaithfulResponsePasses(t *testing.T) {
	c := oaiSemanticsCase(16)
	eng := oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 5, 12))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if !v.Pass {
		t.Fatalf("faithful response must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
}

// TestOAISemanticsMislabeledTruncationFails is the headline defect witness: the
// decode hit max_tokens (5 tokens at max_tokens 5) but the response claims
// finish_reason "stop". The oracle fails and Detail names the finish_reason
// field with the expected "length".
func TestOAISemanticsMislabeledTruncationFails(t *testing.T) {
	c := oaiSemanticsCase(5) // trace has exactly 5 tokens: truncated by length
	eng := oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 5, 12))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if v.Pass {
		t.Fatalf("finish_reason \"stop\" on a length-truncated decode must fail; got %+v", v)
	}
	if want := 4.0 / 5.0; v.Score != want {
		t.Errorf("score = %v, want %v (4 of 5 checks held)", v.Score, want)
	}
	if !strings.Contains(v.Detail, "finish_reason") || !strings.Contains(v.Detail, `want "length"`) {
		t.Errorf("Detail must name finish_reason and the expected \"length\"; got %q", v.Detail)
	}
}

// TestOAISemanticsMiscountedUsageFails injects broken usage arithmetic
// (prompt+completion != total): fail, Detail names usage.total_tokens.
func TestOAISemanticsMiscountedUsageFails(t *testing.T) {
	c := oaiSemanticsCase(16)
	eng := oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 5, 99))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if v.Pass {
		t.Fatalf("miscounted usage must fail; got %+v", v)
	}
	if !strings.Contains(v.Detail, "usage.total_tokens") {
		t.Errorf("Detail must name usage.total_tokens; got %q", v.Detail)
	}
}

// TestOAISemanticsCompletionCountMismatchFails injects a usage block that is
// internally consistent (7+3=10) but disagrees with the 5-token witnessed
// trace: fail, Detail names usage.completion_tokens.
func TestOAISemanticsCompletionCountMismatchFails(t *testing.T) {
	c := oaiSemanticsCase(16)
	eng := oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 3, 10))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if v.Pass {
		t.Fatalf("completion_tokens disagreeing with the trace must fail; got %+v", v)
	}
	if !strings.Contains(v.Detail, "usage.completion_tokens") {
		t.Errorf("Detail must name usage.completion_tokens; got %q", v.Detail)
	}
}

// TestOAISemanticsWrongRoleFails injects a non-assistant message role: fail,
// Detail names message.role.
func TestOAISemanticsWrongRoleFails(t *testing.T) {
	c := oaiSemanticsCase(16)
	eng := oaiEngineTrace(oaiChatResponse("user", oaiFaithfulContent, "stop", 7, 5, 12))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if v.Pass {
		t.Fatalf("non-assistant role must fail; got %+v", v)
	}
	if !strings.Contains(v.Detail, "message.role") {
		t.Errorf("Detail must name message.role; got %q", v.Detail)
	}
}

// TestOAISemanticsToolCallTurn proves both directions of the tool_calls rung: a
// message carrying tool calls with finish_reason "tool_calls" passes, and the
// same body mislabeled "stop" fails naming the finish_reason field.
func TestOAISemanticsToolCallTurn(t *testing.T) {
	c := oaiSemanticsCase(16)
	v := OAISemantics{}.Judge(Trace{}, oaiEngineTrace(oaiToolCallResponse), c)
	if !v.Pass {
		t.Fatalf("faithful tool-call turn must pass; got %+v", v)
	}

	mislabeled := strings.Replace(oaiToolCallResponse, `"finish_reason":"tool_calls"`, `"finish_reason":"stop"`, 1)
	v = OAISemantics{}.Judge(Trace{}, oaiEngineTrace(mislabeled), c)
	if v.Pass {
		t.Fatalf("finish_reason \"stop\" on a tool-call turn must fail; got %+v", v)
	}
	if !strings.Contains(v.Detail, "finish_reason") || !strings.Contains(v.Detail, `want "tool_calls"`) {
		t.Errorf("Detail must name finish_reason and the expected \"tool_calls\"; got %q", v.Detail)
	}
}

// TestOAISemanticsMalformedResponseFailsClosed defines the unreadable-envelope
// edge: non-JSON text, an empty choices list, and a messageless choice all fail
// closed at score 0.
func TestOAISemanticsMalformedResponseFailsClosed(t *testing.T) {
	c := oaiSemanticsCase(16)
	for _, body := range []string{
		"Throughput increased 12%.",
		`{"object":"chat.completion","choices":[]}`,
		`{"object":"chat.completion","choices":[{"index":0,"finish_reason":"stop"}]}`,
	} {
		v := OAISemantics{}.Judge(Trace{}, oaiEngineTrace(body), c)
		if v.Pass || v.Score != 0 {
			t.Errorf("body %q: got %+v, want fail closed at score 0", body, v)
		}
		if !strings.Contains(v.Detail, "response envelope") {
			t.Errorf("body %q: Detail must name the envelope; got %q", body, v.Detail)
		}
	}
}

// TestOAISemanticsMinScoreTolerance proves the rubric threshold gate: the same
// single-violation response (4/5 checks held) passes when the case tolerates
// MinScore 0.8, and the tolerated violation is still named in Detail.
func TestOAISemanticsMinScoreTolerance(t *testing.T) {
	c := oaiSemanticsCase(5)
	c.Rubric.MinScore = 0.8
	eng := oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 5, 12))
	v := OAISemantics{}.Judge(Trace{}, eng, c)
	if !v.Pass {
		t.Fatalf("4/5 checks must pass at MinScore 0.8; got %+v", v)
	}
	if !strings.Contains(v.Detail, "finish_reason") {
		t.Errorf("tolerated detail should still name the violation; got %q", v.Detail)
	}
}

// TestOAISemanticsSpineIntegration runs the mislabeled-truncation defect
// through the full spine: the failure bundle names openai-compat-semantics as
// the failing oracle and carries the finish_reason violation in its detail.
func TestOAISemanticsSpineIntegration(t *testing.T) {
	c := oaiSemanticsCase(5)
	eng := ScriptedRunner{
		Label: "engine-mislabeled-truncation",
		Trace: oaiEngineTrace(oaiChatResponse("assistant", oaiFaithfulContent, "stop", 7, 5, 12)),
	}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("mislabeled truncation must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "openai-compat-semantics" {
		t.Errorf("failing oracle = %q, want openai-compat-semantics", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, "finish_reason") {
		t.Errorf("bundle detail must name finish_reason; got %q", fb.Detail)
	}
}

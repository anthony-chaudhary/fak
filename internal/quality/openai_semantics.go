package quality

import (
	"encoding/json"
	"fmt"
)

// OAISemantics is the OpenAI-compatible response-semantics oracle (#4547): it
// qualifies the response ENVELOPE an OpenAI-compatible serving front end wraps
// around a decode, not the decode itself. A token stream can be bit-exact
// against the reference while the response still lies about WHY the decode
// ended (finish_reason "stop" on a length-truncated completion silently drops
// the "your answer was cut off" signal), miscounts usage (billing and budget
// gates key off prompt+completion=total), or ships a message shape a client
// rejects. Those defects are invisible to token-differential oracles, so this
// rubric judges the envelope directly.
//
// The engine's response travels as JSON in eng.Text (the additive Trace seam);
// eng.Tokens remains the witnessed completion token stream the usage block is
// audited against. Five deterministic checks run, each named by the response
// field it audits:
//
//  1. message.role — the first choice's message role must be "assistant".
//  2. message.content — an assistant message must carry content or tool_calls.
//  3. finish_reason — must match the reason derived from ground truth: a
//     message carrying tool_calls must say "tool_calls"; a decode whose trace
//     hit Params.MaxTokens must say "length"; otherwise "stop". A value
//     outside the closed vocabulary ("stop", "length", "tool_calls",
//     "content_filter") is its own violation; "content_filter" is recognized
//     but never expected here — hermetic spine traces model no moderation.
//  4. usage arithmetic — prompt_tokens + completion_tokens must equal
//     total_tokens, with no negative counts.
//  5. usage/trace agreement — completion_tokens must equal the number of
//     tokens the trace actually witnessed.
//
// Score = passed checks / 5; Pass iff Score >= Rubric.MinScore (default 1:
// every semantics check must hold). On failure Detail names the FIRST bad
// field — `finish_reason: got "stop", want "length"` — per the spine's
// localization contract. An unparseable, choiceless, or messageless response
// fails closed at score 0: a response that cannot be read has no verifiable
// semantics.
type OAISemantics struct{}

func (OAISemantics) Name() string { return "openai-compat-semantics" }
func (OAISemantics) Kind() string { return "rubric" }

func init() { Register(OAISemantics{}) }

// oaiCheckCount is the fixed number of semantics checks Judge scores over.
const oaiCheckCount = 5

// oaiResponse is the OpenAI-compatible chat-completion wire shape this oracle
// audits — only the fields the five checks read.
type oaiResponse struct {
	Object  string      `json:"object"`
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage   `json:"usage"`
}

type oaiChoice struct {
	Index        int         `json:"index"`
	Message      *oaiMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// oaiMessage keeps Content a *string so an explicit JSON null (the legal shape
// for a pure tool-call turn) is distinguishable from an empty string.
type oaiMessage struct {
	Role      string        `json:"role"`
	Content   *string       `json:"content"`
	ToolCalls []oaiToolCall `json:"tool_calls,omitempty"`
}

type oaiToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function oaiFunctionCall `json:"function"`
}

type oaiFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (OAISemantics) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "openai-compat-semantics", Kind: "rubric", Pass: true, Score: 1}
	resp, err := oaiParseResponse(eng.Text)
	if err != nil {
		return rubricFail(v, fmt.Sprintf("response envelope: %v", err))
	}
	violations := oaiAudit(resp, eng, c)
	v.Score = float64(oaiCheckCount-len(violations)) / float64(oaiCheckCount)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: every semantics check must hold
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("%d/%d semantics check(s) failed; first: %s",
			len(violations), oaiCheckCount, violations[0])
		return v
	}
	if len(violations) > 0 {
		v.Detail = fmt.Sprintf("semantics score %.2f >= %.2f (tolerated: %s)",
			v.Score, min, violations[0])
		return v
	}
	u := resp.Usage
	v.Detail = fmt.Sprintf("all %d response-semantics checks held (finish_reason %q, usage %d+%d=%d)",
		oaiCheckCount, resp.Choices[0].FinishReason, u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	return v
}

// oaiParseResponse decodes the engine text as a chat-completion response and
// admits it only if it carries at least one choice with a message — the minimum
// surface the five checks need. Anything less fails closed in Judge.
func oaiParseResponse(text string) (oaiResponse, error) {
	var resp oaiResponse
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		return oaiResponse{}, fmt.Errorf("engine text is not valid response JSON: %v", err)
	}
	if len(resp.Choices) == 0 {
		return oaiResponse{}, fmt.Errorf("response carries no choices")
	}
	if resp.Choices[0].Message == nil {
		return oaiResponse{}, fmt.Errorf("choices[0] carries no message")
	}
	return resp, nil
}

// oaiAudit runs the five checks in their documented order and returns one
// violation string per failed check, each prefixed with the field it audits.
func oaiAudit(resp oaiResponse, eng Trace, c QualityCase) []string {
	var out []string
	msg := resp.Choices[0].Message
	if msg.Role != "assistant" {
		out = append(out, fmt.Sprintf("message.role: got %q, want %q", msg.Role, "assistant"))
	}
	if msg.Content == nil && len(msg.ToolCalls) == 0 {
		out = append(out, "message.content: null with no tool_calls (an assistant message must carry one)")
	}
	if s := oaiAuditFinishReason(resp.Choices[0].FinishReason, msg, eng, c); s != "" {
		out = append(out, s)
	}
	if s := oaiAuditUsageArithmetic(resp.Usage); s != "" {
		out = append(out, s)
	}
	if s := oaiAuditUsageTrace(resp.Usage, eng); s != "" {
		out = append(out, s)
	}
	return out
}

// oaiFinishReasons is the closed finish_reason vocabulary an OpenAI-compatible
// response may carry.
var oaiFinishReasons = map[string]bool{
	"stop":           true,
	"length":         true,
	"tool_calls":     true,
	"content_filter": true,
}

// oaiAuditFinishReason checks the declared finish_reason against the reason the
// trace and params prove. It returns at most one violation for the field: an
// unknown value is reported as unknown, a known-but-wrong value as a mismatch
// with the ground-truth evidence in parentheses.
func oaiAuditFinishReason(got string, msg *oaiMessage, eng Trace, c QualityCase) string {
	if !oaiFinishReasons[got] {
		return fmt.Sprintf("finish_reason: unknown value %q", got)
	}
	want, why := oaiExpectedFinishReason(msg, eng, c)
	if got != want {
		return fmt.Sprintf("finish_reason: got %q, want %q (%s)", got, want, why)
	}
	return ""
}

// oaiExpectedFinishReason derives the finish_reason the response MUST declare
// from ground truth the responder does not control: the message's tool calls,
// the witnessed token count, and the case's pinned max_tokens. It returns the
// expected value and the evidence sentence a mismatch Detail cites.
func oaiExpectedFinishReason(msg *oaiMessage, eng Trace, c QualityCase) (string, string) {
	if len(msg.ToolCalls) > 0 {
		return "tool_calls", fmt.Sprintf("message carries %d tool call(s)", len(msg.ToolCalls))
	}
	if c.Params.MaxTokens > 0 && len(eng.Tokens) >= c.Params.MaxTokens {
		return "length", fmt.Sprintf("decode emitted %d token(s) at max_tokens %d",
			len(eng.Tokens), c.Params.MaxTokens)
	}
	return "stop", fmt.Sprintf("decode ended at %d token(s) under max_tokens %d",
		len(eng.Tokens), c.Params.MaxTokens)
}

// oaiAuditUsageArithmetic checks the usage block's internal arithmetic:
// present, non-negative, and prompt + completion = total.
func oaiAuditUsageArithmetic(u *oaiUsage) string {
	if u == nil {
		return "usage: block missing from response"
	}
	if u.PromptTokens < 0 || u.CompletionTokens < 0 || u.TotalTokens < 0 {
		return fmt.Sprintf("usage: negative count (prompt_tokens %d, completion_tokens %d, total_tokens %d)",
			u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	}
	if u.PromptTokens+u.CompletionTokens != u.TotalTokens {
		return fmt.Sprintf("usage.total_tokens: %d != prompt_tokens %d + completion_tokens %d",
			u.TotalTokens, u.PromptTokens, u.CompletionTokens)
	}
	return ""
}

// oaiAuditUsageTrace checks the usage block against the witnessed trace: the
// declared completion_tokens must equal the tokens the decode actually emitted.
func oaiAuditUsageTrace(u *oaiUsage, eng Trace) string {
	if u == nil {
		return "usage.completion_tokens: unavailable (usage block missing)"
	}
	if u.CompletionTokens != len(eng.Tokens) {
		return fmt.Sprintf("usage.completion_tokens: %d != %d token(s) the trace witnessed",
			u.CompletionTokens, len(eng.Tokens))
	}
	return ""
}

// chat.go is the planner seam: provider transcript adapters + a typed
// message/tool vocabulary + the Planner interface both the live client and the
// offline mock satisfy. See doc.go for the package's trust framing (this is the
// host-side loop, not the guarded guest) and the A/B-loop purpose.

package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Role constants for chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	// RoleGoal is a message carrying the session's ACTIVE GOAL — the intentional GC
	// root of the context heap (#845, epic #844). It is not a chat turn the model
	// emits; a host injects it (e.g. from the harness /goal) so the context planner
	// can PIN the goal as a root distinct from the first user turn, which the planner
	// previously used as a proxy. A goal span is pinned resident regardless of its
	// relevance/recency score, so a long session pursuing one goal never elides the
	// span that goal depends on. Absent (no goal message), the planner is unchanged.
	RoleGoal = "goal"
)

// ToolCall is one function call the model emitted. Arguments is the RAW JSON
// string the model produced — kept verbatim (never re-marshaled) so a malformed
// or alias-shaped argument object survives to the kernel exactly as the model
// emitted it (the whole point of the repair measurement).
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function Func   `json:"function"`
}

// Func is the function half of a tool call.
type Func struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // raw JSON string as emitted by the model
}

// UnmarshalJSON decodes a tool call's function object, keeping the arguments as the
// RAW JSON string the model emitted: a JSON-string `arguments` is unquoted to its
// inner text, an object/array is kept verbatim, and null/empty becomes "".
func (f *Func) UnmarshalJSON(raw []byte) error {
	var aux struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(raw, &aux); err != nil {
		return err
	}
	f.Name = aux.Name
	f.Arguments = ""
	arg := bytes.TrimSpace(aux.Arguments)
	if len(arg) == 0 || bytes.Equal(arg, []byte("null")) {
		return nil
	}
	if arg[0] == '"' {
		var s string
		if err := json.Unmarshal(arg, &s); err != nil {
			return err
		}
		f.Arguments = s
		return nil
	}
	f.Arguments = string(arg)
	return nil
}

// Message is one chat-completions message (request or response).
type Message struct {
	Role         string     `json:"role"`
	Content      string     `json:"content,omitempty"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
	FunctionCall *Func      `json:"function_call,omitempty"` // legacy OpenAI-compatible single-call shape
	ToolCallID   string     `json:"tool_call_id,omitempty"`  // for role=tool
	Name         string     `json:"name,omitempty"`

	// Thinking carries a Claude extended-thinking ("thinking") content block
	// through the proxy instead of dropping it; ThinkingSignature is the opaque
	// signature the Anthropic API requires to round-trip the block back upstream
	// on a later turn. RedactedThinking holds any redacted_thinking blocks verbatim
	// (encrypted reasoning that must be echoed back unmodified). All three are
	// additive over the OpenAI shape; an OpenAI client simply ignores them.
	Thinking          string   `json:"thinking,omitempty"`
	ThinkingSignature string   `json:"thinking_signature,omitempty"`
	RedactedThinking  []string `json:"redacted_thinking,omitempty"`

	// ReasoningContent carries the reasoning block a Qwen3.5-style reasoning model
	// (e.g. Ornith) emits inside <think>…</think>, split off the in-kernel decode path
	// so it does not leak into Content (and thus into downstream Claude Code context).
	// It mirrors the `reasoning_content` field vLLM produces with --reasoning-parser
	// qwen3. Additive over the OpenAI shape; an OpenAI client simply ignores it.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// UnmarshalJSON decodes a chat message, flattening a `content` field that may be a
// plain string OR an array of typed content parts into a single text Content while
// carrying the tool-call, function-call, and Claude thinking fields through unchanged.
func (m *Message) UnmarshalJSON(raw []byte) error {
	var aux struct {
		Role              string          `json:"role"`
		Content           json.RawMessage `json:"content"`
		ToolCalls         []ToolCall      `json:"tool_calls,omitempty"`
		FunctionCall      *Func           `json:"function_call,omitempty"`
		ToolCallID        string          `json:"tool_call_id,omitempty"`
		Name              string          `json:"name,omitempty"`
		Thinking          string          `json:"thinking,omitempty"`
		ThinkingSignature string          `json:"thinking_signature,omitempty"`
		RedactedThinking  []string        `json:"redacted_thinking,omitempty"`
	}
	if err := json.Unmarshal(raw, &aux); err != nil {
		return err
	}
	content, err := contentText(aux.Content)
	if err != nil {
		return err
	}
	m.Role = aux.Role
	m.Content = content
	m.ToolCalls = aux.ToolCalls
	m.FunctionCall = aux.FunctionCall
	m.ToolCallID = aux.ToolCallID
	m.Name = aux.Name
	m.Thinking = aux.Thinking
	m.ThinkingSignature = aux.ThinkingSignature
	m.RedactedThinking = aux.RedactedThinking
	return nil
}

// trimmedTextScalar handles the two leaf cases both contentText and contentPartText
// share: empty / JSON null trims to "", and a JSON string literal decodes to its value.
// done is true when one of those cases applied (s and err are then authoritative); when
// done is false the caller inspects the remaining trimmed bytes itself.
func trimmedTextScalar(raw json.RawMessage) (rest json.RawMessage, s string, done bool, err error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return raw, "", true, nil
	}
	if raw[0] == '"' {
		if e := json.Unmarshal(raw, &s); e != nil {
			return raw, "", true, e
		}
		return raw, s, true, nil
	}
	return raw, "", false, nil
}

func contentText(raw json.RawMessage) (string, error) {
	raw, s, done, err := trimmedTextScalar(raw)
	if done {
		return s, err
	}
	if raw[0] == '[' {
		var parts []json.RawMessage
		if err := json.Unmarshal(raw, &parts); err != nil {
			return "", err
		}
		texts := make([]string, 0, len(parts))
		for _, part := range parts {
			text, err := contentPartText(part)
			if err != nil {
				return "", err
			}
			if text != "" {
				texts = append(texts, text)
			}
		}
		return strings.Join(texts, "\n"), nil
	}
	return string(raw), nil
}

func contentPartText(raw json.RawMessage) (string, error) {
	raw, s, done, err := trimmedTextScalar(raw)
	if done {
		return s, err
	}
	var part struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &part); err != nil {
		return "", err
	}
	if part.Text != "" {
		return part.Text, nil
	}
	return part.Content, nil
}

// ToolDef is an OpenAI function/tool declaration advertised to the model.
type ToolDef struct {
	Type     string          `json:"type"` // always "function"
	Function ToolDefFunction `json:"function"`
}

// ToolDefFunction is the function half of a ToolDef: the tool name, its description,
// and its parameter JSON Schema as advertised to the model.
type ToolDefFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // JSON Schema
}

// Usage is the token accounting a completion reports.
type Usage struct {
	PromptTokens             int                `json:"prompt_tokens"`
	CompletionTokens         int                `json:"completion_tokens"`
	TotalTokens              int                `json:"total_tokens"`
	PromptTokensDetails      *UsageTokenDetails `json:"prompt_tokens_details,omitempty"`
	InputTokensDetails       *UsageTokenDetails `json:"input_tokens_details,omitempty"`
	CacheReadInputTokens     int                `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int                `json:"cache_creation_input_tokens,omitempty"`
}

// UsageTokenDetails carries provider-specific prompt/input token subcounters.
type UsageTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CachedPromptTokens is the provider-reported prompt-cache hit count, normalized
// across OpenAI chat-completions, OpenAI Responses, and Anthropic-style counters.
func (u Usage) CachedPromptTokens() int {
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		return u.PromptTokensDetails.CachedTokens
	}
	if u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0 {
		return u.InputTokensDetails.CachedTokens
	}
	if u.CacheReadInputTokens > 0 {
		return u.CacheReadInputTokens
	}
	return 0
}

// UncachedPromptTokens is the prompt the model actually re-ingested this turn — the
// full prompt minus the provider's cache-read hit — normalized so the count means the
// same thing across providers. Anthropic already reports prompt/input_tokens as the
// UNCACHED remainder (cache_read_input_tokens is a separate field), so it is returned
// as-is. OpenAI (chat + Responses) and Gemini fold the cached hit INTO prompt_tokens,
// so the cached portion is peeled back off to leave the uncached remainder. The result
// is never negative, and UncachedPromptTokens() + CachedPromptTokens() == the full
// resident prompt on every provider. This is the companion of CachedPromptTokens(): a
// consumer that splits a turn into (uncached, cached) — e.g. the vCache observe plane's
// baseline-token-equiv — gets a provider-consistent split from the pair.
func (u Usage) UncachedPromptTokens() int {
	n := u.PromptTokens
	// OpenAI/Gemini shape: prompt_tokens INCLUDES the cache-read hit (reported in
	// prompt_tokens_details/input_tokens_details), so subtract it to match Anthropic's
	// already-uncached input_tokens. The Anthropic shape carries its cache read in the
	// separate CacheReadInputTokens field and is left untouched.
	if (u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0) ||
		(u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0) {
		n -= u.CachedPromptTokens()
	}
	if n < 0 {
		n = 0
	}
	return n
}

// ContextWindowTokens is the prompt/context size that should count against a
// long-session context budget. OpenAI-style prompt_tokens already include cached
// prompt tokens, so their details are NOT added again. Anthropic reports
// input_tokens as the uncached remainder and cache_read/cache_creation separately;
// those counters are added back so the budget reflects the full context the model
// attended to.
func (u Usage) ContextWindowTokens() int {
	n := u.PromptTokens
	if u.CacheReadInputTokens > 0 {
		n += u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens > 0 {
		n += u.CacheCreationInputTokens
	}
	return n
}

// Completion is a planner's response for one turn.
type Completion struct {
	Message            Message
	FinishReason       string
	Usage              Usage
	ProviderCache      *cachemeta.Entry
	Raw                []byte // the raw response body (transcript witness for the live seam)
	PreSendQuarantines int    // tool-result payloads held out before provider serialization
	// PreSendRedactions counts the outbound messages whose content was span-redacted
	// (rung 5, #572) before provider serialization on the re-marshal path. It mirrors
	// PreSendQuarantines so a caller can observe that something was redacted, not only
	// that something was held out. Zero on the default-inert path (FAK_WIRE_REDACT
	// unset → wirescreen.ActiveRedactor() nil) and on the Anthropic raw-passthrough
	// path (which forwards req.Raw verbatim and never re-marshals these messages).
	PreSendRedactions int
	// PreSendRedactionRecords are the full reversible records behind PreSendRedactions
	// (#882): each carries the message index, the redactor, the redacted spans, and a
	// CAS handle to the UNREDACTED original (wirescreen.Restore(ctx, .Original) returns
	// it byte-exact) — the reversible-on-audit data a count alone cannot give. Nil on
	// the default-inert and Anthropic-passthrough paths, exactly like the count.
	PreSendRedactionRecords []TranscriptRedaction

	// Model is the model id the UPSTREAM reported it served this completion with
	// (the provider response's `model` field), or "" when the provider omitted it.
	// The /v1/chat/completions proxy echoes this as the response `model` so a client
	// sees what actually served its request, not merely what the gateway is
	// configured for — the response half of the request-model pass-through (#82).
	Model string

	// ToolCallsDropped is the tool-call CONFORMANCE signal: the upstream's raw
	// finish_reason said it was making tool calls ("tool_calls" / "function_call")
	// but ZERO structured calls survived parsing + the text-lift fallback. That is
	// the silent-no-op a non-OpenAI-shaped emitter (e.g. a GLM-5.2 variant that
	// buries calls in reasoning_content or a non-standard wrapper) would cause:
	// the agent would proceed as if no tool was invoked and adjudication would be
	// skipped. Callers MUST treat a dropped turn as a fail-closed condition, not a
	// benign empty turn — the kernel's permission floor must never be bypassed by a
	// format it failed to parse. Set by normalizeCompletionToolCalls.
	ToolCallsDropped bool
}

// SampleParams are the per-request sampling overrides a CALLER may attach to one
// Complete turn. A nil pointer / nil slice means "the caller did not specify this"
// — the planner keeps its configured default, so an omitted field is byte-for-byte
// the pre-seam behavior. The pointer fields (not bare values) are what let an
// EXPLICIT temperature:0 be distinguished from an omitted one: a fixed-default
// planner like HTTPPlanner already runs temperature 0, so the two only differ when
// the caller also wants top_p/stop, and a bare float64 could not carry that intent.
type SampleParams struct {
	// Model, when non-empty, overrides the planner's configured ModelID for THIS
	// request — the gateway's request-model pass-through (#82). It is the model id
	// that reaches the upstream request body (and, for a path-templated provider
	// like Gemini, the upstream URL), so a client asking for a model the gateway was
	// not configured with reaches the provider verbatim and an unknown model
	// surfaces the provider's own 404 instead of being silently served by the
	// default model. Empty => the planner keeps its configured ModelID (the client
	// omitted `model`), which stays the advertised /v1/models id and default.
	Model       string
	MaxTokens   *int     // output-token ceiling (the #62 hard-cap; nil => planner default)
	Temperature *float64 // sampling temperature (nil => planner default)
	TopP        *float64 // nucleus sampling (nil => unset on the wire)
	TopK        *int     // top-k truncation (nil => unset; <=0 => no truncation)
	Stop        []string // stop sequences (empty => unset on the wire)
	// ResponseFormat is the OpenAI structured-output carrier (the #560 guided-decode
	// seam): the raw `response_format` object the client sent (a json_object or a
	// json_schema spec). Empty => unset on the wire, byte-for-byte the pre-seam body.
	// On the ride path it forwards verbatim so a ridden engine (vLLM/SGLang) enforces
	// the schema; the whole-turn adjudication gate still runs on the constrained output.
	ResponseFormat json.RawMessage
	// LogitBias is the OpenAI per-token logit-bias map (token id -> bias, the standard
	// -100..100 mask). Empty => unset on the wire. Like ResponseFormat it rides verbatim
	// to the upstream so the engine applies the mask at its own logit step; the native
	// in-kernel mask is a sibling-lane (internal/model) concern, out of this seam.
	LogitBias map[int]float64
	// FrequencyPenalty is the OpenAI per-request frequency penalty (nil => planner
	// default / unset on the wire). Subtracted from each candidate token's logit
	// scaled by how many times that token has already been generated this turn — see
	// sampleLogitsWithPenalty. A nil pointer (including the common all-defaults
	// request) is byte-for-byte the pre-penalty sampler behavior.
	FrequencyPenalty *float64
	// PresencePenalty is the OpenAI per-request presence penalty (nil => planner
	// default / unset on the wire). Subtracted once from a candidate token's logit
	// if that token has appeared at all this turn (count>0), independent of how many
	// times — see sampleLogitsWithPenalty. nil is a no-op.
	PresencePenalty *float64
	// GuidedDecode carries provider-native guided-decode fields that are not part of
	// the OpenAI core wire but are accepted by OpenAI-compatible ride engines such as
	// vLLM/SGLang (`guided_json`, `guided_regex`, `guided_grammar`, `guided_choice`,
	// `json_schema`, `regex`, `ebnf`). Empty => unset on the wire. The gateway only
	// populates this map from an allowlist, so client unknowns are still ignored.
	GuidedDecode map[string]json.RawMessage
	// RawRequestBody, when non-empty, is sent to the upstream VERBATIM instead of a
	// freshly-marshalled body — the anthropic→anthropic passthrough path. Forwarding
	// the client's ORIGINAL bytes preserves its prompt-cache prefix (so the upstream
	// returns a real cache hit, not a re-billed prefix). It makes the other sampling
	// fields no-ops by construction (the client's own values are already in the bytes).
	RawRequestBody []byte
	// UpstreamAPIKey, when non-empty, overrides the planner's configured key for THIS
	// request — the transparent-hop credential on the passthrough path, where the
	// inbound client authenticates directly against the real upstream with its own key.
	UpstreamAPIKey string
	// UpstreamBeta, when non-empty, is merged into the upstream "anthropic-beta"
	// header (Anthropic wire only) — the inbound client's own beta flags forwarded
	// on the passthrough hop so features it negotiated (extended thinking,
	// fine-grained tool streaming, the oauth subscription path) survive. It is
	// UNIONED with any scheme-required beta the adapter already set (e.g. the OAuth
	// flag), deduped, so neither clobbers the other. A no-op off the Anthropic wire.
	UpstreamBeta string
}

// SampleOpt is a functional option that mutates a SampleParams. The variadic
// option shape keeps Complete's signature additive: every existing call site —
// the A/B loop, the injector decorator, the mock — compiles unchanged, and only
// the gateway (which actually has a client request to forward) passes options.
type SampleOpt func(*SampleParams)

// WithModel overrides the planner's configured model id for this one request — the
// gateway's request-model pass-through (#82). An empty string is a NO-OP, so a
// caller can forward a client's raw `model` field unconditionally: an omitted model
// arrives as "" and falls through to the planner's configured ModelID (which stays
// the advertised /v1/models id and the default when the client names no model).
func WithModel(model string) SampleOpt {
	return func(sp *SampleParams) {
		if model != "" {
			sp.Model = model
		}
	}
}

// WithMaxTokens sets the per-request output-token ceiling. It is a NO-OP for n<=0
// so a caller can forward a client's raw value unconditionally: an omitted
// max_tokens arrives as 0 and naturally falls through to the planner default.
func WithMaxTokens(n int) SampleOpt {
	return func(sp *SampleParams) {
		if n > 0 {
			sp.MaxTokens = &n
		}
	}
}

// WithTemperature sets the per-request temperature. The pointer argument carries
// the omitted/explicit distinction straight through: a nil t is a no-op (keep the
// default), a non-nil t (including a pointer to 0) sets it explicitly.
func WithTemperature(t *float64) SampleOpt {
	return func(sp *SampleParams) {
		if t != nil {
			v := *t
			sp.Temperature = &v
		}
	}
}

// WithTopP sets the per-request nucleus-sampling cutoff. nil is a no-op.
func WithTopP(p *float64) SampleOpt {
	return func(sp *SampleParams) {
		if p != nil {
			v := *p
			sp.TopP = &v
		}
	}
}

// WithTopK sets the per-request top-k truncation (keep only the k highest-logit
// tokens before the draw). nil is a no-op; a non-nil k<=0 explicitly disables
// truncation, matching the planner's "0 => full distribution" convention.
func WithTopK(k *int) SampleOpt {
	return func(sp *SampleParams) {
		if k != nil {
			v := *k
			sp.TopK = &v
		}
	}
}

// WithStop sets the per-request stop sequences. An empty/nil slice is a no-op.
func WithStop(s []string) SampleOpt {
	return func(sp *SampleParams) {
		if len(s) > 0 {
			sp.Stop = s
		}
	}
}

// WithResponseFormat sets the per-request OpenAI `response_format` carrier (the
// #560 structured/guided-decode seam) from the raw object the client sent. An
// empty/nil slice is a no-op so a caller can forward a client's value
// unconditionally: an omitted response_format stays absent from the wire,
// byte-for-byte the pre-seam request.
func WithResponseFormat(raw json.RawMessage) SampleOpt {
	return func(sp *SampleParams) {
		if len(raw) > 0 {
			sp.ResponseFormat = raw
		}
	}
}

// WithLogitBias sets the per-request OpenAI `logit_bias` map (token id -> bias).
// An empty/nil map is a no-op, so an omitted logit_bias stays absent from the wire.
func WithLogitBias(bias map[int]float64) SampleOpt {
	return func(sp *SampleParams) {
		if len(bias) > 0 {
			sp.LogitBias = bias
		}
	}
}

// WithFrequencyPenalty sets the per-request OpenAI frequency penalty. nil is a
// no-op (keep the planner default); a non-nil p (including a pointer to 0) sets it
// explicitly, matching the WithTemperature/WithTopP pointer-carries-omitted pattern.
func WithFrequencyPenalty(p *float64) SampleOpt {
	return func(sp *SampleParams) {
		if p != nil {
			v := *p
			sp.FrequencyPenalty = &v
		}
	}
}

// WithPresencePenalty sets the per-request OpenAI presence penalty. nil is a no-op;
// a non-nil p sets it explicitly, matching WithFrequencyPenalty.
func WithPresencePenalty(p *float64) SampleOpt {
	return func(sp *SampleParams) {
		if p != nil {
			v := *p
			sp.PresencePenalty = &v
		}
	}
}

// WithGuidedDecode sets the per-request provider-native guided-decode carriers.
// It is intentionally narrower than RawRequestBody/ExtraBody: callers pass only the
// allowlisted structured-output fields parsed from the client request, and the
// planner merges them into the OpenAI-compatible ride-engine body.
func WithGuidedDecode(fields map[string]json.RawMessage) SampleOpt {
	return func(sp *SampleParams) {
		if len(fields) == 0 {
			return
		}
		sp.GuidedDecode = make(map[string]json.RawMessage, len(fields))
		for k, v := range fields {
			if len(v) == 0 {
				continue
			}
			sp.GuidedDecode[k] = append(json.RawMessage(nil), v...)
		}
		if len(sp.GuidedDecode) == 0 {
			sp.GuidedDecode = nil
		}
	}
}

// WithRawRequestBody forwards the client's ORIGINAL request bytes to the upstream
// verbatim (the anthropic→anthropic passthrough path), preserving its prompt-cache
// prefix. An empty slice is a no-op (the planner marshals a fresh body as usual).
func WithRawRequestBody(raw []byte) SampleOpt {
	return func(sp *SampleParams) {
		if len(raw) > 0 {
			sp.RawRequestBody = raw
		}
	}
}

// WithUpstreamAPIKey overrides the planner's configured key for this one request —
// the transparent-hop credential on the passthrough path. An empty string is a no-op.
func WithUpstreamAPIKey(key string) SampleOpt {
	return func(sp *SampleParams) {
		if key != "" {
			sp.UpstreamAPIKey = key
		}
	}
}

// WithUpstreamBeta forwards the inbound client's "anthropic-beta" header to the
// upstream on the passthrough hop (Anthropic wire only). An empty string is a no-op.
func WithUpstreamBeta(beta string) SampleOpt {
	return func(sp *SampleParams) {
		if beta != "" {
			sp.UpstreamBeta = beta
		}
	}
}

// forceAnthropicNonStreaming returns the raw Anthropic request body with its
// top-level "stream" flag set to false, so the passthrough upstream returns a
// buffered JSON body (which this non-streaming planner can parse) rather than an
// SSE event stream. A body that carries NO stream field is returned UNCHANGED
// (byte-identical), so the common non-streaming case keeps its exact cache prefix;
// only a streaming body is re-marshalled, and the cached prefix is the
// system/tools/messages content — unaffected by the top-level key order or the
// stream flag. A body that does not parse as a JSON object is returned unchanged
// (the planner then surfaces the upstream's own error).
func forceAnthropicNonStreaming(raw []byte) []byte {
	return setAnthropicStreamFlag(raw, "false", func(_ json.RawMessage, present bool) bool {
		return !present // a body with no stream field is already non-streaming — keep its exact prefix
	})
}

// setAnthropicStreamFlag returns raw with its top-level "stream" key set to value,
// re-marshalling the object — UNLESS skip reports the rewrite is unnecessary (the body
// is already in the wanted state), in which case raw is returned byte-identical so the
// provider cache prefix survives. A body that does not parse as a JSON object, or that
// fails to re-marshal, is returned unchanged. It is the shared core of the streaming /
// non-streaming forcing pair; only the target value and the skip predicate differ.
func setAnthropicStreamFlag(raw []byte, value string, skip func(v json.RawMessage, present bool) bool) []byte {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw
	}
	v, present := obj["stream"]
	if skip(v, present) {
		return raw
	}
	obj["stream"] = json.RawMessage(value)
	out, err := json.Marshal(obj)
	if err != nil {
		return raw
	}
	return out
}

// mergeBeta unions two comma-separated "anthropic-beta" header values, preserving
// first-seen order and dropping duplicates and blanks. Either side may be empty.
// It is how a scheme-required flag (e.g. the OAuth beta the adapter sets) and the
// inbound client's negotiated betas coexist in one header without one overwriting
// the other.
func mergeBeta(a, b string) string {
	seen := map[string]bool{}
	var out []string
	for _, part := range strings.Split(a+","+b, ",") {
		t := strings.TrimSpace(part)
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return strings.Join(out, ",")
}

// applySampleOpts folds a variadic option list into a single SampleParams. An
// empty list yields the zero value (every field nil/empty == "all defaults").
func applySampleOpts(opts ...SampleOpt) SampleParams {
	var sp SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&sp)
		}
	}
	return sp
}

// Planner is the seam both the live HTTP client and the offline mock satisfy. One
// Complete call == one model TURN.
type Planner interface {
	// Complete sends the running message list + the tool catalog and returns the
	// assistant's next message (tool calls or a final answer). The optional
	// SampleOpts carry per-request sampling overrides (max_tokens, temperature,
	// top_p, top_k, stop) plus the structured/guided-decode carriers (response_format,
	// logit_bias, provider-native guided fields); with none passed, the planner uses
	// its configured defaults.
	Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error)
	// Model is the model id (for provenance).
	Model() string
}

// KVMemoryStats is an optional planner-owned snapshot of local KV-cache residency.
// It is separate from Usage cache-read counters: those count work saved on a turn,
// while this reports resident KV memory pressure in the local process. Planners that
// proxy an upstream model do not implement it; the gateway emits no resident-KV series
// for them rather than publishing a fake zero.
type KVMemoryStats struct {
	Enabled            bool   // true when a reusable local KV cache is active
	Backend            string // radixkv, device backend name, or empty when unknown
	MemoryClass        string // kv_cache
	Scope              string // host/device
	DType              string // storage dtype for the local KV rows, currently f32 for HAL KV
	BytesPerToken      int64  // bytes per resident KV position under this model layout
	ResidentTokens     int    // true resident prefix positions, not the LRU edge-token budget
	ResidentBytes      int64
	CapacityKnown      bool
	CapacityFreeKnown  bool
	CapacityTotalBytes int64
	CapacityFreeBytes  int64
	HeadroomRatio      float64
	FitBudgetBytes     int64
	FitMarginBytes     int64
	BudgetTokens       int // configured LRU budget metric; 0 means unbounded or unavailable
	LRUTokens          int // Σ edge lengths, the budget metric radixkv enforces
	MaxDepthTokens     int
	Nodes              int
	Leaves             int
	Evictions          int
	PolicyEvictions    int
	Splits             int
}

// KVMemoryReporter is the optional interface a local planner implements when it
// can report resident KV-cache memory state.
type KVMemoryReporter interface {
	KVMemoryStats() KVMemoryStats
}

// RequestMemoryDemand is one row from the most recent local request memory plan.
// It mirrors compute.MemoryDemand without making gateway depend on compute types.
type RequestMemoryDemand struct {
	Class  string
	Scope  string
	DType  string
	Bytes  int64
	Detail string
}

type RequestMemoryCapacity struct {
	Scope      string
	TotalBytes int64
	FreeBytes  int64
	Known      bool
	FreeKnown  bool
}

// RequestMemoryStats is the optional planner-owned snapshot of the last in-kernel
// request admission plan. It reports successful plans too, so request memory pressure
// is visible before an OOM happens.
type RequestMemoryStats struct {
	Observed      bool
	Backend       string
	PromptTokens  int
	MaxNewTokens  int
	PlannedTokens int
	HeadroomRatio float64
	MemoryPlan    []RequestMemoryDemand
	Capacities    []RequestMemoryCapacity
}

// RequestMemoryReporter is implemented by local planners that can report their last
// request-time memory plan. Proxy planners do not implement it, so the gateway emits no
// local request-memory series for upstream providers.
type RequestMemoryReporter interface {
	RequestMemoryStats() RequestMemoryStats
}

// InKernelOOMRetryClassStats is one bounded-label row for decode retries that were
// attempted after a local in-kernel device allocation OOM.
type InKernelOOMRetryClassStats struct {
	Class           string
	Attempts        uint64
	Successes       uint64
	Failures        uint64
	LastFailedBytes uint64
	LastSite        string
}

// InKernelOOMRetryStats is the optional planner-owned snapshot of idle-pool trim retries
// after in-kernel device allocation OOMs. It is intentionally class-bucketed; allocator
// sites stay out of Prometheus labels and are exposed only in debug output.
type InKernelOOMRetryStats struct {
	Backend string
	Rows    []InKernelOOMRetryClassStats
}

// InKernelOOMRetryReporter is implemented by local planners that can report in-kernel
// OOM retry attempts. Proxy planners do not implement it.
type InKernelOOMRetryReporter interface {
	InKernelOOMRetryStats() InKernelOOMRetryStats
}

// InKernelMemoryPressureTrimClassStats is one bounded-label row for proactive
// memory-pressure trims before a served in-kernel device decode enters allocation-heavy
// work. "resolved" means a capacity-precheck refusal fit after the trim.
type InKernelMemoryPressureTrimClassStats struct {
	Scope           string
	Class           string
	Reason          string
	Attempts        uint64
	Trimmed         uint64
	NoHooks         uint64
	Resolved        uint64
	LastWantBytes   uint64
	LastBudgetBytes uint64
	LastMarginBytes int64
}

// InKernelMemoryPressureTrimStats reports proactive idle-pool trims triggered by
// known request-memory pressure. It is separate from OOM retry stats: these happen
// before decode allocation, not after a recovered DeviceAllocError.
type InKernelMemoryPressureTrimStats struct {
	Backend string
	Rows    []InKernelMemoryPressureTrimClassStats
}

// InKernelMemoryPressureTrimReporter is implemented by local planners that can
// report proactive memory-pressure trims. Proxy planners do not implement it.
type InKernelMemoryPressureTrimReporter interface {
	InKernelMemoryPressureTrimStats() InKernelMemoryPressureTrimStats
}

// ---------------------------------------------------------------------------
// Live planner — provider API client.
// ---------------------------------------------------------------------------

// HTTPPlanner drives closed-API and OpenAI-compatible chat endpoints through a
// provider transcript adapter. base_url selects the provider root; Provider
// selects the wire shape.
type HTTPPlanner struct {
	BaseURL string
	ModelID string
	APIKey  string
	// APIKeyFunc, when non-nil, supplies the upstream credential FRESH on every request
	// instead of the frozen APIKey string. It is the fix for a short-lived bearer (a Claude
	// Pro/Max subscription OAuth access token, which the provider rotates roughly hourly):
	// a planner built once at `fak guard` startup would otherwise pin the boot-time token
	// for the whole session and 401 the moment it expires — even after the user re-logs in,
	// because the refreshed token lands in the on-disk credential file the frozen string
	// never re-reads. With APIKeyFunc set, the auth path re-resolves the token per request,
	// so a long session always sends the live credential. A non-empty per-request
	// UpstreamAPIKey (the transparent passthrough hop) still wins over both; an empty/failed
	// APIKeyFunc result falls back to the static APIKey. nil leaves the static-key path
	// byte-for-byte unchanged.
	APIKeyFunc           func() string
	Provider             Provider
	Adapter              TranscriptAdapter
	ExtraBody            json.RawMessage
	Temperature          float64
	MaxTokens            int
	Client               *http.Client
	QuarantineTranscript bool

	// CoherenceShaper, when non-nil, is applied to the outbound messages just before
	// the request is marshaled — the GLM52-HOSTED-CACHE-COHERENCE §A4 hook. The agent
	// loop sets it to a closure that runs SegmentsFromMessages -> ShapeGLMTurnSegment
	// Witnessed(..., vdso.Default.Revoked) and re-emits the shaped turn, so a refuted
	// world witness breaks the now-stale provider-prefix span. nil = behavior unchanged
	// (the default): no shaping, byte-for-byte the prior request path.
	CoherenceShaper func([]Message) []Message

	// RetryNotify, when non-nil, is called ONCE before each retry of Complete's backoff loop
	// (i.e. on attempt 1..N-1, never on the first try), with the upcoming attempt index, the
	// status that triggered the retry (the upstream HTTP status for a 429/5xx, or 0 for a
	// transient transport error), and the backoff wait about to elapse. It is the observability
	// hook for the otherwise-INVISIBLE retry window: a 429/5xx storm used to burn up to ~8s of
	// silent backoff with no log, metric, or debug line. The gateway sets it to a closure that
	// bumps a retry counter and prints a `fak-turn … retry` debug line, so an operator sees the
	// backoff happening instead of a frozen terminal. nil = behavior byte-for-byte unchanged.
	RetryNotify func(attempt int, status int, wait time.Duration)

	// AuthRefreshNotify, when non-nil, is called when a 401 on the rotating-subscription path
	// is handled — separately from RetryNotify so a token-expiry self-heal is never conflated
	// with a 429/5xx backoff (different cause, different metric). outcome is "recovered" when a
	// fresh token was adopted and the call re-sent in place (the live session healed across a
	// re-login), or "exhausted" when no fresher token appeared within the grace window and the
	// 401 is about to surface to the wrapped agent (the session is about to drop into its own
	// /login). It is the observability hook for the otherwise-INVISIBLE token-rotation event —
	// the single most operationally important guard credential signal. The gateway sets it to a
	// closure that bumps a per-outcome counter and prints a "fak-turn auth-refresh" line. nil =
	// behavior byte-for-byte unchanged (the self-heal itself is independent of the hook).
	AuthRefreshNotify func(outcome string, attempt int)
}

// NewHTTPPlanner builds a live planner with a bounded timeout. The per-request
// timeout defaults to 60s but is overridable via FAK_PLANNER_TIMEOUT_S — a small
// CPU-served local model (e.g. a 3B through the transformers shim) can take
// minutes per turn, so the benchmark needs a longer ceiling than a hosted API.
func NewHTTPPlanner(baseURL, model, apiKey string) *HTTPPlanner {
	return &HTTPPlanner{
		BaseURL:              baseURL,
		ModelID:              model,
		APIKey:               apiKey,
		Provider:             ProviderOpenAI,
		Temperature:          0, // as deterministic as a live model allows
		MaxTokens:            1024,
		Client:               &http.Client{Timeout: plannerTimeout()},
		QuarantineTranscript: true,
	}
}

// NewProviderHTTPPlanner selects a native provider transcript adapter while
// preserving NewHTTPPlanner's OpenAI-compatible default.
func NewProviderHTTPPlanner(provider, baseURL, model, apiKey string) (*HTTPPlanner, error) {
	pv, ok := ParseProvider(provider)
	if !ok {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	p := NewHTTPPlanner(baseURL, model, apiKey)
	p.Provider = pv
	if raw := os.Getenv("FAK_PROVIDER_EXTRA_BODY_JSON"); raw != "" {
		if err := p.SetExtraBodyJSON(raw); err != nil {
			return nil, err
		}
	}
	return p, nil
}

// effectiveAPIKey is the credential the upstream hop authenticates with when the
// caller did not supply a per-request override. It prefers a live APIKeyFunc result
// (the rotating-token path — see APIKeyFunc) and falls back to the static APIKey when
// the func is nil or returns empty, so a transient credential-read miss degrades to the
// boot-time token rather than dropping auth entirely.
func (p *HTTPPlanner) effectiveAPIKey() string {
	if p.APIKeyFunc != nil {
		if k := p.APIKeyFunc(); k != "" {
			return k
		}
	}
	return p.APIKey
}

// SetExtraBodyJSON validates and installs provider-specific top-level request
// fields. It is intentionally additive: callers cannot override the canonical
// model/messages/tools fields that the adapter owns.
func (p *HTTPPlanner) SetExtraBodyJSON(raw string) error {
	extra, err := ParseExtraBodyJSON(raw)
	if err != nil {
		return err
	}
	p.ExtraBody = extra
	return nil
}

// ParseExtraBodyJSON validates a JSON object that will be merged into
// OpenAI-compatible request bodies for serving engines such as vLLM/SGLang.
func ParseExtraBodyJSON(raw string) (json.RawMessage, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return nil, fmt.Errorf("provider extra body: %w", err)
	}
	if obj == nil {
		return nil, fmt.Errorf("provider extra body must be a JSON object")
	}
	for k := range obj {
		if reservedExtraBodyKey(k) {
			return nil, fmt.Errorf("provider extra body must not override %q", k)
		}
	}
	normalized, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return normalized, nil
}

func mergeGuidedDecodeExtraBody(extra json.RawMessage, guided map[string]json.RawMessage) (json.RawMessage, error) {
	if len(guided) == 0 {
		return extra, nil
	}
	obj := map[string]json.RawMessage{}
	if len(extra) > 0 {
		if err := json.Unmarshal(extra, &obj); err != nil {
			return nil, fmt.Errorf("provider extra body: %w", err)
		}
	}
	for k, v := range guided {
		if len(v) == 0 {
			continue
		}
		if reservedExtraBodyKey(k) {
			return nil, fmt.Errorf("provider extra body must not override %q", k)
		}
		if _, exists := obj[k]; exists {
			return nil, fmt.Errorf("provider extra body must not override %q", k)
		}
		obj[k] = append(json.RawMessage(nil), v...)
	}
	if len(obj) == 0 {
		return nil, nil
	}
	normalized, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return normalized, nil
}

func reservedExtraBodyKey(k string) bool {
	switch strings.ToLower(strings.TrimSpace(k)) {
	case "model", "messages", "input", "tools", "tool_choice", "temperature",
		"max_tokens", "max_output_tokens", "top_p", "stop", "stop_sequences",
		"stream", "stream_options", "store":
		return true
	default:
		return false
	}
}

// plannerTimeout is the per-request HTTP timeout, 60s unless FAK_PLANNER_TIMEOUT_S
// overrides it (clamped to a sane [5s, 1h] band).
func plannerTimeout() time.Duration {
	return envClampedTimeout("FAK_PLANNER_TIMEOUT_S", 60*time.Second, 5, 3600)
}

// Model returns the planner's configured model id (for provenance).
func (p *HTTPPlanner) Model() string { return p.ModelID }

// Complete performs one chat-completions round-trip with one backoff retry on a
// transport error. The optional SampleOpts override the planner's configured
// sampling defaults for THIS request only: a caller-supplied max_tokens replaces
// the fixed 1024 ceiling, temperature/top_p/top_k/stop are forwarded to the provider
// wire. top_k rides only on the providers with a native field (Anthropic, Gemini);
// OpenAI/xAI/Responses have none, so a top_k for them must go via ExtraBody. An
// omitted field keeps the planner default, so a no-opt call is identical to the
// pre-seam behavior.
func (p *HTTPPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	call, err := p.prepareUpstream(messages, tools, false, opts...)
	if err != nil {
		return nil, err
	}
	// Retry on a TRANSIENT transport error OR a retryable status (429 rate-limit,
	// 5xx overload) with exponential backoff — the live-API-limit failure mode. A
	// 4xx other than 429 is a request error and is NOT retried. A DETERMINISTIC
	// transport failure (connection refused, DNS NXDOMAIN, TLS handshake) is a
	// misconfiguration that a retry cannot fix, so it fails fast without burning the
	// ~8s backoff budget and is tagged so the gateway can surface its cause (#346).
	// maxAttempts is the TOTAL number of tries (first attempt + retries). The default is
	// deliberately generous and operator-tunable (FAK_PLANNER_MAX_ATTEMPTS): a fleet
	// sharing one upstream account rides out a long 429/529 overload window far better with
	// more, longer-spaced retries than with a fast give-up.
	maxAttempts, deadline, budgetOn := retryBounds(time.Now())
	var lastErr error
	// lastStatusErr holds the last error that carried a real upstream HTTP status (and any
	// Retry-After). It is kept SEPARATE from lastErr and is NEVER cleared by a subsequent
	// transient transport error, so a network glitch on a later attempt cannot shadow the
	// 429/503/529 that actually drove the failure: on exhaustion we surface the real status
	// (and Retry-After), not an opaque 502 (#1358).
	var lastStatusErr *UpstreamStatusError
	lastStatus := 0      // the status that triggered the pending retry (0 = a transient transport error)
	lastRetryAfter := "" // the triggering response's Retry-After header, honored as the next wait
	lastCapWait := ""    // classified account-cap wait (#1362): toward the named reset when Retry-After is absent
	// A 401 on the pinned/rotating subscription path is recoverable ONCE: the on-disk
	// OAuth token may have rotated (or been briefly torn) between resolve and send, so we
	// re-read it fresh and retry. triedAuthRefresh caps that at a single extra attempt so a
	// genuinely-bad credential still fails fast instead of looping.
	triedAuthRefresh := false
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Surface the retry BEFORE the silent backoff sleep, then wait; when the TIME
			// budget is the bound a spent budget stops the loop (surface the last error) and a
			// cancelled context returns promptly. See retryBackoffWait for the shared step.
			stop, err := p.retryBackoffWait(ctx, attempt, lastStatus, lastRetryAfter, lastCapWait, deadline, budgetOn)
			if err != nil {
				return nil, err
			}
			if stop {
				break
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", call.url, bytes.NewReader(call.body))
		if err != nil {
			return nil, err
		}
		call.applyHeaders(req)
		resp, err := p.Client.Do(req)
		if err != nil {
			// A deterministic dial-time failure (refused / NXDOMAIN / TLS) will not
			// resolve on retry — retrying only adds ~8s of backoff latency to what is a
			// configuration error. Fail fast and tag it as unreachable (#346).
			if deterministicTransportError(err) {
				return nil, &UpstreamUnreachableError{Err: err}
			}
			lastErr = err
			lastStatus = 0      // a transient transport error has no HTTP status
			lastRetryAfter = "" // ...and no Retry-After to honor — fall back to backoff
			lastCapWait = ""    // a glitch is not a cap: never stretch its retry to a cap probe
			continue            // lastStatusErr is left intact: a glitch can't shadow a real status
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			if retryableStatus(resp.StatusCode) {
				ra := resp.Header.Get("Retry-After")
				se := &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 200), RetryAfter: ra}
				// Classify a 429 LIVE against the closed rate-limit vocabulary (#1362): the
				// error we may finally surface names the kind the recovery acted on, and a
				// session/weekly/usage cap waits toward its named reset instead of hammering
				// the transient schedule. A non-429 (or a plain throttle) leaves the wait
				// decision byte-for-byte unchanged.
				cls, capWait := classifyLimit429(resp.StatusCode, raw, resp.Header, time.Now())
				se.LimitReason, se.LimitResetHint = cls.Reason, cls.ResetHint
				lastErr = se
				lastStatusErr = se
				lastStatus = resp.StatusCode
				lastRetryAfter = ra // a 429/503/529 may NAME when to retry — honor it as the next wait
				lastCapWait = capWait
				continue
			}
			// A 401 on the rotating-subscription path: re-resolve the credential fresh and
			// retry ONCE. refreshAPIKeyWait returns false (so we fall through to the raw
			// error) when there is no fresher token to try — a static/passthrough key, or
			// the same token the upstream just rejected with no re-login landing within the
			// grace window — so a truly-bad credential is not masked. The wait closes the
			// re-login race: a user logging back in mid-session has a beat to rewrite the
			// credential file, and this poll adopts the fresh token so the live session
			// self-heals in place instead of the 401 surfacing to the wrapped agent.
			if resp.StatusCode == http.StatusUnauthorized && !triedAuthRefresh && call.authRefreshable {
				if call.refreshAPIKeyWait(ctx, p) {
					triedAuthRefresh = true
					notifyAuthRefresh(p, AuthRefreshRecovered, attempt)
					lastErr = &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 200), RetryAfter: resp.Header.Get("Retry-After")}
					lastStatus = resp.StatusCode
					lastRetryAfter = "" // re-send immediately with the fresh token; do not wait
					// Do not count the credential-refresh retry against the backoff schedule:
					// rewind so the next iteration re-sends immediately with the fresh token.
					attempt--
					continue
				}
				// On the rotating path we WAITED the grace window and no fresher token landed —
				// the session is about to drop into its own /login. Surface that distinctly so
				// the otherwise-silent give-up is visible (counted once: triedAuthRefresh would
				// gate a second 401, but the window already elapsed so we fail now).
				notifyAuthRefresh(p, AuthRefreshExhausted, attempt)
			}
			return nil, &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 400), RetryAfter: resp.Header.Get("Retry-After")}
		}
		comp, err := call.adapter.ParseResponse(raw)
		if err != nil {
			return nil, fmt.Errorf("planner: %s: %w", call.adapter.Provider(), err)
		}
		comp = normalizeCompletionToolCalls(comp)
		p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
		comp.Raw = raw
		comp.PreSendQuarantines = call.quarantined
		comp.PreSendRedactions = call.redacted
		comp.PreSendRedactionRecords = call.redactions
		return comp, nil
	}
	// Prefer the last error that carried a real upstream status (and Retry-After) over a
	// later transient transport glitch, so the gateway surfaces the true 429/503/529 +
	// Retry-After rather than an opaque 502 (#1358).
	if lastStatusErr != nil {
		return nil, fmt.Errorf("planner: failed after retries: %w", lastStatusErr)
	}
	return nil, fmt.Errorf("planner: failed after retries: %w", lastErr)
}

func (p *HTTPPlanner) attachProviderCacheTelemetry(comp *Completion, reqBody []byte, provider Provider) {
	if comp == nil || comp.ProviderCache != nil {
		return
	}
	cached := comp.Usage.CachedPromptTokens()
	if cached <= 0 {
		return
	}
	endpoint, reasoning, toolSet, region := p.providerVaryAxes(reqBody)
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider:       string(provider),
		ModelID:        p.ModelID,
		CachedTokens:   int64(cached),
		PromptTokens:   int64(comp.Usage.PromptTokens),
		SerializerID:   cachemeta.DigestBytes(reqBody),
		BreakpointMode: "implicit",
		Endpoint:       endpoint,
		ReasoningMode:  reasoning,
		ToolSetID:      toolSet,
		Region:         region,
		Owner:          "agent.HTTPPlanner",
	})
	comp.ProviderCache = &entry
}

// providerVaryAxes derives the provider-prefix cache-Vary axes that silently
// break the implicit cache, so they shape the cache-family identity rather than
// blend two request shapes into one hit rate. Endpoint and reasoning mode are
// the GLM-5.2 (Z.AI) axes from GLM52-HOSTED-CACHE-COHERENCE-2026-06-19.md §A2;
// tool set and region/affinity are the remaining two from the cache-frontier
// default-enablement plan (item 7, #1525). Best-effort and additive: an axis it
// cannot determine is left empty (no identity contribution).
func (p *HTTPPlanner) providerVaryAxes(reqBody []byte) (endpoint, reasoning, toolSet, region string) {
	// Endpoint: the Z.AI Coding-Plan route carries a "coding" segment in either
	// the model id (zai-coding-plan/glm-5.2) or the base URL (.../coding/paas/...).
	if strings.Contains(p.ModelID, "coding") || strings.Contains(p.BaseURL, "/coding/") {
		endpoint = "coding"
	}
	// Reasoning mode: read reasoning_effort, then thinking.type, from the extra
	// body the operator threaded in. The implicit cache is sensitive to these.
	if len(p.ExtraBody) > 0 {
		var obj map[string]json.RawMessage
		if json.Unmarshal(p.ExtraBody, &obj) == nil {
			if raw, ok := obj["reasoning_effort"]; ok {
				var s string
				if json.Unmarshal(raw, &s) == nil && s != "" {
					reasoning = s
				}
			}
			if reasoning == "" {
				if raw, ok := obj["thinking"]; ok {
					var th struct {
						Type string `json:"type"`
					}
					if json.Unmarshal(raw, &th) == nil && th.Type != "" {
						reasoning = th.Type
					}
				}
			}
		}
	}
	toolSet = toolSetDigest(reqBody)
	region = regionFromBaseURL(p.BaseURL)
	return endpoint, reasoning, toolSet, region
}

// toolSetDigest returns a STABLE digest of the request's tool set, or "" when the
// request carries no tools. The tool definitions sit in the provider's cacheable
// PREFIX — Anthropic folds the tool schema into the cached system block, an
// OpenAI-compatible body prefixes the `tools` array ahead of the messages — so a
// silent tool-set change breaks the implicit cache. Hashing ONLY the tools (not
// the whole request body, which the per-request SerializerID already covers)
// yields the stable cache-FAMILY axis: two turns that share the same tools share
// the digest, and adding/removing/reordering a tool is recorded as a distinct
// cache-write rather than an invisible miss. Both Anthropic Messages and OpenAI
// Chat name the field `tools` at the top level.
func toolSetDigest(reqBody []byte) string {
	if len(reqBody) == 0 {
		return ""
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(reqBody, &obj) != nil {
		return ""
	}
	raw, ok := obj["tools"]
	if !ok {
		return ""
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" || string(trimmed) == "[]" {
		return ""
	}
	return cachemeta.DigestBytes(trimmed)
}

// regionFromBaseURL best-effort extracts a cloud region/affinity token from an
// endpoint host where the provider encodes it there — e.g. AWS Bedrock's
// "bedrock-runtime.us-east-1.amazonaws.com". A provider prompt cache is warm only
// in the region/zone that wrote it, so a request routed elsewhere is a distinct
// COLD family, not a hit-rate dip. It returns "" for the hosted endpoints that do
// not name a region in the host (api.anthropic.com, api.openai.com, Z.AI), so
// region stays an honest "where known" axis rather than a guess: only a
// well-formed AWS-style geo-direction-number label is recognized.
func regionFromBaseURL(baseURL string) string {
	host := baseURL
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+3:]
	}
	if i := strings.IndexAny(host, "/?#"); i >= 0 {
		host = host[:i]
	}
	for _, label := range strings.Split(host, ".") {
		if isAWSRegionToken(label) {
			return label
		}
	}
	return ""
}

// isAWSRegionToken reports whether s is a well-formed AWS-style region label of
// the shape <geo>-<direction>-<number> (e.g. "us-east-1", "ap-southeast-2"). The
// check is intentionally tight so an ordinary host label is never misread as a
// region.
func isAWSRegionToken(s string) bool {
	parts := strings.Split(s, "-")
	if len(parts) != 3 {
		return false
	}
	geo, dir, num := parts[0], parts[1], parts[2]
	if len(geo) < 2 || len(geo) > 3 || !isLowerAlpha(geo) {
		return false
	}
	if dir == "" || !isLowerAlpha(dir) {
		return false
	}
	if num == "" || !isDigits(num) {
		return false
	}
	return true
}

func isLowerAlpha(s string) bool {
	for _, r := range s {
		if r < 'a' || r > 'z' {
			return false
		}
	}
	return true
}

func isDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func (p *HTTPPlanner) transcriptAdapter() (TranscriptAdapter, error) {
	if p.Adapter != nil {
		return p.Adapter, nil
	}
	return NewTranscriptAdapter(p.Provider)
}

// UpstreamStatusError is returned by Complete when the upstream provider answered
// with a non-2xx HTTP status that was not retried away — a 4xx request error (e.g.
// a 404 for an unknown model), or a 5xx that survived every retry. It carries the
// upstream's own status code so the gateway can SURFACE it to the client: a model
// the upstream 404s must reach the caller as a non-200, not be silently swallowed
// into a misleading 200/502 (#82). Body is a short, truncated copy of the
// provider's error text for the OPERATOR LOG only — it is not meant to cross the
// trust boundary verbatim to a (possibly unauthenticated) downstream caller.
type UpstreamStatusError struct {
	Status int
	Body   string
	// RetryAfter is the upstream's Retry-After response header VERBATIM ("" when
	// absent). It is the one piece of upstream-supplied error metadata fak
	// propagates downstream — a rate-limited (429) or overloaded (503) upstream
	// names when to retry, and a wrapped agent that backs off correctly instead of
	// hammering is the whole point of surfacing it. fak NEVER parses or interprets
	// the value (RFC 7231 allows delta-seconds OR an HTTP-date); it is echoed only
	// as the downstream Retry-After header, so a malformed upstream value can never
	// reach fak's control flow. Unlike Body it is safe to forward: it carries no
	// provider error text, only timing. Empty for every non-rate-limit/overload
	// status (the header is not set on those), so it is a clean no-op there.
	RetryAfter string
	// LimitReason and LimitResetHint are sanitized provider-limit metadata for
	// HTTP 429 responses. They are operator-readable category/reset hints, not the
	// raw upstream body; the gateway may use them in downstream-safe messages.
	LimitReason    string
	LimitResetHint string
}

// Error formats the upstream's HTTP status and truncated error body as
// "planner: HTTP <status>: <body>". RetryAfter is deliberately NOT embedded — a
// downstream caller that logs err.Error() must not pick up an echoed header
// (the value is surfaced only as the response header, never the message body).
func (e *UpstreamStatusError) Error() string {
	return fmt.Sprintf("planner: HTTP %d: %s", e.Status, e.Body)
}

// UpstreamUnreachableError is returned by Complete when the upstream could not be
// reached AT ALL — a deterministic dial-time transport failure (connection
// refused, DNS NXDOMAIN, TLS handshake) that a retry cannot fix. Unlike a
// transient timeout it is returned IMMEDIATELY, skipping the 4-attempt backoff
// loop that otherwise stalls a misconfigured --base-url for ~8s (#346). The
// gateway maps it to a distinct, actionable client signal (code
// "upstream_unreachable") instead of the generic "upstream model error". Err
// carries the underlying dial cause for the OPERATOR LOG; it is not forwarded
// verbatim across the trust boundary.
type UpstreamUnreachableError struct {
	Err error
}

// Error formats the underlying dial-time cause as "planner: upstream unreachable: <err>".
func (e *UpstreamUnreachableError) Error() string {
	return fmt.Sprintf("planner: upstream unreachable: %v", e.Err)
}

// Unwrap returns the underlying dial-time transport error for errors.Is/As.
func (e *UpstreamUnreachableError) Unwrap() error { return e.Err }

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "..."
	}
	return string(b)
}

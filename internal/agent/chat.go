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
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/httptrust"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
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
	Namespace string `json:"namespace,omitempty"` // Responses namespace; empty on Chat Completions
	Arguments string `json:"arguments"`           // raw JSON string as emitted by the model
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
	// Witness binds a client-supplied tool result to an externally verifiable resource version.
	Witness string `json:"fak_witness,omitempty"`
	// RefutesWitness names an older version invalidated by this observed result or write.
	RefutesWitness string `json:"fak_refutes_witness,omitempty"`

	// Thinking carries a Claude extended-thinking ("thinking") content block
	// through the proxy instead of dropping it; ThinkingSignature is the opaque
	// signature the Anthropic API requires to round-trip the block back upstream
	// on a later turn. RedactedThinking holds any redacted_thinking blocks verbatim
	// (encrypted reasoning that must be echoed back unmodified). All three are
	// additive over the OpenAI shape; an OpenAI client simply ignores them.
	Thinking          string   `json:"thinking,omitempty"`
	ThinkingSignature string   `json:"thinking_signature,omitempty"`
	RedactedThinking  []string `json:"redacted_thinking,omitempty"`

	// ReasoningContent carries the reasoning block an OpenAI-compatible reasoning model
	// (DeepSeek V4, GLM/Qwen via vLLM --reasoning-parser qwen3, or fak's in-kernel
	// split) emits beside the final answer. It is deliberately separate from Content so
	// reasoning text is not treated as final answer text, while still round-tripping when
	// a provider requires it on a later tool-result turn.
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
		ReasoningContent  string          `json:"reasoning_content,omitempty"`
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
	m.ReasoningContent = aux.ReasoningContent
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
	Type     string          `json:"type"` // normally "function"
	Function ToolDefFunction `json:"function"`
	// ResponsesWire preserves non-function Responses tools (for example custom/freeform
	// tools) across the gateway model hop without interpreting or authorizing them.
	ResponsesWire json.RawMessage `json:"-"`
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
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	// CostUSD is an upstream-reported completion cost. It stays nil when the
	// provider omits dollars; fak never derives it from token counts here.
	CostUSD                  *float64           `json:"cost_usd,omitempty"`
	CostStatus               string             `json:"cost_status,omitempty"`
	CostProvenance           string             `json:"cost_provenance,omitempty"`
	TotalTokens              int                `json:"total_tokens"`
	PromptTokensDetails      *UsageTokenDetails `json:"prompt_tokens_details,omitempty"`
	InputTokensDetails       *UsageTokenDetails `json:"input_tokens_details,omitempty"`
	CacheReadInputTokens     int                `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int                `json:"cache_creation_input_tokens,omitempty"`
	// PromptCacheHitTokens / PromptCacheMissTokens are DeepSeek's TOP-LEVEL prompt-cache
	// counters (context caching is on by default there): hit is the prefix served from the
	// provider's KV cache, miss is the remainder it re-ingested, and prompt_tokens == hit
	// + miss. Both are OBSERVED provider-relayed counters — a DeepSeek cache hit is the
	// provider's own doing unless a separate fak-authored mechanism shaped the request.
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens,omitempty"`
	// CompletionTokensDetails carries the reasoning subcounter DeepSeek-style reasoning
	// models report; completion_tokens still INCLUDES it, so it is a breakdown, not an
	// additional axis.
	CompletionTokensDetails *UsageCompletionTokenDetails `json:"completion_tokens_details,omitempty"`
}

// UsageTokenDetails carries provider-specific prompt/input token subcounters.
type UsageTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// UsageCompletionTokenDetails carries provider-specific completion token subcounters
// (the DeepSeek/OpenAI-compatible completion_tokens_details block).
type UsageCompletionTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// CachedPromptTokens is the provider-reported prompt-cache hit count, normalized
// across OpenAI chat-completions, OpenAI Responses, Anthropic-style, and DeepSeek
// top-level counters.
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
	// DeepSeek reports its prompt-cache hit as a TOP-LEVEL field, not nested details.
	if u.PromptCacheHitTokens > 0 {
		return u.PromptCacheHitTokens
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
	// DeepSeek reports the uncached remainder DIRECTLY as prompt_cache_miss_tokens
	// (prompt_tokens == hit + miss), so the provider's own miss counter is returned
	// as-is — it already satisfies uncached + cached == the full resident prompt.
	if u.PromptCacheMissTokens > 0 {
		return u.PromptCacheMissTokens
	}
	n := u.PromptTokens
	// OpenAI/Gemini shape: prompt_tokens INCLUDES the cache-read hit (reported in
	// prompt_tokens_details/input_tokens_details), so subtract it to match Anthropic's
	// already-uncached input_tokens. The Anthropic shape carries its cache read in the
	// separate CacheReadInputTokens field and is left untouched. DeepSeek's fully-cached
	// edge (hit > 0, miss == 0) also folds its hit INTO prompt_tokens, so it peels off
	// the same way.
	if (u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0) ||
		(u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0) ||
		u.PromptCacheHitTokens > 0 {
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
	// DeepSeek folds hit + miss INTO prompt_tokens (like OpenAI), so its cache counters
	// are NOT added again — that would double-count the resident context. Only when a
	// wire omits prompt_tokens entirely do the two counters reconstruct it.
	if n == 0 && (u.PromptCacheHitTokens > 0 || u.PromptCacheMissTokens > 0) {
		n = u.PromptCacheHitTokens + u.PromptCacheMissTokens
	}
	if u.CacheReadInputTokens > 0 {
		n += u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens > 0 {
		n += u.CacheCreationInputTokens
	}
	return n
}

// ReasoningTokens is the provider-reported reasoning/thinking slice of the completion
// (DeepSeek-style completion_tokens_details.reasoning_tokens), or 0 when the wire does
// not report one. It is surfaced as a SEPARATE subcounter: CompletionTokens keeps the
// provider's own meaning (DeepSeek's completion_tokens INCLUDES reasoning), so reasoning
// is never mixed into — or silently subtracted from — final-answer token accounting.
func (u Usage) ReasoningTokens() int {
	if u.CompletionTokensDetails != nil {
		return u.CompletionTokensDetails.ReasoningTokens
	}
	return 0
}

// Completion is a planner's response for one turn.
type Completion struct {
	Message            Message
	FinishReason       string
	Usage              Usage
	ProviderCache      *cachemeta.Entry
	CacheHint          *CacheHintResult // requested/emitted/effective provider-cache receipt
	Raw                []byte           // the raw response body (transcript witness for the live seam)
	PreSendQuarantines int              // tool-result payloads held out before provider serialization
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
	ServiceTier      modelroute.ServiceMode
	// NativeInference is populated only when the caller explicitly requested a
	// receipt and this planner executed the model math in-kernel. It is captured at
	// the logits/decode seam, never reconstructed from text or gateway timing.
	NativeInference *model.NativeInferenceReceipt
	InKernelBatch   *InKernelBatchReceipt
	// DecodeTrace is populated only for an explicitly requested in-kernel decode.
	// Its events are authored at the native token-commit seam; proxy text and SSE
	// fragments are never eligible sources.
	DecodeTrace *NativeDecodeTrace
	// NativeDecodeTokenIDs is the lightweight token-ID companion to DecodeTrace.
	// It is captured at the same commit seam without inspecting logits.
	NativeDecodeTokenIDs *NativeDecodeTokenIDs
}

const (
	NativeDecodeTraceSchema      = "fak.native-decode-trace/1"
	NativeDecodeTraceEngine      = "fak-native"
	NativeDecodeTokenIDsSchema   = "fak.native-decode-token-ids/1"
	NativeDecodeTokenIDsEngine   = "fak-native"
	NativeForwardSessionStep     = "session_step"
	NativeForwardStepBatchActive = "step_batch_active"
)

// NativeDecodeTrace is the versioned, engine-authored timeline of generated-token
// commits for one buffered in-kernel completion.
type NativeDecodeTrace struct {
	Schema string                   `json:"schema"`
	Engine string                   `json:"engine"`
	Events []NativeDecodeTraceEvent `json:"events"`
}

// NativeDecodeTraceEvent records one non-stop generated token after it has been
// emitted and counted. TokenIndex is 1-based and consecutive within the attempt.
type NativeDecodeTraceEvent struct {
	TokenIndex int                  `json:"token_index"`
	ElapsedNS  int64                `json:"elapsed_ns"`
	Forward    *NativeForwardTiming `json:"forward,omitempty"`
}

// NativeForwardTiming binds the direct model-forward wall time to the committed
// token which caused it. StepBatchActive duration is one shared batch wall time;
// ActiveLanes records how many lanes shared that call.
type NativeForwardTiming struct {
	Kind        string `json:"kind"`
	DurationNS  int64  `json:"duration_ns"`
	ActiveLanes int    `json:"active_lanes"`
}

// NativeDecodeTokenIDs is a lightweight ordered receipt for tokens already
// selected by native decode. It deliberately carries no timing or logprobs;
// callers bind TokenIDs[i] to the trace event whose TokenIndex is i+1.
type NativeDecodeTokenIDs struct {
	Schema   string `json:"schema"`
	Engine   string `json:"engine"`
	TokenIDs []int  `json:"token_ids"`
}

// SampleParams are the per-request sampling overrides a CALLER may attach to one
// Complete turn. A nil pointer / nil slice means "the caller did not specify this"
// — the planner keeps its configured default, so an omitted field is byte-for-byte
// the pre-seam behavior. The pointer fields (not bare values) are what let an
// EXPLICIT temperature:0 be distinguished from an omitted one: a fixed-default
// planner like HTTPPlanner already runs temperature 0, so the two only differ when
// the caller also wants top_p/stop, and a bare float64 could not carry that intent.
type SampleParams struct {
	ServiceTier modelroute.ServiceMode
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
	// ToolChoice carries an explicit OpenAI forced-function choice into native
	// prompt rendering. Empty/auto leaves model choice unchanged.
	ToolChoice json.RawMessage
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
	// NativeInferenceReceipt requests the strict native measurement envelope. It is
	// false by default, preserving both planner work and response bytes.
	NativeInferenceReceipt bool
	// DecodeTrace requests native token-commit timestamps. It is false by default,
	// so ordinary turns allocate no trace slice and perform no per-token clock read.
	DecodeTrace bool
	// NativeDecodeTokenIDs requests the lightweight token-ID companion to a decode
	// trace. It is valid only when DecodeTrace is also requested.
	NativeDecodeTokenIDs bool
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

// NativeDecodeTracePlanner marks planners which author token traces at their own
// native decode seam. Callers use this capability before inference so a proxy or
// wrapper cannot be mistaken for a native trace source.
type NativeDecodeTracePlanner interface {
	Planner
	NativeDecodeTraceSupported() bool
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
	APIKeyFunc func() string
	// ExtraHeaders are trusted host-supplied upstream headers applied after the adapter's
	// normal auth/content headers. They are for provider account-routing metadata that is
	// not part of the generic adapter contract, e.g. the ChatGPT-Account-Id header the
	// Codex ChatGPT backend requires beside its bearer token. Copied per request so callers
	// can keep their config map immutable by convention.
	ExtraHeaders map[string]string
	// ExtraHeadersFunc supplies fresh upstream headers per request, paired with APIKeyFunc
	// for rotating subscription credentials whose routing metadata lives in the same file
	// as the token. Dynamic headers override ExtraHeaders on matching names. nil leaves the
	// static/no-extra-header path unchanged.
	ExtraHeadersFunc func() map[string]string
	// AnthropicAuthScheme declares how the Anthropic wire presents this planner's
	// credential. The zero value (AnthropicAuthAuto) sniffs the token shape, which is
	// correct for first-party api.anthropic.com and is byte-for-byte the pre-field
	// behavior. Set AnthropicAuthBearer when BaseURL points at a THIRD-PARTY
	// Anthropic-compatible endpoint whose tenant credential is not an sk-ant-* token and
	// is accepted only as a bearer — otherwise the sniff sends x-api-key and the call
	// 401s. Ignored for every non-Anthropic provider.
	AnthropicAuthScheme AnthropicAuthScheme
	// ForceResponsesStream asks a Responses upstream for SSE even when the caller used the
	// buffered Complete path. Codex's ChatGPT-subscription backend requires stream=true;
	// ordinary OpenAI API-key Responses traffic leaves this false.
	ForceResponsesStream bool
	Provider             Provider
	Adapter              TranscriptAdapter
	ExtraBody            json.RawMessage
	// CacheIntent negotiates provider-owned prompt-cache behavior. Nil preserves the legacy wire.
	CacheIntent *CacheIntent
	// OpenAIToolMessagesAsText is an opt-in compatibility mode for OpenAI-compatible
	// upstreams whose chat template accepts Qwen text tool blocks but rejects native
	// role=tool continuation messages. Default false preserves the normal OpenAI wire.
	OpenAIToolMessagesAsText bool
	Temperature              float64
	MaxTokens                int
	// MaxTokensCap clamps the outbound provider request's output-token ceiling after
	// caller/session sampling overrides. Zero leaves the request unchanged. This is for
	// OpenAI-compatible providers that reject Claude Code's large default max_tokens even
	// when the account has no token-rate quota.
	MaxTokensCap int
	// StreamProgressTimeout is the streaming CONTENT-progress deadline (#5486): how long a
	// stream may stay WARM — keepalives arriving, so the inter-byte deadline never fires —
	// without one frame that advances the turn. It is the CONFIG-SURFACE home for that knob,
	// the same shape MaxTokensCap and ForceResponsesStream take: the value is passed IN by
	// whoever builds the planner (gateway.newConfiguredHTTPPlanner threads every such knob
	// through in one place) rather than re-read from the process environment.
	//
	// Zero — every planner nobody configures — means DefaultStreamProgressTimeout. A NEGATIVE
	// value DISABLES the deadline, the one escape hatch for a provider whose prefill
	// legitimately outlasts the window. A positive value outside the [5s, 600s] band falls
	// back to the default: a window past `fak guard`'s 600s whole-request ceiling could never
	// fire, and one under the idle deadline would only mislabel a plain dead socket.
	// Resolved by streamProgressWindow.
	StreamProgressTimeout time.Duration
	Client                *http.Client
	QuarantineTranscript  bool

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

	// PendingTurnCheckpoint, when non-nil, is called at the retry boundary of Complete's backoff
	// loop (on attempt 1..N-1, BEFORE the otherwise-invisible sleep), with the 1-based attempt now
	// in progress, the last upstream status observed (the 429/5xx that triggered the retry, or 0
	// for a transient transport error), and the wall-clock instant this turn began (unix nanos).
	// It is the WRITE-AHEAD durable twin of RetryNotify (#1363, epic #1193): where RetryNotify is
	// observability that evaporates on exit, this hook records how far the in-flight turn had gotten
	// so a kill -9 mid-retry resumes at the checkpointed attempt instead of a fresh turn-0. The
	// agent loop binds it (RunArm) to a closure that writes session.Table.SetPendingTurn keyed on
	// the run's trace; chat.go stays decoupled from internal/session behind this scalar seam, exactly
	// like RetryNotify. nil = behavior byte-for-byte unchanged (no checkpoint is written).
	PendingTurnCheckpoint func(attempt int, lastStatus int, startedAtUnixNano int64)

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

	// ForbiddenRetryNotify, when non-nil, is called when a 403's bounded recovery arm resolves —
	// separately from RetryNotify and AuthRefreshNotify so a transient-permission flap is never
	// conflated with a 429/5xx backoff or a 401 token rotation (three different causes, three
	// different metrics). outcome is "recovered" when a retry within the short window returned
	// 200 (a transient abuse/capacity gate cleared and the live session healed in place instead
	// of dropping into a spurious /login), or "exhausted" when the window/attempts elapsed still
	// 403ing (the denial is the permanent entitlement kind and now surfaces with the actionable
	// answer). It is the observability hook for the otherwise-INVISIBLE transient-403 event that
	// the 2026-07-03 gem8 storm made visible. The gateway sets it to a closure that bumps a
	// per-outcome counter and prints a "fak-turn forbidden-retry" line. nil = behavior
	// byte-for-byte unchanged (the recovery arm itself is independent of the hook).
	ForbiddenRetryNotify func(outcome string, attempt int)

	// AccountFailoverFunc, when non-nil, supplies a REPLACEMENT upstream credential when the
	// current one hits an ACCOUNT-SCOPED wall — a 403 whose body says this credential's
	// organization (or region/billing) is denied, even though the credential itself is valid
	// (see classifyUpstream -> RemedyFailoverAccount; the canonical case is the org-OAuth-
	// disabled 403). No retry or re-login on THIS account can clear such a wall, so the arm
	// asks for a different account whose org still permits the request. reason is a classified
	// enum label (never the raw upstream body — the body must not cross this boundary), telling
	// the func WHY the swap is needed. It returns the new credential (a permitted sibling
	// account's live token) and ok=true when a failover target exists, or ok=false when there is
	// none (every sibling is walled/absent) — in which case the 403 surfaces terminally with the
	// actionable message, exactly as before. The guard builds this closure to enumerate sibling
	// config homes, pick one on a different, permitted, non-demoted org, and return its live
	// token; it also STICKILY redirects the per-request APIKeyFunc to the adopted account so the
	// swap persists across turns (the session heals in place, no restart). nil leaves every path
	// byte-for-byte unchanged.
	AccountFailoverFunc func(reason string) (newCred string, ok bool)

	// TransientTargetFunc supplies a distinct replacement credential after a transient
	// upstream 5xx/529 survives one quick same-target retry. Unlike AccountFailoverFunc,
	// it MUST NOT mark the current account permanently walled.
	TransientTargetFunc func(status int) (newCred string, ok bool)

	// AccountFailoverNotify, when non-nil, is called when the account-failover arm resolves —
	// separately from the other three notify hooks so an org/region/billing failover is never
	// conflated with a 429/5xx backoff, a 401 token rotation, or a transient-403 flap (four
	// distinct causes, four metrics). outcome is "recovered" when a permitted sibling credential
	// was adopted and the call re-sent in place (the walled session healed onto a working
	// account), or "exhausted" when no failover target existed and the account-scoped 403 is
	// about to surface. It is the observability hook for the otherwise-INVISIBLE account-swap
	// event. The gateway sets it to a per-outcome counter + a "fak-turn account-failover" line.
	// nil = behavior byte-for-byte unchanged (the arm itself is independent of the hook).
	AccountFailoverNotify func(outcome string, attempt int)
}

// NewHTTPPlanner builds a live planner with a bounded timeout. The per-request
// timeout defaults to 60s but is overridable via FAK_PLANNER_TIMEOUT_S — a small
// CPU-served local model (e.g. a 3B through the transformers shim) can take
// minutes per turn, so the benchmark needs a longer ceiling than a hosted API.
func NewHTTPPlanner(baseURL, model, apiKey string) *HTTPPlanner {
	// #8172: Client is built through httptrust, so a declared corporate CA bundle
	// (FAK_CA_BUNDLE) is honored on the ONE outbound model client — which covers BOTH
	// paths that reach an upstream, `fak guard`'s gateway and `fak serve --stdio`'s MCP
	// planner. With nothing declared this is byte-for-byte the historical
	// &http.Client{Timeout: plannerTimeout()}, nil Transport included.
	return &HTTPPlanner{
		BaseURL:              baseURL,
		ModelID:              model,
		APIKey:               apiKey,
		Provider:             ProviderOpenAI,
		Temperature:          0, // as deterministic as a live model allows
		MaxTokens:            1024,
		Client:               httptrust.Client(plannerTimeout()),
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
	if pv == ProviderOpenAI || pv == ProviderXAI {
		p.OpenAIToolMessagesAsText = envTruthy("FAK_OPENAI_TOOL_MESSAGES_AS_TEXT")
	}
	p.MaxTokensCap = envPositiveInt("FAK_PROVIDER_MAX_TOKENS")
	return p, nil
}

func envTruthy(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func envPositiveInt(key string) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return 0
	}
	return n
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

func (p *HTTPPlanner) effectiveExtraHeaders() map[string]string {
	var out map[string]string
	add := func(in map[string]string) {
		for k, v := range in {
			if strings.TrimSpace(k) == "" {
				continue
			}
			if out == nil {
				out = make(map[string]string, len(in))
			}
			out[k] = v
		}
	}
	add(p.ExtraHeaders)
	if p.ExtraHeadersFunc != nil {
		add(p.ExtraHeadersFunc())
	}
	return out
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

// ProbeReachability performs a bounded, zero-generation request against the exact
// configured provider endpoint. It validates the network hop and authentication
// without creating a model turn: 2xx and request-shape 4xx responses prove the
// route answered, while auth, throttling, 5xx, and transport failures do not.
func (p *HTTPPlanner) ProbeReachability(ctx context.Context) (int, error) {
	adapter, err := NewTranscriptAdapter(p.Provider)
	if err != nil {
		return 0, err
	}
	endpoint := adapter.Endpoint(p.BaseURL, p.ModelID)
	req, err := http.NewRequestWithContext(ctx, http.MethodOptions, endpoint, nil)
	if err != nil {
		return 0, err
	}
	for key, value := range adapter.Headers(p.effectiveAPIKey()) {
		if !strings.EqualFold(key, "Content-Type") {
			req.Header.Set(key, value)
		}
	}
	for key, value := range p.effectiveExtraHeaders() {
		req.Header.Set(key, value)
	}
	client := p.Client
	if client == nil {
		// A planner built by hand rather than through NewHTTPPlanner still probes over
		// the declared trust source (#8172); an unconfigured box gets the same plain
		// client as before.
		client = httptrust.Client(plannerTimeout())
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	if resp.StatusCode >= 200 && resp.StatusCode < 500 && resp.StatusCode != http.StatusUnauthorized && resp.StatusCode != http.StatusForbidden && resp.StatusCode != http.StatusTooManyRequests {
		return resp.StatusCode, nil
	}
	return resp.StatusCode, fmt.Errorf("provider endpoint returned %s", resp.Status)
}

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
	turnStart := time.Now()
	maxAttempts, deadline, budgetOn := retryBounds(turnStart)
	_, attemptsPinned := plannerMaxAttemptsPinned()
	var rs retryState // shared between-attempt truth (#1358, #1362) — see retry_state.go
	// A 401 on the pinned/rotating subscription path is recoverable ONCE: the on-disk
	// OAuth token may have rotated (or been briefly torn) between resolve and send, so we
	// re-read it fresh and retry. triedAuthRefresh caps that at a single extra attempt so a
	// genuinely-bad credential still fails fast instead of looping.
	triedAuthRefresh := false
	// A 403 gets a bounded, SEPARATE recovery arm (see retry.go): a transient abuse/capacity
	// gate clears in seconds, so retry a few times across a short window before surfacing the
	// 403 terminally. fbState tracks that arm's own attempt count + deadline, independent of the
	// 429/5xx budget so a permanent entitlement 403 can never inherit the multi-hour window.
	var fbState forbiddenRetryState
	// A 403 that names an ACCOUNT-SCOPED wall (org OAuth disabled, region, billing) is recoverable
	// ONCE by swapping to a permitted sibling account — no retry or re-login on THIS credential can
	// clear it. triedFailover caps that at a single account swap so a run of walled siblings still
	// fails fast instead of looping through the roster. failoverPending is set when a swap has
	// happened but is not yet confirmed by a 200, so the "recovered" notify reports a CONFIRMED
	// heal (fired at the next success), never an optimistic one — mirroring the forbidden arm.
	triedFailover := false
	failoverPending := false
	// A 429 that classifies as an ACCOUNT CAP (session/weekly/usage — isAccountCap429) is
	// recoverable ONCE by REHOMING the session to a permitted sibling seat: such a cap can hold
	// for the full 5h/7d reset window (and is often a multi-account/billing condition longer than
	// it looks), so waiting on the capped seat toward its named reset is expensive when a free
	// sibling seat could serve the turn now. triedRehome caps that at a single seat swap so a run
	// of capped siblings still falls through to the cap-aware backoff instead of looping the
	// roster; rehomePending defers the "rehomed" notify to the confirming 200, mirroring the 403
	// failover arm. A transient rate_limited throttle (capWait "") never triggers this — it keeps
	// its seat and rides the short backoff.
	triedRehome := false
	rehomePending := false
	triedTransientRetry := false
	triedTransientTarget := false
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Write-ahead durable checkpoint (#1363): record how far this turn has gotten
			// BEFORE the backoff sleep, so a kill -9 during the wait resumes at this attempt
			// instead of a fresh turn-0. attempt+1 is the 1-based try now in progress (the
			// PendingTurn.Attempt convention where 1 = first attempt); rs.lastStatus is the
			// 429/5xx that triggered this retry. nil hook = unchanged.
			if p.PendingTurnCheckpoint != nil {
				p.PendingTurnCheckpoint(attempt+1, rs.lastStatus, turnStart.UnixNano())
			}
		}
		// Surface the retry before the silent backoff sleep. A spent time budget stops the
		// loop, while a cancelled context returns the classified pending status (#2257).
		stop, err := p.waitBeforeAttempt(ctx, attempt, &rs, deadline, budgetOn)
		if err != nil {
			return nil, err
		}
		if stop {
			break
		}
		req, err := http.NewRequestWithContext(ctx, "POST", call.url, bytes.NewReader(call.body))
		if err != nil {
			return nil, err
		}
		call.applyHeaders(req)
		finishProvider := BeginProviderCall(req)
		if call.responsesStreamed {
			req.Header.Set("Accept", "text/event-stream")
		}
		resp, err := p.Client.Do(req)
		finishProvider(providerResponseStatus(resp), err)
		if err != nil {
			// A deterministic dial-time failure (refused / NXDOMAIN / TLS) will not
			// resolve on retry — retrying only adds ~8s of backoff latency to what is a
			// configuration error. Fail fast and tag it as unreachable (#346).
			if uerr := classifyDoError(err, &rs); uerr != nil {
				return nil, uerr
			}
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			transientRetryTried := triedTransientRetry
			transientTargetTried := triedTransientTarget
			retry, rewind, statusErr := call.handleRejectedResponse(ctx, p, &rs, resp, raw, attempt, rejectedResponseRetry{
				triedAuthRefresh: &triedAuthRefresh, forbidden: &fbState,
				triedRehome: &triedRehome, rehomePending: &rehomePending,
				triedFailover: &triedFailover, failoverPending: &failoverPending,
				triedTransientRetry: &triedTransientRetry, triedTransientTarget: &triedTransientTarget,
				recordRefreshState: true, bodyCap: 200,
			})
			if statusErr != nil {
				return nil, statusErr
			}
			if rewind && (!attemptsPinned || (p.TransientTargetFunc != nil && !transientRetryTried && triedTransientRetry && triedTransientTarget == transientTargetTried)) {
				attempt--
			}
			if retry {
				continue
			}
		}
		// A 200 after the 403 arm fired is a CONFIRMED transient-403 self-heal: the abuse/capacity
		// gate cleared and the live session healed in place instead of dropping into a spurious
		// /login. Report it once (fired is one-way, so a later success cannot double-count).
		fbState.noteRecovered(p, attempt)
		// A 200 after an account swap is a CONFIRMED failover heal: the walled session adopted a
		// permitted sibling account and completed the turn in place instead of surfacing a futile
		// /login. Report it once (failoverPending is one-way here).
		if failoverPending {
			notifyAccountFailover(p, AccountFailoverRecovered, attempt)
			failoverPending = false
		}
		// A 200 after a 429-account-cap seat rehome is a CONFIRMED rehome: the capped session
		// adopted a permitted sibling seat and completed the turn in place instead of sleeping
		// toward a hours-away reset. Report it once (rehomePending is one-way here).
		notifyRehomeRecovered(p, &rehomePending, attempt)
		if call.responsesStreamed {
			decoded, derr := openAIResponsesSSEFinalResponse(raw)
			if derr != nil {
				return nil, fmt.Errorf("planner: %s stream: %w", call.adapter.Provider(), derr)
			}
			raw = decoded
		}
		comp, err := call.adapter.ParseResponse(raw)
		if err != nil {
			return nil, fmt.Errorf("planner: %s: %w", call.adapter.Provider(), err)
		}
		comp = normalizeCompletionToolCalls(comp)
		attachProviderReportedCost(comp, raw)
		p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
		if call.cacheHint.Requested != nil {
			hint := call.cacheHint
			comp.CacheHint = &hint
		}
		comp.Raw = raw
		comp.PreSendQuarantines = call.quarantined
		comp.PreSendRedactions = call.redacted
		comp.PreSendRedactionRecords = call.redactions
		// Write-ahead CLEAR (#4123 — the symmetric other half of #1363's retry checkpoint):
		// the turn completed, so drop any in-flight checkpoint back to the zero value through
		// the SAME hook the retry boundary writes (checkpointPending maps (0,0,0) to
		// SetPendingTurn(trace, PendingTurn{})). This fires on BOTH the retry-recovered path
		// (429/5xx-then-200, clearing the {attempt,status} the loop just wrote) AND the
		// first-try 200 — because a RESUMED process re-enters a turn whose RESTORED checkpoint
		// must be cleared even though THIS process never wrote one, and Complete cannot tell a
		// fresh first-try from a resumed one, so the clear is unconditional on success. A
		// restart then reads IsZero() and starts fresh instead of re-attaching a finished turn.
		// The zero write keeps the wire byte-identical (omitzero); only Rev moves. nil hook =
		// unchanged (the pre-#1363 path).
		if p.PendingTurnCheckpoint != nil {
			p.PendingTurnCheckpoint(0, 0, 0)
		}
		return comp, nil
	}
	return nil, rs.exhausted("planner: failed after retries")
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
	// An explicitly declared Anthropic auth scheme has to reach the adapter, which is
	// the only place the credential is turned into a header. An empty scheme falls
	// through to the generic constructor, so the default path is unchanged.
	if p.Provider == ProviderAnthropic && p.AnthropicAuthScheme != AnthropicAuthAuto {
		return NewAnthropicTranscriptAdapter(p.AnthropicAuthScheme), nil
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
//
// A certificate-verification failure additionally carries the trust hint (#8172):
// this classifier already knew the chain was untrusted rather than the endpoint
// unreachable, and dropping that on the floor is what made an intercepted TLS
// connection read to every operator as a firewall. Other causes are formatted
// exactly as before.
func (e *UpstreamUnreachableError) Error() string {
	if hint := httptrust.TLSTrustHint(e.Err); hint != "" {
		return fmt.Sprintf("planner: upstream unreachable: %v — %s", e.Err, hint)
	}
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

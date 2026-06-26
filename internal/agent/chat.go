// chat.go is the planner seam: provider transcript adapters + a typed
// message/tool vocabulary + the Planner interface both the live client and the
// offline mock satisfy. See doc.go for the package's trust framing (this is the
// host-side loop, not the guarded guest) and the A/B-loop purpose.

package agent

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
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

func contentText(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		return s, nil
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
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		return s, nil
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
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw
	}
	if _, ok := obj["stream"]; !ok {
		return raw
	}
	obj["stream"] = json.RawMessage("false")
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
	// logit_bias); with none passed, the planner uses its configured defaults.
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
	Enabled         bool   // true when a reusable local KV cache is active
	Backend         string // radixkv, device backend name, or empty when unknown
	MemoryClass     string // kv_cache
	Scope           string // host/device
	DType           string // storage dtype for the local KV rows, currently f32 for HAL KV
	BytesPerToken   int64  // bytes per resident KV position under this model layout
	ResidentTokens  int    // true resident prefix positions, not the LRU edge-token budget
	ResidentBytes   int64
	BudgetTokens    int // configured LRU budget metric; 0 means unbounded or unavailable
	LRUTokens       int // Σ edge lengths, the budget metric radixkv enforces
	MaxDepthTokens  int
	Nodes           int
	Leaves          int
	Evictions       int
	PolicyEvictions int
	Splits          int
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

// ---------------------------------------------------------------------------
// Live planner — provider API client.
// ---------------------------------------------------------------------------

// HTTPPlanner drives closed-API and OpenAI-compatible chat endpoints through a
// provider transcript adapter. base_url selects the provider root; Provider
// selects the wire shape.
type HTTPPlanner struct {
	BaseURL              string
	ModelID              string
	APIKey               string
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
	d := 60 * time.Second
	if v := os.Getenv("FAK_PLANNER_TIMEOUT_S"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 5 && n <= 3600 {
			d = time.Duration(n) * time.Second
		}
	}
	return d
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
	const maxAttempts = 4
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			if err := backoff(ctx, attempt); err != nil {
				return nil, err
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", call.url, bytes.NewReader(call.body))
		if err != nil {
			return nil, err
		}
		for k, v := range call.headers() {
			if v != "" {
				req.Header.Set(k, v)
			}
		}
		resp, err := p.Client.Do(req)
		if err != nil {
			// A deterministic dial-time failure (refused / NXDOMAIN / TLS) will not
			// resolve on retry — retrying only adds ~8s of backoff latency to what is a
			// configuration error. Fail fast and tag it as unreachable (#346).
			if deterministicTransportError(err) {
				return nil, &UpstreamUnreachableError{Err: err}
			}
			lastErr = err
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			if retryableStatus(resp.StatusCode) {
				lastErr = &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 200)}
				continue
			}
			return nil, &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 400)}
		}
		comp, err := call.adapter.ParseResponse(raw)
		if err != nil {
			return nil, fmt.Errorf("planner: %s: %w", call.adapter.Provider(), err)
		}
		comp = normalizeCompletionToolCalls(comp)
		p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
		comp.Raw = raw
		comp.PreSendQuarantines = call.quarantined
		return comp, nil
	}
	return nil, fmt.Errorf("planner: failed after %d attempts: %w", maxAttempts, lastErr)
}

func (p *HTTPPlanner) attachProviderCacheTelemetry(comp *Completion, reqBody []byte, provider Provider) {
	if comp == nil || comp.ProviderCache != nil {
		return
	}
	cached := comp.Usage.CachedPromptTokens()
	if cached <= 0 {
		return
	}
	endpoint, reasoning := p.providerVaryAxes()
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider:       string(provider),
		ModelID:        p.ModelID,
		CachedTokens:   int64(cached),
		PromptTokens:   int64(comp.Usage.PromptTokens),
		SerializerID:   cachemeta.DigestBytes(reqBody),
		BreakpointMode: "implicit",
		Endpoint:       endpoint,
		ReasoningMode:  reasoning,
		Owner:          "agent.HTTPPlanner",
	})
	comp.ProviderCache = &entry
}

// providerVaryAxes derives the GLM-5.2 (Z.AI) cache-Vary axes called out in
// GLM52-HOSTED-CACHE-COHERENCE-2026-06-19.md §A2: the Coding-Plan vs general
// endpoint and the reasoning_effort/thinking mode are silent cache-breakers, so
// they must shape the provider-prefix cache identity rather than blend two
// request shapes into one hit rate. Best-effort and additive: an axis it cannot
// determine is left empty (no identity contribution).
func (p *HTTPPlanner) providerVaryAxes() (endpoint, reasoning string) {
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
	return endpoint, reasoning
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
}

// Error formats the upstream's HTTP status and truncated error body as
// "planner: HTTP <status>: <body>".
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

// deterministicTransportError reports whether a transport error from Client.Do is
// a configuration error a retry cannot fix: a refused connection (nothing
// listening on the port — the canonical "wrong port / server not started"
// misconfiguration), a DNS name that does not resolve (NXDOMAIN — a wrong host),
// or a TLS handshake failure (a wrong scheme / untrusted cert). A plain timeout
// or a reset mid-flight is NOT deterministic — it may be transient packet loss —
// so it stays on the retry path.
func deterministicTransportError(err error) bool {
	if err == nil {
		return false
	}
	// DNS name does not resolve (NXDOMAIN) — a wrong host. A *temporary* DNS
	// failure (IsNotFound false) may clear, so it stays on the retry path.
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return dnsErr.IsNotFound
	}
	// TLS handshake failures — a wrong scheme (https to a plaintext port) or an
	// untrusted certificate; neither is transient.
	var recErr tls.RecordHeaderError
	if errors.As(err, &recErr) {
		return true
	}
	var certErr *tls.CertificateVerificationError
	if errors.As(err, &certErr) {
		return true
	}
	// Connection refused is the canonical "wrong port / server not started"
	// misconfiguration. errors.Is(syscall.ECONNREFUSED) catches it on Linux/macOS;
	// on Windows the OS errno (WSAECONNREFUSED) does NOT equal the BSD constant, so
	// fall back to a dial-time, non-timeout *net.OpError — which also covers "no
	// route to host" / "network unreachable", equally deterministic. A dial that
	// TIMED OUT may be transient packet loss, so it is left to retry.
	if errors.Is(err, syscall.ECONNREFUSED) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" && !opErr.Timeout() {
		return true
	}
	return false
}

// retryableStatus reports whether an HTTP status warrants a backoff retry: 429
// (rate limited) and the 5xx overload/transient family.
func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests ||
		code == http.StatusInternalServerError ||
		code == http.StatusBadGateway ||
		code == http.StatusServiceUnavailable ||
		code == http.StatusGatewayTimeout
}

// backoff sleeps attempt^2 * 600ms (0.6s, 2.4s, 5.4s), honoring context
// cancellation, before the next retry.
func backoff(ctx context.Context, attempt int) error {
	d := time.Duration(attempt*attempt) * 600 * time.Millisecond
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "..."
	}
	return string(b)
}

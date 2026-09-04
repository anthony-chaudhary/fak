package agent

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

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

// WithToolChoice preserves an explicit OpenAI tool_choice for native prompt rendering.
func WithToolChoice(raw json.RawMessage) SampleOpt {
	return func(sp *SampleParams) {
		if len(raw) > 0 {
			sp.ToolChoice = raw
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

// WithNativeInferenceReceipt asks the in-kernel planner to capture a guarded
// per-token native execution receipt. Planners which do not own the model logits
// leave the request unsupported; callers must not reconstruct this evidence from
// response text or gateway wall time.
func WithNativeInferenceReceipt(enabled bool) SampleOpt {
	return func(sp *SampleParams) {
		sp.NativeInferenceReceipt = enabled
	}
}

// WithDecodeTrace asks a capable in-kernel planner to timestamp each committed
// generated token. Unsupported planners must be rejected by the caller before
// inference; they must not synthesize this evidence from response fragments.
func WithDecodeTrace(enabled bool) SampleOpt {
	return func(sp *SampleParams) {
		sp.DecodeTrace = enabled
	}
}

// WithNativeDecodeTokenIDs asks native decode to retain the already-selected
// token ID at the same commit seam as DecodeTrace. It performs no logits scan
// and is valid only alongside WithDecodeTrace(true).
func WithNativeDecodeTokenIDs(enabled bool) SampleOpt {
	return func(sp *SampleParams) {
		sp.NativeDecodeTokenIDs = enabled
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

// WithServiceTier requests a portable provider service mode. Standard remains omitted on wire.
func WithServiceTier(mode modelroute.ServiceMode) SampleOpt {
	return func(sp *SampleParams) { sp.ServiceTier = mode }
}

// WithReasoningEffort sets the reasoning effort tier (e.g. "none", "low", "medium", "balanced", "adaptive", "high").
func WithReasoningEffort(effort string) SampleOpt {
	return func(sp *SampleParams) {
		if effort != "" {
			sp.ReasoningEffort = effort
		}
	}
}

// WithThinkingBudget sets the explicit reasoning token budget.
func WithThinkingBudget(budget int) SampleOpt {
	return func(sp *SampleParams) {
		v := budget
		sp.ThinkingBudget = &v
	}
}

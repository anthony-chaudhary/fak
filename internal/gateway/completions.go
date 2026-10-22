package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// handleCompletions serves the LEGACY OpenAI text-completion wire (`POST
// /v1/completions`) — the pre-chat surface vLLM, SGLang, and llama.cpp-server all
// still expose, and that older clients and eval/embedding harnesses still hit. It
// adapts the request onto the SAME served completion path the chat route uses
// (session admission, budget, the in-kernel or upstream planner) by wrapping the
// request `prompt` as a single user message. There are no tools on this wire, so it
// is strictly simpler than the chat handler: no tool-call adjudication, no
// conformance fail-close. The response carries the legacy `text_completion` object
// with a bare `text` field per choice.
//
// With stream:true and no seed it takes the same EARLY-preamble split #5399 gave the chat wire
// (#5514): the 200, the SSE headers and an opening chunk go out BEFORE the decode, and a
// failure after that point rides an in-band SSE error event. Only the frame shape stays
// legacy — see completionStreamWriter. That includes a native structured-output refusal:
// chat's live stream can still answer it with a 400, this wire reports it in-band.
// Seeded streams defer the preamble like chat so unsupported seeds retain HTTP 400.
func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	if s.checkWarmupPending(w) {
		return
	}
	if dl := s.ScalarConfig().CompletionDeadlineMs; dl > 0 {
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(dl)*time.Millisecond)
		defer cancel()
		r = r.WithContext(ctx)
	}
	// Release the EP follower ranks BEFORE this rank enters the decode, exactly as the
	// chat wire does, and onto THIS wire's own route (#5523). completeServed below
	// blocks for the whole multi-rank decode, and rank-local expert parallelism makes
	// progress only if every rank runs the same forward pass — so a legacy request that
	// never started its followers leaves the front rank alone in a collective the other
	// ranks were never released into. Retain the bounded body before decoding; with
	// a roster, release waits for admission and occurs only on the boot-model path.
	//
	// Inert on a single-rank serve, which is why this went unnoticed: with
	// FAK_EP_FANOUT_ADDRS unset there are no follower URLs and this is a no-op. The
	// consequence on real multi-rank hardware — hang, timeout, or a silently degraded
	// single-rank answer — is inferred from the AllReduce contract, not measured.
	releaseEPFanout, waitEPFanout, ok := s.prepareChatEPFanout(w, r, epRouteCompletions)
	if !ok {
		return
	}
	defer waitEPFanout()
	var body completionCarrierRequest
	if !decodeRequestBody(w, r, &body) {
		return
	}
	req, carriers := body.CompletionRequest, body.carriers()
	prompt, batched := normalizePrompt(req.Prompt)
	if batched {
		writeErrCode(w, http.StatusBadRequest, "prompt_batch_unsupported", "prompt: an array of more than one prompt is a batch, which this route does not serve; send one request per prompt")
		return
	}
	if strings.TrimSpace(prompt) == "" {
		writeErr(w, http.StatusBadRequest, "prompt: field required")
		return
	}
	if rejectStopLimits(w, normalizeStop(req.Stop)) {
		return
	}
	if rejectGuidedRegexLimits(w, carriers) {
		return
	}
	if rejectInvalidStructuredOutputs(w, carriers.StructuredOutputs) {
		return
	}
	if rejectInvalidSampling(w, validateCompletionSampling(req)) {
		return
	}
	if msg := validateSamplingPenalties(body.FrequencyPenalty, body.PresencePenalty); msg != "" {
		writeErrCode(w, http.StatusBadRequest, codeSamplingParamOutOfRange, msg)
		return
	}
	if rejectInvalidResponseFormatCarrier(w, carriers.ResponseFormat) {
		return
	}
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
	if msg := validateCompletionDecodeTrace(req); msg != "" {
		writeErr(w, http.StatusBadRequest, msg)
		return
	}
	r, ok = s.prepareChatRoute(w, r, req.Model)
	if !ok {
		return
	}
	if s.refuseNativeModelMismatch(w, r, req.Model, "") {
		return
	}
	if decodeTraceRequested && !s.chatDecodeTraceSupported(req.Model) {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a fak-native model route")
		return
	}
	if !releaseEPFanout(r) {
		return
	}
	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}
	// The legacy wire has no message array; the whole request is one user prompt.
	messages := []agent.Message{{Role: agent.RoleUser, Content: prompt}}

	// Retain only the authenticated context principal, never a caller attribution
	// header. Admission can replace the context/trace on a session reset.
	outcomeOwner := principalFromContext(r.Context())
	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, messages)
	if !admitted {
		return
	}
	finishOutcome := s.beginRequestOutcome(ctx, outcomeOwner, reqTrace, w.Header())
	defer finishOutcome(nil, nil)

	// #5514 (the legacy half of #5399): completeServed below blocks for the WHOLE decode,
	// and this handler used to write its first byte only after that returned —
	// writeCompletionStream is what set the SSE headers and the 200. So a stream:true
	// legacy-completions request served by a Complete-only planner (every in-kernel serve:
	// agent.InKernelPlanner is not an agent.StreamingPlanner) emitted no status line, no
	// headers and no SSE byte for the entire multi-rank decode, exactly the stall #5399
	// removed from the chat wire. Open the stream NOW instead, through the SAME shared
	// preamble the chat route uses — only the frame shape differs (text_completion with a
	// bare `text`, never a chat-shaped delta).
	//
	// Placement is load-bearing, as on the chat route: every PRE-decode refusal — the
	// method/body/sampling 400s and writeSessionRefusal — is already behind us and kept its
	// real HTTP status. The upstream error below is the only thing left that can fail, and
	// it now has to report in-band unless a seed delays the preamble below.
	var stream *completionStreamWriter
	// Like chat, keep the status uncommitted until a seeded planner request has
	// succeeded: a route that cannot honor seed must return its typed refusal.
	if req.Stream && body.Seed == nil {
		stream = newCompletionStreamWriter(w, reqModel)
		if err := stream.open(); err != nil {
			// The client is already gone; do not spend a decode on a socket nobody reads.
			s.logf("gateway: client vanished before the streamed preamble landed: %v", err)
			return
		}
	}

	began := time.Now()
	comp, err := s.completeServed(ctx, sessionTurn, messages, nil,
		agent.WithModel(req.Model), // no-op when the client omitted model
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithSeed(body.Seed),
		agent.WithLogitBias(body.LogitBias),
		agent.WithFrequencyPenalty(body.FrequencyPenalty),
		agent.WithPresencePenalty(body.PresencePenalty),
		agent.WithChatWireTopK(body.TopK),
		agent.WithMinP(body.MinP),
		agent.WithStop(normalizeStop(req.Stop)),
		agent.WithResponseFormat(carriers.ResponseFormat),
		agent.WithGuidedDecode(carriers.GuidedDecodeFields()),
		agent.WithRawPrompt(),
		agent.WithDecodeTrace(decodeTraceRequested),
		agent.WithNativeDecodeTokenIDs(decodeTokenIDsRequested),
	)
	finishOutcome(comp, err)
	if err != nil {
		s.logf("gateway: upstream model error: %v", err)
		if stream != nil {
			// The 200 + SSE headers went out before the decode, so the status line is spent:
			// report the SAME classified failure in-band as an SSE error event + [DONE]
			// rather than truncating the stream. plannerErrorStatus carries the identical
			// metric/observation side effects writeUpstreamErr would have run, and msg is the
			// same client-facing string (never the upstream's raw body).
			status, code, msg := s.plannerErrorStatus(err)
			stream.fail(status, code, msg)
			return
		}
		s.writeUpstreamErr(w, err)
		return
	}
	if decodeTraceRequested && comp.DecodeTrace == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce a decode trace")
		return
	}
	if decodeTokenIDsRequested && comp.NativeDecodeTokenIDs == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce native decode token IDs")
		return
	}
	if req.Stream && stream == nil {
		stream = newCompletionStreamWriter(w, reqModel)
	}

	finish := comp.FinishReason
	if finish == "" {
		finish = "stop"
	}
	respModel := s.responseModel(comp.Model, reqModel, completionStreamModel(stream), "#5514")
	s.logInferenceTurn(reqTrace, "openai_completions", req.Stream, comp.Usage, finish, time.Since(began), false)

	resp := CompletionResponse{
		ID:      "cmpl-fak-" + itoa(uint64(time.Now().UnixNano())),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   respModel,
		Choices: []CompletionChoice{{Index: 0, Text: comp.Message.Content, FinishReason: &finish}},
		Usage:   comp.Usage,
	}
	if decodeTraceRequested {
		resp.Fak = &FakExt{DecodeTrace: comp.DecodeTrace}
		if decodeTokenIDsRequested {
			resp.Fak.NativeDecodeTokenIDs = comp.NativeDecodeTokenIDs
		}
	}
	if stream != nil {
		writeCompletionStream(stream, resp)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// completionCarrierRequest adds the chat route's structured-output carriers, with
// ChatRequest's tags, to the legacy body. Embedding ChatRequest instead would make
// the fields it shares with CompletionRequest ambiguous, and encoding/json drops
// ambiguous fields silently.
// Sampling controls also use the chat wire's tags and presence semantics.
type completionCarrierRequest struct {
	CompletionRequest
	Seed                  *int64          `json:"seed,omitempty"`
	LogitBias             map[int]float64 `json:"logit_bias,omitempty"`
	FrequencyPenalty      *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty       *float64        `json:"presence_penalty,omitempty"`
	TopK                  *int            `json:"top_k,omitempty"`
	MinP                  *float64        `json:"min_p,omitempty"`
	ResponseFormat        json.RawMessage `json:"response_format,omitempty"`
	GuidedJSON            json.RawMessage `json:"guided_json,omitempty"`
	GuidedRegex           json.RawMessage `json:"guided_regex,omitempty"`
	GuidedChoice          json.RawMessage `json:"guided_choice,omitempty"`
	GuidedGrammar         json.RawMessage `json:"guided_grammar,omitempty"`
	GuidedDecodingBackend json.RawMessage `json:"guided_decoding_backend,omitempty"`
	JSONSchema            json.RawMessage `json:"json_schema,omitempty"`
	Regex                 json.RawMessage `json:"regex,omitempty"`
	EBNF                  json.RawMessage `json:"ebnf,omitempty"`
	StructuredOutputs     json.RawMessage `json:"structured_outputs,omitempty"`
}

func (r completionCarrierRequest) carriers() ChatRequest {
	return ChatRequest{
		ResponseFormat:        r.ResponseFormat,
		GuidedJSON:            r.GuidedJSON,
		GuidedRegex:           r.GuidedRegex,
		GuidedChoice:          r.GuidedChoice,
		GuidedGrammar:         r.GuidedGrammar,
		GuidedDecodingBackend: r.GuidedDecodingBackend,
		JSONSchema:            r.JSONSchema,
		Regex:                 r.Regex,
		EBNF:                  r.EBNF,
		StructuredOutputs:     r.StructuredOutputs,
	}
}

func completionStreamModel(stream *completionStreamWriter) string {
	if stream == nil {
		return ""
	}
	return stream.model
}

// validateCompletionSampling enforces the same sampling-param floor the chat route
// applies (#326): a negative max_tokens, an out-of-range temperature/top_p is a
// client 400 rather than forwarded bad input. max_tokens == 0 falls through to the
// planner default (the omitempty wire field cannot distinguish explicit-0 from
// omitted), matching validateSampling.
func validateCompletionSampling(req CompletionRequest) string {
	if req.MaxTokens < 0 {
		return "max_tokens: must be a positive integer"
	}
	return validateSamplingRanges(req.Temperature, req.TopP)
}

// validateCompletionDecodeTrace applies the chat route's decode-trace floor to the
// legacy wire. The legacy wire has no native inference receipt or input trigger, so
// those fak fields are refused rather than silently ignored.
func validateCompletionDecodeTrace(req CompletionRequest) string {
	if req.Fak != nil && (req.Fak.NativeInferenceReceipt || req.Fak.InputTrigger != nil) {
		return "fak: only native_decode_token_ids is supported on /v1/completions"
	}
	if req.Fak != nil && req.Fak.NativeDecodeTokenIDs && !req.FakDecodeTrace {
		return "native decode token IDs require fak_decode_trace"
	}
	if req.FakDecodeTrace && req.Stream {
		return "fak_decode_trace requires a buffered fak-native request"
	}
	return ""
}

// normalizePrompt folds the legacy `prompt` field (a bare string, a one-element
// array of strings, or absent/null) into a single prompt string. An array of two
// or more strings is an OpenAI batch (one choice per prompt), which this route
// does not serve, so it reports batched instead of merging the prompts into one.
// Anything malformed degrades to "" so a bad prompt surfaces as the same
// empty-prompt 400 as a missing one rather than erroring the decode.
func normalizePrompt(raw json.RawMessage) (prompt string, batched bool) {
	b := []byte(raw)
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	if i >= len(b) {
		return "", false
	}
	switch b[i] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return s, false
		}
	case '[':
		var arr []string
		if err := json.Unmarshal(raw, &arr); err == nil {
			switch len(arr) {
			case 0:
				return "", false
			case 1:
				return arr[0], false
			default:
				return "", true
			}
		}
	}
	return "", false
}

// completionStreamWriter owns the client half of one stream:true LEGACY completion. It
// embeds the shared sseStreamWriter — the same preamble machinery #5399 gave the chat
// wire (stream identity, SSE headers, idempotent open-and-flush, in-band failure) — and
// adds only what is genuinely legacy-specific: a `text_completion` chunk carrying a bare
// `text` per choice. The two wires cannot share a frame builder without breaking one of
// them; they can and now do share the timing, which is what #5514 is about.
type completionStreamWriter struct {
	sseStreamWriter
}

// newCompletionStreamWriter mints the stream identity at REQUEST time, under the legacy
// `cmpl-` id namespace, because the opening chunk leaves before the turn exists and every
// later chunk has to agree with it. model is the request model (constant for the life of
// the stream; see the handler).
func newCompletionStreamWriter(w http.ResponseWriter, model string) *completionStreamWriter {
	return &completionStreamWriter{sseStreamWriter: newSSEStreamWriter(w, "cmpl-fak-", model)}
}

// chunk builds one legacy `text_completion` SSE chunk carrying this stream's identity.
// Index is 0 because the gateway normalizes every served turn to a single choice.
func (p *completionStreamWriter) chunk(text string, finish *string, usage *agent.Usage) CompletionStreamResponse {
	return CompletionStreamResponse{
		ID:      p.id,
		Object:  "text_completion",
		Created: p.created,
		Model:   p.model,
		Choices: []CompletionStreamChoice{{Index: 0, Text: text, FinishReason: finish}},
		Usage:   usage,
	}
}

// open puts the 200, the SSE headers and an EMPTY-text opening chunk on the wire before
// the decode starts. The empty chunk is the legacy wire's answer to "what can the
// preamble say before any model byte exists": it is schema-identical to every other
// chunk on this stream (text_completion, index 0, finish_reason null) and this wire
// already ends with an empty-text chunk, so a client that concatenates `text` across
// chunks is byte-unaffected — while a client watching for liveness gets a real SSE frame,
// not just headers a proxy might still be buffering. The chat wire's opening ROLE delta
// has no legacy counterpart, and inventing one would emit chat-shaped frames on a
// text-completion socket.
func (p *completionStreamWriter) open() error {
	return p.openWith(p.chunk("", nil, nil))
}

// writeCompletionStream emits the completed turn onto an ALREADY-OPENED legacy SSE
// stream: the incremental `text` fragments, then the terminal finish_reason + usage
// chunk, then [DONE]. The opening chunk left in open(), before the decode — that split is
// the whole point of #5514. segmentContent preserves every byte, so concatenating the
// text fragments reproduces the completion exactly.
func writeCompletionStream(p *completionStreamWriter, resp CompletionResponse) {
	if err := p.open(); err != nil {
		return
	}
	choice := resp.Choices[0]
	for _, seg := range segmentContent(choice.Text) {
		if err := writeSSEData(p.w, p.chunk(seg, nil, nil)); err != nil {
			return
		}
	}
	if err := writeSSEData(p.w, p.chunk("", choice.FinishReason, &resp.Usage)); err != nil {
		return
	}
	writeSSEDone(p.w, p.flusher)
}

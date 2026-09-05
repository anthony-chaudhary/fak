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
// With stream:true it takes the same EARLY-preamble split #5399 gave the chat wire
// (#5514): the 200, the SSE headers and an opening chunk go out BEFORE the decode, and a
// failure after that point rides an in-band SSE error event. Only the frame shape stays
// legacy — see completionStreamWriter.
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
	// ranks were never released into. Placement matches handleChatCompletions: after the
	// method check, before anything reads the body (the helper reads and restores it).
	//
	// Inert on a single-rank serve, which is why this went unnoticed: with
	// FAK_EP_FANOUT_ADDRS unset there are no follower URLs and this is a no-op. The
	// consequence on real multi-rank hardware — hang, timeout, or a silently degraded
	// single-rank answer — is inferred from the AllReduce contract, not measured.
	waitEPFanout, ok := s.startEPFanoutFollowers(w, r, epRouteCompletions)
	if !ok {
		return
	}
	defer waitEPFanout()
	var req CompletionRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	prompt := normalizePrompt(req.Prompt)
	if strings.TrimSpace(prompt) == "" {
		writeErr(w, http.StatusBadRequest, "prompt: field required")
		return
	}
	if rejectInvalidSampling(w, validateCompletionSampling(req)) {
		return
	}
	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}
	// The legacy wire has no message array; the whole request is one user prompt.
	messages := []agent.Message{{Role: agent.RoleUser, Content: prompt}}

	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, messages)
	if !admitted {
		return
	}

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
	// it now has to report in-band; see sseStreamWriter.fail.
	var stream *completionStreamWriter
	if req.Stream {
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
		agent.WithStop(normalizeStop(req.Stop)),
	)
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
	if stream != nil {
		writeCompletionStream(stream, resp)
		return
	}
	writeJSON(w, http.StatusOK, resp)
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

// normalizePrompt folds the legacy `prompt` field (a bare string, an array of
// strings, or absent/null) into a single prompt string. The array form is joined
// with newlines (the common multi-line-prompt convention); anything malformed
// degrades to "" so a bad prompt surfaces as the same empty-prompt 400 as a missing
// one rather than erroring the decode.
func normalizePrompt(raw json.RawMessage) string {
	b := []byte(raw)
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	if i >= len(b) {
		return ""
	}
	switch b[i] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return s
		}
	case '[':
		var arr []string
		if err := json.Unmarshal(raw, &arr); err == nil {
			return strings.Join(arr, "\n")
		}
	}
	return ""
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

package gateway

import (
	"encoding/json"
	"fmt"
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
func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
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
	ctx := r.Context()

	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}
	// The legacy wire has no message array; the whole request is one user prompt.
	messages := []agent.Message{{Role: agent.RoleUser, Content: prompt}}

	reqTrace := s.useHTTPTrace(w, r, "")
	sessionTurn, ok, canceled := s.beginServedSessionTurn(ctx, reqTrace)
	if canceled {
		return
	}
	if !ok {
		// Budget drained: opt-in human-like reset when wired, else the historical
		// refusal. Mirrors the chat + Anthropic wires.
		if newTrace, seed, reset := s.maybeResetOnBudget(ctx, sessionTurn.state, messages); reset {
			messages = spliceSeed(seed, messages)
			reqTrace = newTrace
			if sessionTurn, ok, canceled = s.beginServedSessionTurn(ctx, reqTrace); canceled {
				return
			}
			if !ok {
				writeSessionRefusal(w, sessionTurn.state)
				return
			}
		} else {
			writeSessionRefusal(w, sessionTurn.state)
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
		s.writeUpstreamErr(w, err)
		return
	}

	finish := comp.FinishReason
	if finish == "" {
		finish = "stop"
	}
	respModel := comp.Model
	if respModel == "" {
		respModel = reqModel
	}
	s.logInferenceTurn(reqTrace, "openai_completions", req.Stream, comp.Usage, finish, time.Since(began), false)

	resp := CompletionResponse{
		ID:      "cmpl-fak-" + itoa(uint64(time.Now().UnixNano())),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   respModel,
		Choices: []CompletionChoice{{Index: 0, Text: comp.Message.Content, FinishReason: &finish}},
		Usage:   comp.Usage,
	}
	if req.Stream {
		writeCompletionStream(w, resp)
		return
	}
	writeJSON(w, http.StatusOK, resp)
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

// writeCompletionStream synthesizes the legacy SSE stream for `stream: true` on
// /v1/completions, after the whole turn has completed. It mirrors
// writeChatCompletionStream but each chunk carries a `text` fragment (the legacy
// text-completion delta) instead of a chat `delta`; concatenating the text
// fragments reproduces the completion byte-for-byte. The final chunk carries the
// finish_reason + usage, followed by the `data: [DONE]` terminator.
func writeCompletionStream(w http.ResponseWriter, resp CompletionResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	choice := resp.Choices[0]
	chunk := func(text string, finish *string, usage *agent.Usage) CompletionStreamResponse {
		return CompletionStreamResponse{
			ID:      resp.ID,
			Object:  "text_completion",
			Created: resp.Created,
			Model:   resp.Model,
			Choices: []CompletionStreamChoice{{Index: choice.Index, Text: text, FinishReason: finish}},
			Usage:   usage,
		}
	}

	for _, seg := range segmentContent(choice.Text) {
		if err := writeSSEData(w, chunk(seg, nil, nil)); err != nil {
			return
		}
	}
	if err := writeSSEData(w, chunk("", choice.FinishReason, &resp.Usage)); err != nil {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

package gateway

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// streamChatLive serves POST /v1/chat/completions as a TRUE token stream: it
// forwards each upstream CONTENT fragment to the client as an OpenAI SSE chunk the
// instant the model emits it, so time-to-first-token tracks the model rather than the
// whole turn. The buffered path (writeChatCompletionStream) only synthesizes an SSE
// stream AFTER the complete turn is generated — so its first byte costs the entire
// generation; this is the half that makes fak a real low-latency server in front of a
// streaming upstream (a hosted OpenAI-compatible API, or a local vLLM/SGLang).
//
// The kernel's adjudication invariant is preserved by construction. The caller only
// enters this path when the request declares NO tools (see handleChatCompletions), so
// there is no proposed tool call to gate — the one thing that must stay buffered until
// k.Decide runs. Any tool call a model hallucinates with no tools offered is STILL
// routed through adjudicateProposed before a survivor is emitted, so nothing
// un-adjudicated ever reaches the wire. Streamed content is the model's own prose,
// which the buffered path forwards verbatim too, so streaming it live changes nothing
// about the trust posture.
//
// It returns true once it owns the response (streamed a turn, or wrote a clean HTTP
// error before any byte hit the wire); false when the configured planner cannot
// stream this wire, in which case it has written NOTHING and the caller falls back to
// the buffered+synthesized path.
func (s *Server) streamChatLive(ctx context.Context, w http.ResponseWriter, req ChatRequest, reqModel, reqTrace string, resultAdmissions []ResultAdmission) bool {
	sp, ok := s.planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return false
	}
	flusher, _ := w.(http.Flusher)
	id := "chatcmpl-fak-" + itoa(uint64(time.Now().UnixNano()))
	created := time.Now().Unix()

	chunk := func(d ChatDelta, finish *string, usage *agent.Usage) ChatStreamResponse {
		return ChatStreamResponse{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: reqModel,
			Choices: []ChatStreamChoice{{Index: 0, Delta: d, FinishReason: finish}},
			Usage:   usage,
		}
	}

	// Headers + the opening role chunk are written lazily on the first content
	// fragment, so an upstream failure BEFORE any token still lets us return a real
	// HTTP status (a 200 + SSE error is far worse for a client than a clean 502).
	var started bool
	start := func() error {
		if started {
			return nil
		}
		started = true
		h := w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		return writeSSEData(w, chunk(ChatDelta{Role: agent.RoleAssistant}, nil, nil))
	}
	sink := func(contentDelta string) error {
		if err := start(); err != nil {
			return err
		}
		return writeSSEData(w, chunk(ChatDelta{Content: contentDelta}, nil, nil))
	}

	began := time.Now()
	comp, err := sp.CompleteStream(ctx, sink, req.Messages, req.Tools,
		agent.WithModel(req.Model),
		agent.WithMaxTokens(req.MaxTokens),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithStop(normalizeStop(req.Stop)),
	)
	if err != nil {
		if !started {
			// Nothing on the wire yet — surface a real HTTP error, exactly as the
			// buffered path does, and own the response (the message is generic so the
			// upstream body never crosses the trust boundary).
			s.logf("gateway: upstream model error (stream): %v", err)
			status, code, msg := upstreamErrorStatus(err)
			writeErrCode(w, status, code, msg)
			return true
		}
		// Headers + content already went out; we cannot change the status. Emit a
		// terminal error frame + [DONE] so the client's SSE parser ends cleanly rather
		// than hanging, and log the cause for the operator.
		s.logf("gateway: upstream model error mid-stream: %v", err)
		_ = writeSSEData(w, map[string]any{
			"error": map[string]any{"message": "upstream model error", "type": "server_error"},
		})
		writeSSEDone(w, flusher)
		return true
	}

	// The turn finished. The buffered path records inference metrics inside
	// s.complete; this path bypasses it, so account here.
	s.metrics.observeInference(comp.Usage.PromptTokens, comp.Usage.CompletionTokens, comp.Usage.CachedPromptTokens(), comp.FinishReason, time.Since(began))

	// Adjudicate any proposed tool call BEFORE the client sees it — the load-bearing
	// invariant, applied even though tools were absent (a model can still hallucinate
	// a call). Only survivors are emitted.
	kept, adjs, _ := s.adjudicateProposed(ctx, comp.Message.ToolCalls, reqTrace)
	finish := comp.FinishReason
	switch {
	case len(kept) > 0:
		finish = "tool_calls"
	case finish == "" || finish == "tool_calls":
		// No surviving call — either none was proposed, or every hallucinated call
		// was dropped, or the upstream omitted a finish_reason. Any of these is a
		// normal stop to an OpenAI client.
		finish = "stop"
	}

	// Open the stream even for an empty turn (zero content, zero kept calls) so the
	// client always gets a well-formed role → finish → [DONE] sequence.
	if err := start(); err != nil {
		return true
	}
	if len(kept) > 0 {
		if err := writeSSEData(w, chunk(ChatDelta{ToolCalls: streamToolCalls(kept)}, nil, nil)); err != nil {
			return true
		}
	}
	usage := comp.Usage
	final := chunk(ChatDelta{}, &finish, &usage)
	if len(adjs) > 0 || len(resultAdmissions) > 0 {
		final.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions}
	}
	_ = writeSSEData(w, final)
	writeSSEDone(w, flusher)
	return true
}

// writeSSEDone writes the terminal `data: [DONE]` sentinel and flushes, closing an
// OpenAI-compatible SSE stream.
func writeSSEDone(w http.ResponseWriter, flusher http.Flusher) {
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

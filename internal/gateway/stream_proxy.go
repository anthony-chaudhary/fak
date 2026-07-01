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
// The kernel's adjudication invariant is preserved by construction even when tools
// ARE offered. A tool call is the one thing that must stay buffered until k.Decide
// runs, and CompleteStream HOLDS it: the native delta.tool_calls channel is
// accumulated off-wire (never streamed), and every proposed call — native or one a
// model emitted as content TEXT and LiftTextToolCalls recovered — is routed through
// adjudicateProposed before a survivor is emitted. Streamed content is the model's
// own prose, which the buffered path forwards verbatim too. The one residual hazard,
// a model burying a call in CONTENT (where a denied call's raw text could leak before
// lift strips it), is closed by liftGuard, which withholds any text-form dialect span
// from the live stream so the bytes that reach the wire are a prefix of the buffered
// post-lift content (see stream_lift_guard.go).
//
// It returns true once it owns the response (streamed a turn, or wrote a clean HTTP
// error before any byte hit the wire); false when the configured planner cannot
// stream this wire, in which case it has written NOTHING and the caller falls back to
// the buffered+synthesized path.
func (s *Server) streamChatLive(ctx context.Context, w http.ResponseWriter, req ChatRequest, reqModel, reqTrace string, sessionTurn servedSessionTurn, resultAdmissions []ResultAdmission) bool {
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
	emitContent := func(contentDelta string) error {
		if contentDelta == "" {
			return nil
		}
		if err := start(); err != nil {
			return err
		}
		return writeSSEData(w, chunk(ChatDelta{Content: contentDelta}, nil, nil))
	}
	// The sink streams prose through the lift-guard so a text-form tool-call dialect a
	// model buries in content never reaches the wire before adjudication. Whatever the
	// guard withheld is reconciled against the buffered post-lift content below.
	guard := newLiftGuard(emitContent)
	opts := []agent.SampleOpt{
		agent.WithModel(req.Model),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithStop(normalizeStop(req.Stop)),
		// Structured-output passthrough (#907): the streamed wire forwards the same
		// response_format / logit_bias constraints as the buffered path, so a streamed
		// tool-bearing turn is generated under the ride engine's constraint and still
		// adjudicated whole. No-op when absent (bit-exact drop-in).
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithLogitBias(req.LogitBias),
		agent.WithGuidedDecode(req.GuidedDecodeFields()),
	}
	lease, err := s.beginServedAdmission(ctx, sessionTurn, req.Messages, req.Tools, sampleMaxTokens(opts))
	if err != nil {
		s.logf("gateway: scheduler admission refused (stream): %v", err)
		s.writeUpstreamErr(w, err)
		return true
	}
	defer lease.Release()

	began := time.Now()
	comp, err := sp.CompleteStream(ctx, guard.write, req.Messages, req.Tools, opts...)
	if err != nil {
		if _, _, _, ok := inKernelOOMObservation(err); ok {
			s.observePlannerRequestMemory()
		}
		// Surface the failure on the default --debug-stats stderr line (the s.logf calls
		// below only reach the --log stream, OFF by default) so an errored/stalled streamed
		// turn is visible instead of a silent freeze. plannerErrorStatus below also bumps the
		// upstream-error counter.
		s.renderTurnDebugError(reqTrace, "openai_chat_completions", err, time.Since(began))
		if !started {
			// Nothing on the wire yet — surface a real HTTP error, exactly as the
			// buffered path does, and own the response (the message is generic so the
			// upstream body never crosses the trust boundary).
			s.logf("gateway: upstream model error (stream): %v", err)
			s.writeUpstreamErr(w, err)
			return true
		}
		// Headers + content already went out; we cannot change the status. Emit a
		// terminal error frame + [DONE] so the client's SSE parser ends cleanly rather
		// than hanging, and log the cause for the operator.
		_, _, _ = s.plannerErrorStatus(err)
		s.logf("gateway: upstream model error mid-stream: %v", err)
		_ = writeSSEData(w, map[string]any{
			"error": map[string]any{"message": "upstream model error", "type": "server_error"},
		})
		writeSSEDone(w, flusher)
		return true
	}

	// The turn finished. The buffered path records inference metrics inside
	// s.complete; this path bypasses it, so account here.
	lease.SettleUsage(comp.Usage) // settle the token-rate window with real usage (#2019)
	s.accountStreamedTurn(ctx, sessionTurn, comp, req.Messages, began)

	// Tool-call conformance fail-closed: the upstream announced tool_calls but none
	// survived parsing + the text-lift fallback. Proceeding would skip adjudication on
	// a call the model intended to make — the exact silent no-op the buffered path
	// refuses (handleChatCompletions). Fail closed here too: a clean 502 if nothing has
	// streamed, else a terminal error frame so the client never reads a benign empty
	// stop on a skipped call.
	if comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
		if !started {
			s.logf("gateway: upstream announced tool_calls but none parsed (stream conformance fail-closed); model=%s", s.model)
			writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
			return true
		}
		s.logf("gateway: upstream announced tool_calls but none parsed mid-stream (conformance fail-closed); model=%s", s.model)
		_ = writeSSEData(w, map[string]any{
			"error": map[string]any{"message": "upstream tool-call format not recognized", "type": "server_error"},
		})
		writeSSEDone(w, flusher)
		return true
	}

	// Adjudicate any proposed tool call BEFORE the client sees it — the load-bearing
	// invariant, applied whether or not tools were offered (a model can hallucinate a
	// call even with none offered). Only survivors are emitted.
	kept, adjs, dropped, servedText, servedHits := s.adjudicateProposedServed(ctx, comp.Message.ToolCalls, reqTrace)
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	finish := comp.FinishReason
	switch {
	case len(kept) > 0:
		finish = "tool_calls"
	case finish == "" || finish == "tool_calls":
		// No surviving call — either none was proposed, or every hallucinated call
		// was dropped, every read served inline from the vDSO, or the upstream omitted
		// a finish_reason. Any of these is a normal stop to an OpenAI client.
		finish = "stop"
	}
	s.logInferenceTurn(reqTrace, "openai_chat_completions", true, comp.Usage, finish, time.Since(began), false)

	// Open the stream even for an empty turn (zero content, zero kept calls) so the
	// client always gets a well-formed role → finish → [DONE] sequence.
	if err := start(); err != nil {
		return true
	}
	// Flush the content the guard withheld: the buffered post-lift content beyond the
	// prose already streamed. Concatenated with the live bytes this reproduces the
	// buffered path's content exactly (modulo leading whitespace lift trims).
	remaining := liftRemainder(guard.streamed(), comp.Message.Content)
	if err := emitContent(remaining); err != nil {
		return true
	}
	// vDSO served-inline (vDSO live in the hot path): a fresh cache hit is emitted as
	// assistant content; the call was dropped from kept so the client never re-runs it.
	if servedText != "" {
		if err := emitContent("\n" + servedText); err != nil {
			return true
		}
	}
	// Parity with the buffered path: when every proposed call was refused AND the turn
	// carried no content of its own, give even a fak-unaware client an actionable note
	// (which tools were denied and why) rather than an empty turn.
	if len(kept) == 0 && dropped > 0 && guard.streamed() == "" && remaining == "" {
		if err := emitContent(denySummary(adjs)); err != nil {
			return true
		}
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

package gateway

// messages_stream_planner.go bridges the generic agent.StreamingPlanner seam to
// the Anthropic Messages SSE wire. It is used when the downstream client speaks
// /v1/messages but fak is backed by an OpenAI-compatible/vLLM/SGLang upstream
// whose planner can stream content callbacks. The real Anthropic passthrough has
// its own byte-preserving relay in messages_stream_passthrough.go.

import (
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// streamAnthropicPlannerLive translates a generic planner content stream into
// Anthropic text_delta events. Natural-language content is emitted as the planner
// produces it; tool calls are held in the final Completion, adjudicated as one
// whole-turn batch, and only surviving tool_use blocks are emitted afterward.
//
// It returns false only when this request cannot use the streaming seam and the
// caller should fall back to streamAnthropicPending without anything having been
// written. Once it admits inbound results or writes a response, it owns the request.
func (s *Server) streamAnthropicPlannerLive(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn) bool {
	sp, ok := s.planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return false
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return false
	}

	floorBegan := time.Now()
	resultAdmissions, err := s.admitInboundResults(r.Context(), req.Messages, req.Tools, reqTrace)
	if err != nil {
		s.logf("gateway: result-floor error (messages stream): %v", err)
		writeErr(w, http.StatusBadGateway, "upstream model error")
		// The client now has its terminal 502, but "the client was told" and "the operator
		// was told" are different questions and only the first used to be satisfied here:
		// the s.logf above reaches the --log stream, OFF by default, so a turn refused by
		// the result floor on THIS route was invisible on /metrics and printed no FAILED
		// line. That is the under-count nobody notices, because the number is too SMALL.
		// #5525 closed the identical hole on the streaming passthrough's terminal switch;
		// this is the same cure on the planner-live arm, reusing the same two observers.
		// Unlike over there, no sentinel stands between the failure and the observation:
		// this IS the producer, so `err` is already the REAL cause and lands in its own
		// kind rather than the coarse `other` bucket. Nothing else on this path observes,
		// so the failure is counted exactly once. The response is not touched again.
		s.metrics.observeUpstreamError(err)
		s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(floorBegan))
		return true
	}

	model, id := s.anthropicTurnIdentity(req.Model)
	var send func(string, any)
	var sendMu sync.Mutex
	sendLocked := func(event string, data any) {
		sendMu.Lock()
		defer sendMu.Unlock()
		send(event, data)
	}
	started := false
	start := func() {
		if started {
			return
		}
		started = true
		h := w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("Connection", "keep-alive")
		h.Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		send = anthropicSSESender(w, flusher)
		sendLocked("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": model,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req), "output_tokens": 0},
			},
		})
	}

	outIdx := 0
	stopBuf := NewStopHoldbackBuffer(req.StopSequences)
	textOpen := false
	textIdx := -1
	closeText := func() {
		if textOpen {
			if tail := stopBuf.Flush(); tail != "" {
				sendLocked("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": textIdx,
					"delta": map[string]any{"type": "text_delta", "text": tail},
				})
			}
			sendLocked("content_block_stop", map[string]any{"type": "content_block_stop", "index": textIdx})
			textOpen = false
			textIdx = -1
		}
	}
	emitText := func(text string) error {
		safe := stopBuf.Append(text)
		if safe == "" {
			return nil
		}
		start()
		if !textOpen {
			textIdx = outIdx
			outIdx++
			textOpen = true
			sendLocked("content_block_start", map[string]any{
				"type": "content_block_start", "index": textIdx,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
		}
		sendLocked("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": textIdx,
			"delta": map[string]any{"type": "text_delta", "text": safe},
		})
		return nil
	}

	opts := s.plannerSampleOpts(req, sessionTurn)
	lease, ok := s.admitStreamedTurn(r.Context(), w, "messages stream", sessionTurn, req.Messages, req.Tools, sampleMaxTokens(opts))
	if !ok {
		return true
	}
	defer lease.Release()

	start()
	if note := s.toolFailureNoteOnce(reqTrace, req.Messages); note != "" {
		emitAnthropicTextBlock(sendLocked, &outIdx, note)
	}
	if note := resultAdmissionNote(freshAdmissionNotes(resultAdmissions)); note != "" {
		emitAnthropicTextBlock(sendLocked, &outIdx, note)
	}

	guard := newLiftGuard(emitText)
	messages := s.maybePlanMessages(r.Context(), reqTrace, req.Messages)
	messages = s.maybeElideMessages(messages) // decoded-path elision for a local model (GLM/Qwen), default-on
	began := time.Now()
	stopPing := make(chan struct{})
	pingDone := make(chan struct{})
	go func() {
		defer close(pingDone)
		ticker := time.NewTicker(anthropicStreamPingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				sendLocked("ping", map[string]any{"type": "ping"})
			case <-stopPing:
				return
			case <-r.Context().Done():
				return
			}
		}
	}()
	comp, err := sp.CompleteStream(r.Context(), guard.write, messages, req.Tools, opts...)
	close(stopPing)
	<-pingDone
	if err != nil {
		return s.streamPlannerUpstreamError(w, err, started, reqTrace, began, sendLocked, closeText)
	}
	lease.SettleUsage(comp.Usage) // settle the token-rate window with real usage (#2019)
	s.accountStreamedTurn(r.Context(), sessionTurn, comp, req.Messages, began, req.Model)

	// Tool-call conformance fail-closed (the rule itself lives in
	// failClosedOnUnparsedToolCalls). Mid-stream this surface ends the turn in the
	// Anthropic dialect: close any open text block first, then an error event.
	if s.failClosedOnUnparsedToolCalls(w, comp, started, "messages stream conformance fail-closed", "messages", func() {
		closeText()
		sendLocked("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": "upstream tool-call format not recognized"},
		})
	}) {
		return true
	}

	kept, adjs, dropped, servedText, servedHits := s.adjudicateProposedServed(r.Context(), comp.Message.ToolCalls, reqTrace)
	if remaining := liftRemainder(guard.streamed(), comp.Message.Content); remaining != "" {
		_ = emitText(remaining)
	}
	start()
	closeText()

	emitAnthropicToolUseBlocks(sendLocked, &outIdx, kept)
	if dropped > 0 || anyRepaired(adjs) || anyLivelock(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			emitAnthropicTextBlock(sendLocked, &outIdx, note)
		}
	}
	// vDSO served-inline (vDSO live in the hot path): a fresh cache hit is folded into a
	// synthetic assistant text block; the call was already dropped from kept so no
	// tool_use is emitted for it and the client never re-runs it.
	if servedText != "" {
		emitAnthropicTextBlock(sendLocked, &outIdx, servedText)
	}
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}

	usage := anthropicUsage{
		InputTokens:              comp.Usage.PromptTokens,
		OutputTokens:             comp.Usage.CompletionTokens,
		CacheReadInputTokens:     comp.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: comp.Usage.CacheCreationInputTokens,
	}
	stop := agent.AnthropicStopReason(comp.FinishReason, len(kept) > 0)
	sendAnthropicTerminal(sendLocked, stop, usage)
	return true
}

// plannerSampleOpts assembles the agent.SampleOpt set for a planner stream from the
// inbound request and the served turn's token ceiling. The dual planner (local model
// alongside the API upstream) routes on the requested model, so forward it — ONLY there:
// the single-planner paths historically never forwarded the client model on this wire.
func (s *Server) plannerSampleOpts(req *agent.AnthropicMessagesRequest, sessionTurn servedSessionTurn) []agent.SampleOpt {
	var temp *float64
	if req.Temperature != 0 {
		temp = &req.Temperature
	}
	opts := []agent.SampleOpt{
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(temp),
		agent.WithTopP(req.TopP),
		agent.WithTopK(req.TopK),
		agent.WithStop(req.StopSequences),
	}
	if _, dual := s.planner.(*DualPlanner); dual {
		opts = append(opts, agent.WithModel(req.Model))
	}
	return opts
}

// emitAnthropicToolUseBlocks streams the surviving tool_use blocks as
// content_block_start/input_json_delta/content_block_stop triples, advancing outIdx for
// each block so the client-facing index numbering stays contiguous.
func emitAnthropicToolUseBlocks(send func(string, any), outIdx *int, kept []agent.ToolCall) {
	for _, blk := range agent.AnthropicResponseBlocks(agent.Message{Role: agent.RoleAssistant, ToolCalls: kept}) {
		oi := *outIdx
		*outIdx++
		send("content_block_start", map[string]any{
			"type": "content_block_start", "index": oi,
			"content_block": map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": map[string]any{}},
		})
		send("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": oi,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": string(blk.Input)},
		})
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": oi})
	}
}

// streamPlannerUpstreamError renders the right failure for a CompleteStream error and
// returns true (the turn is fully handled). Before any byte is sent (`started` false) it
// maps the error to its distinct status/code + Retry-After so the client sees WHAT failed
// (a 429 → upstream_rate_limited, a stall → 504) rather than a generic 502; mid-stream it
// closes any open text block and emits a terminal SSE error event. The metric is observed
// here so a stalled or errored streamed turn is a counted failure, not a silent freeze.
func (s *Server) streamPlannerUpstreamError(w http.ResponseWriter, err error, started bool, reqTrace string, began time.Time, sendLocked func(string, any), closeText func()) bool {
	if _, _, _, ok := inKernelOOMObservation(err); ok {
		s.observePlannerRequestMemory()
	}
	s.metrics.observeUpstreamError(err)
	s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(began))
	if !started {
		s.surfaceUpstreamStatus(w, err, "upstream model error (messages stream)")
		return true
	}
	s.logf("gateway: upstream model error mid-stream (messages): %v", err)
	closeText()
	errType := "api_error"
	if errors.As(err, new(*agent.UpstreamStalledError)) {
		errType = "upstream_stalled"
	}
	sendLocked("error", map[string]any{
		"type":  "error",
		"error": map[string]any{"type": errType, "message": "upstream model error"},
	})
	return true
}

func sendAnthropicTerminal(send func(string, any), stop string, usage anthropicUsage) {
	finalUsage := map[string]int{
		"input_tokens":  usage.InputTokens,
		"output_tokens": usage.OutputTokens,
	}
	if usage.CacheReadInputTokens > 0 {
		finalUsage["cache_read_input_tokens"] = usage.CacheReadInputTokens
	}
	if usage.CacheCreationInputTokens > 0 {
		finalUsage["cache_creation_input_tokens"] = usage.CacheCreationInputTokens
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": finalUsage,
	})
	send("message_stop", map[string]any{"type": "message_stop"})
}

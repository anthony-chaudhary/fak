package gateway

// messages_stream_planner.go bridges the generic agent.StreamingPlanner seam to
// the Anthropic Messages SSE wire. It is used when the downstream client speaks
// /v1/messages but fak is backed by an OpenAI-compatible/vLLM/SGLang upstream
// whose planner can stream content callbacks. The real Anthropic passthrough has
// its own byte-preserving relay in messages_stream_passthrough.go.

import (
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

	resultAdmissions, err := s.admitInboundResults(r.Context(), req.Messages, reqTrace)
	if err != nil {
		s.logf("gateway: result-floor error (messages stream): %v", err)
		writeErr(w, http.StatusBadGateway, "upstream model error")
		return true
	}

	model := req.Model
	if model == "" {
		model = s.model
	}
	id := "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
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
	textOpen := false
	textIdx := -1
	closeText := func() {
		if textOpen {
			sendLocked("content_block_stop", map[string]any{"type": "content_block_stop", "index": textIdx})
			textOpen = false
			textIdx = -1
		}
	}
	emitText := func(text string) error {
		if text == "" {
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
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
		return nil
	}

	start()
	if note := resultAdmissionNote(resultAdmissions); note != "" {
		emitAnthropicTextBlock(sendLocked, &outIdx, note)
	}

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
		if _, _, _, ok := inKernelOOMObservation(err); ok {
			s.observePlannerRequestMemory()
		}
		if !started {
			s.logf("gateway: upstream model error (messages stream): %v", err)
			writeErr(w, http.StatusBadGateway, "upstream model error")
			return true
		}
		s.logf("gateway: upstream model error mid-stream (messages): %v", err)
		closeText()
		sendLocked("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": "upstream model error"},
		})
		return true
	}
	s.accountStreamedTurn(r.Context(), sessionTurn, comp, req.Messages, began)

	if comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
		if !started {
			s.logf("gateway: upstream announced tool_calls but none parsed (messages stream conformance fail-closed); model=%s", s.model)
			writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
			return true
		}
		s.logf("gateway: upstream announced tool_calls but none parsed mid-stream (messages); model=%s", s.model)
		closeText()
		sendLocked("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": "upstream tool-call format not recognized"},
		})
		return true
	}

	kept, adjs, dropped := s.adjudicateProposed(r.Context(), comp.Message.ToolCalls, reqTrace)
	if remaining := liftRemainder(guard.streamed(), comp.Message.Content); remaining != "" {
		_ = emitText(remaining)
	}
	start()
	closeText()

	for _, blk := range agent.AnthropicResponseBlocks(agent.Message{Role: agent.RoleAssistant, ToolCalls: kept}) {
		oi := outIdx
		outIdx++
		sendLocked("content_block_start", map[string]any{
			"type": "content_block_start", "index": oi,
			"content_block": map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": map[string]any{}},
		})
		sendLocked("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": oi,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": string(blk.Input)},
		})
		sendLocked("content_block_stop", map[string]any{"type": "content_block_stop", "index": oi})
	}
	if dropped > 0 || anyRepaired(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			emitAnthropicTextBlock(sendLocked, &outIdx, note)
		}
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

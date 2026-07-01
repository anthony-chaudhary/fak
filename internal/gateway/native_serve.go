package gateway

// native_serve.go — the native-harness keystone (#1316/#1837). When
// `fak serve --native` is set, /v1/messages turns are driven by fak's OWN agent loop
// (agent.RunArm / RunArmStream) instead of the single-shot proxy turn at gateway.go's
// complete(). This is the FIRST live, non-test serve-path caller of RunArm +
// WithSessionGate + WithRouteManifest + the operator steer bus — the options that, per
// the program survey, were fully built and tested but had zero live callers.
//
// The thesis (docs/notes/native-harness-progress-tracking-1315.md): on the proxy path
// the external harness (Claude Code, codex) owns the turn loop and consumes tool calls
// outside fak. The native loop is fak OWNING dispatch: fak's loop drives the turns and
// the in-kernel syscall boundary is the only tool path. This handler is that ownership,
// reachable from the wire.
//
// Scope of THIS child (honest fence): the loop is seeded with the request's last user
// message and drives the kernel-owned tool catalog (agent.ToolCatalog over
// kernel.New("localtools")) to a final answer — the AgentDojo-shaped run the program's
// definition-of-done names ("an AgentDojo run driven entirely by fak serve --native").
// Generalizing the served loop to an ARBITRARY inbound tools[] surface remains a tracked
// follow-on (#1320/#1321 wire the operator console and full session control).

import (
	"context"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// nativeMaxTurnsOr resolves the configured native loop turn cap, defaulting a
// non-positive value to DefaultNativeMaxTurns.
func nativeMaxTurnsOr(n int) int {
	if n <= 0 {
		return DefaultNativeMaxTurns
	}
	return n
}

// serveNativeMessages handles a buffered /v1/messages turn by driving fak's owned
// agent loop and rendering its final answer (plus the per-turn ArmMetrics witness) back
// on the Anthropic wire. It is the native counterpart to completeAnthropicTurn.
func (s *Server) serveNativeMessages(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string) {
	began := time.Now()
	m, err := s.runNativeArm(r.Context(), req, reqTrace)
	if err != nil {
		// An owned-loop failure is classified like any served turn error: a device OOM
		// becomes an actionable 503, a genuine model failure stays a 502 with the raw
		// provider body kept off the wire.
		s.logf("gateway: native serve loop error (trace %s): %v", reqTrace, err)
		s.writeUpstreamErr(w, err)
		return
	}

	// Render the loop's final answer as the assistant turn. The kernel already mediated
	// every tool call INSIDE the loop (vDSO-served, adjudicated, quarantined as it
	// decided), so there are no proposed-call adjudications to fold here — the owned loop
	// consumed them itself. That is exactly the "fak owns dispatch" distinction.
	asst := agent.Message{Role: agent.RoleAssistant, Content: m.FinalAnswer}
	blocks := agent.AnthropicResponseBlocks(asst)
	// A session boundary that ended the loop early (PAUSED/DRAINING/STOPPED/budget) is a
	// clean stop, not a model end-of-turn; the closed reason rides the ArmMetrics witness.
	stop := agent.AnthropicStopReason(nativeFinishReason(m), false)
	usage := anthropicUsage{InputTokens: m.PromptTokens, OutputTokens: m.CompletionTokens}

	s.logInferenceTurn(reqTrace, "anthropic_messages_native", false, agent.Usage{
		PromptTokens:     m.PromptTokens,
		CompletionTokens: m.CompletionTokens,
	}, stop, time.Since(began), false)

	arm := m // copy so the response holds a stable address, not a loop-local
	writeJSON(w, http.StatusOK, anthropicMessageResponse{
		ID:           "msg_fak_" + itoa(uint64(began.UnixNano())),
		Type:         "message",
		Role:         "assistant",
		Model:        s.modelOr(req.Model),
		Content:      blocks,
		StopReason:   stop,
		StopSequence: nil,
		Usage:        usage,
		Fak:          &FakExt{NativeArm: &arm},
	})
}

func (s *Server) serveNativeMessagesStream(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.serveNativeMessages(w, r, req, reqTrace)
		return
	}
	sp, ok := s.planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		s.serveNativeMessages(w, r, req, reqTrace)
		return
	}

	began := time.Now()
	id := "msg_fak_" + itoa(uint64(began.UnixNano()))
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	send := anthropicSSESender(w, flusher)
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": s.modelOr(req.Model),
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req), "output_tokens": 0},
		},
	})

	outIdx := 0
	textOpen := false
	textIdx := -1
	closeText := func() {
		if !textOpen {
			return
		}
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": textIdx})
		textOpen = false
		textIdx = -1
	}
	emitText := func(text string) error {
		if text == "" {
			return nil
		}
		if !textOpen {
			textIdx = outIdx
			outIdx++
			textOpen = true
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": textIdx,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
		}
		send("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": textIdx,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
		return nil
	}

	m, err := s.runNativeArmStream(r.Context(), req, reqTrace, emitText)
	if err != nil {
		s.logf("gateway: native stream loop error (trace %s): %v", reqTrace, err)
		closeText()
		send("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": "upstream model error"},
		})
		return
	}
	closeText()

	stop := agent.AnthropicStopReason(nativeFinishReason(m), false)
	usage := anthropicUsage{InputTokens: m.PromptTokens, OutputTokens: m.CompletionTokens}
	s.logInferenceTurn(reqTrace, "anthropic_messages_native", false, agent.Usage{
		PromptTokens:     m.PromptTokens,
		CompletionTokens: m.CompletionTokens,
	}, stop, time.Since(began), false)

	arm := m
	sendAnthropicTerminalWithNativeArm(send, stop, usage, &arm)
}

// runNativeArm drives agent.RunArm(fak=true) for one served request, wiring the
// already-built-but-uncalled loop options to live serve-path sources:
//
//   - WithSessionGate: the SAME injected DecideSession/DebitSession hooks the proxy
//     request boundary uses (serveSessions in cmd/fak), so the owned loop gates each turn
//     boundary on the live drive state — run-state, budget, pace — and reports usage back.
//     Wiring the trace also arms drainSteer, so an operator POST .../steer is folded into
//     the next turn of THIS loop (the consumer half that had no live caller).
//   - WithRouteManifest: the live, hot-reloadable routing policy (s.route), so a per-call
//     model route is bound before each in-loop k.Syscall, exactly as the proxy path does.
//
// The loop is seeded with the request's last user message; the kernel-owned tool catalog
// is the sole tool path. It returns the per-turn ArmMetrics — the witness that the loop,
// not an external harness, drove the turn.
func (s *Server) runNativeArm(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace string) (agent.ArmMetrics, error) {
	task := lastUserText(req.Messages)
	return agent.RunArm(ctx, s.planner, task, true, s.nativeMaxTurns, nil, s.nativeRunOptions(ctx, reqTrace)...)
}

func (s *Server) runNativeArmStream(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace string, sink agent.StreamSink) (agent.ArmMetrics, error) {
	task := lastUserText(req.Messages)
	return agent.RunArmStream(ctx, s.planner, task, true, s.nativeMaxTurns, sink, nil, s.nativeRunOptions(ctx, reqTrace)...)
}

func (s *Server) nativeRunOptions(ctx context.Context, reqTrace string) []agent.RunOption {
	opts := make([]agent.RunOption, 0, 2)
	if s.decideSession != nil {
		opts = append(opts, agent.WithSessionGate(agent.SessionGate{
			Decide: func(trace string) (int, bool, int, string) {
				v := s.decideSession(ctx, trace)
				return v.MaxTokens, v.Proceed, v.MinGapMs, v.Reason
			},
			Debit: func(trace string, out, cx int) {
				if s.debitSession == nil {
					return
				}
				s.debitSession(ctx, trace, SessionUsage{CompletionTokens: out, ContextTokens: cx})
			},
		}, reqTrace))
	}
	if s.route != nil {
		if mfst := s.route.Manifest(); mfst != nil {
			opts = append(opts, agent.WithRouteManifest(mfst))
		}
	}
	return opts
}

func sendAnthropicTerminalWithNativeArm(send func(string, any), stop string, usage anthropicUsage, arm *agent.ArmMetrics) {
	finalUsage := map[string]int{
		"input_tokens":  usage.InputTokens,
		"output_tokens": usage.OutputTokens,
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": finalUsage,
	})
	send("message_stop", map[string]any{
		"type": "message_stop",
		"fak":  &FakExt{NativeArm: arm},
	})
}

// nativeFinishReason maps the owned loop's outcome to a planner-style finish reason for
// the Anthropic stop-reason projection: a session-boundary stop is reported as "stop"
// (a clean end_turn carrying the reason on the ArmMetrics witness), as is a normal final
// answer; only a turn-cap hit reports "length".
func nativeFinishReason(m agent.ArmMetrics) string {
	if m.HitTurnCap {
		return "length"
	}
	return "stop"
}

// lastUserText returns the content of the last user-role message in the canonical
// transcript — the task the owned loop is seeded with. DecodeAnthropicMessagesRequest
// has already flattened each inbound user content block into a single text Content, so a
// plain scan from the end is sufficient. Returns "" when there is no user message (the
// loop then runs from its system prompt alone).
func lastUserText(messages []agent.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleUser {
			return messages[i].Content
		}
	}
	return ""
}

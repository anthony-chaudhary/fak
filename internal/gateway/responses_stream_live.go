package gateway

// responses_stream_live.go gives POST /v1/responses a LIVE passthrough: when the
// downstream client sent stream:true and the planner can stream the wire live, the
// Responses SSE events are emitted incrementally as upstream content fragments
// arrive, instead of waiting for the whole buffered turn. The synthesized path
// (writeResponsesStream in responses.go) stays untouched as the fallback and is the
// schema authority: this file emits the SAME event names, reuses the SAME JSON
// envelope structs (responsesResponse / responsesOutputItem / responsesContentPart)
// and the SAME sequence-number discipline (running count from 0, terminal
// response.completed, no [DONE]).
//
// The kernel's adjudication invariant is preserved by construction even when tools
// ARE offered, exactly as on the chat wire: the liftGuard withholds any text-form
// tool-call dialect from the live stream (stream_lift_guard.go), CompleteStream
// HOLDS the structured tool-call deltas off-wire, adjudicateProposedServed runs the
// capability floor on the accumulated set, and only surviving calls are emitted —
// after the message item closes. Streamed content is the model's own prose, which
// the buffered path forwards verbatim too.
//
// This mirrors streamChatLive (stream_proxy.go) and streamAnthropicPlannerLive
// (messages_stream_planner.go) step for step; only the dialect is Responses.

import (
	"context"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// responsesLiveGate is the optional planner capability that answers whether the
// planner's provider wire can deliver a LIVE token stream for a Responses turn
// (W1's agent.ResponsesWireStreamsLive). The in-kernel planner streams live under
// its own sink and does not implement the gate; a planner that does implement it
// must resolve its provider for the fail-closed check below.
type responsesLiveGate interface {
	ResponsesWireStreamsLive(provider agent.Provider, streamGate bool) bool
}

// plannerWireProvider types the planner's provider out without reaching into
// agent.WireProfileFor: HTTPPlanner carries a public Provider field; anything else
// may expose an optional Provider() method. ok=false means the provider is
// unknowable — the gate check is then skipped and the planner is treated as
// live-capable (the InKernelPlanner case).
func plannerWireProvider(p agent.Planner) (agent.Provider, bool) {
	if hp, isHTTP := p.(*agent.HTTPPlanner); isHTTP {
		return hp.Provider, true
	}
	type providerHolder interface{ Provider() agent.Provider }
	if ph, hasMethod := p.(providerHolder); hasMethod {
		return ph.Provider(), true
	}
	return "", false
}

// responsesLiveTurn is the stabilized model-input bundle for one live Responses
// stream: the SAME messages slice the buffered completeServed call would receive
// (post admission, canonicalization, elision, compaction), the canonicalized tool
// defs, and the sampling contract the buffered turn samples under. Bundled so the
// function signature stays within the small-parameter discipline.
type responsesLiveTurn struct {
	messages   []agent.Message
	tools      []agent.ToolDef
	sampleOpts []agent.SampleOpt
}

// streamResponsesLive streams POST /v1/responses LIVE over the planner's
// StreamingPlanner seam. It returns false only when the request cannot take the
// live path (planner cannot stream under the W1 gate, no flusher) and NOTHING has
// been written — the caller falls through to the buffered path, which synthesizes
// the same events. Once it admits the turn or writes anything, it owns the request
// and returns true. persistResponse is the buffered path's continuation-store
// closure, called with the SAME id response.created advertised.
func (s *Server) streamResponsesLive(ctx context.Context, w http.ResponseWriter, reqModel, reqTrace string, sessionTurn servedSessionTurn, resultAdmissions []ResultAdmission, turn responsesLiveTurn, restoreContinuation *responsesRestoreContinuation, persistResponse func(id string, asst agent.Message)) bool {
	sp, ok := s.planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return false
	}
	// W1's Responses-wire capability gate: a wire that cannot deliver a live token
	// stream for a Responses turn falls back to the buffered synthesis. Fail-closed:
	// a gate-carrying planner whose provider cannot be resolved does not go live.
	if gate, hasGate := s.planner.(responsesLiveGate); hasGate {
		provider, provOK := plannerWireProvider(s.planner)
		if !provOK || !gate.ResponsesWireStreamsLive(provider, s.isForceResponsesStream()) {
			return false
		}
	}
	if _, isFlusher := w.(http.Flusher); !isFlusher {
		return false
	}

	// #5399 anti-stall discipline, shared with the chat and messages wires: nothing
	// has been written yet, so an admission refusal still surfaces as a real HTTP
	// error (admitStreamedTurn writes it); ok=false means the request is fully
	// handled either way.
	lease, ok := s.admitStreamedTurn(ctx, w, "responses stream", sessionTurn, turn.messages, turn.tools, sampleMaxTokens(turn.sampleOpts))
	if !ok {
		return true
	}
	defer lease.Release()
	began := time.Now()

	// Headers + response.created are written lazily on the first content fragment, so
	// an upstream failure BEFORE any token still returns a real HTTP error exactly as
	// the buffered path does.
	var started bool
	createdID := newResponsesID()
	createdAt := time.Now().Unix()

	// nextSeq is the event sequence-number discipline: every emitted event carries
	// the running count starting at 0 (writeResponsesStream parity). The handler is
	// single-goroutine, so plain mutation is safe.
	nextSeq := 0
	type responseEvent struct {
		Type           string            `json:"type"`
		SequenceNumber int               `json:"sequence_number"`
		Response       responsesResponse `json:"response"`
	}
	type outputItemEvent struct {
		Type           string              `json:"type"`
		SequenceNumber int                 `json:"sequence_number"`
		OutputIndex    int                 `json:"output_index"`
		Item           responsesOutputItem `json:"item"`
	}
	type outputTextDeltaEvent struct {
		Type           string `json:"type"`
		SequenceNumber int    `json:"sequence_number"`
		OutputIndex    int    `json:"output_index"`
		ContentIndex   int    `json:"content_index"`
		Delta          string `json:"delta"`
	}
	type outputTextDoneEvent struct {
		Type           string `json:"type"`
		SequenceNumber int    `json:"sequence_number"`
		OutputIndex    int    `json:"output_index"`
		ContentIndex   int    `json:"content_index"`
		Text           string `json:"text"`
	}
	type functionCallArgsDeltaEvent struct {
		Type           string `json:"type"`
		SequenceNumber int    `json:"sequence_number"`
		OutputIndex    int    `json:"output_index"`
		CallID         string `json:"call_id"`
		Delta          string `json:"delta"`
	}
	type functionCallArgsDoneEvent struct {
		Type           string `json:"type"`
		SequenceNumber int    `json:"sequence_number"`
		OutputIndex    int    `json:"output_index"`
		CallID         string `json:"call_id"`
		Arguments      string `json:"arguments"`
	}

	start := func() {
		if started {
			return
		}
		started = true
		h := w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		// response.created: the same metadata the buffered path emits (in_progress,
		// empty output, zero usage), with sequence_number 0.
		_ = writeSSEEvent(w, "response.created", responseEvent{
			Type:           "response.created",
			SequenceNumber: nextSeq,
			Response: responsesResponse{
				ID:        createdID,
				Object:    "response",
				CreatedAt: createdAt,
				Model:     reqModel,
				Status:    "in_progress",
				Output:    []responsesOutputItem{},
				Usage:     responsesUsage{},
			},
		})
		nextSeq++
	}

	// emitResponseFailed is the mid-stream terminal for an upstream error or a
	// conformance refusal: response.failed is NOT valid for a success tail, so it is
	// only written when the turn did not complete (the buffered error paths own
	// nothing at that point — the SSE status line is already spent).
	emitResponseFailed := func() {
		_ = writeSSEEvent(w, "response.failed", responseEvent{
			Type:           "response.failed",
			SequenceNumber: nextSeq,
			Response: responsesResponse{
				ID:        createdID,
				Object:    "response",
				CreatedAt: createdAt,
				Model:     reqModel,
				Status:    "failed",
				Output:    []responsesOutputItem{},
				Usage:     responsesUsage{},
			},
		})
		nextSeq++
	}

	// The message item is opened ONCE on the first content fragment (output_index 0)
	// and closed with the accumulated text before any kept tool-call item.
	messageItem := responsesOutputItem{
		Type:   "message",
		ID:     "msg_fak_live_0",
		Status: "in_progress",
		Role:   agent.RoleAssistant,
		Content: []responsesContentPart{
			{Type: "output_text", Text: ""},
		},
	}
	itemOpened := false
	fullText := ""

	emitDelta := func(delta string) {
		_ = writeSSEEvent(w, "response.output_text.delta", outputTextDeltaEvent{
			Type:           "response.output_text.delta",
			SequenceNumber: nextSeq,
			OutputIndex:    0,
			ContentIndex:   0,
			Delta:          delta,
		})
		nextSeq++
	}
	emitContent := func(delta string) error {
		if delta == "" {
			return nil
		}
		start()
		if !itemOpened {
			itemOpened = true
			_ = writeSSEEvent(w, "response.output_item.added", outputItemEvent{
				Type:           "response.output_item.added",
				SequenceNumber: nextSeq,
				OutputIndex:    0,
				Item:           messageItem,
			})
			nextSeq++
		}
		emitDelta(delta)
		return nil
	}

	// The sink streams prose through the lift-guard so a text-form tool-call dialect a
	// model buries in content never reaches the wire before adjudication, then
	// through the UTF-8 fragment joiner so a final incomplete rune cannot emit as its
	// own fragment — the exact chain streamChatLive uses.
	guard := newLiftGuard(emitContent)
	fragments := newUTF8FragmentBuffer(guard.write)

	comp, err := sp.CompleteStream(ctx, fragments.write, turn.messages, turn.tools, turn.sampleOpts...)
	if err != nil {
		s.renderTurnDebugError(reqTrace, "openai_responses", err, time.Since(began))
		if !started {
			// Nothing on the wire yet — surface a real HTTP error exactly as the
			// buffered path does, and own the request. writeUpstreamErr folds the
			// classification AND observes the upstream-error metric exactly once.
			s.logf("gateway: upstream model error (responses stream): %v", err)
			s.writeUpstreamErr(w, err)
			return true
		}
		// Headers already went out; the status is spent. Emit a terminal
		// response.failed so the client's SSE parser ends cleanly rather than reading
		// a silent truncated stream, and count the failure.
		s.logf("gateway: upstream model error mid-stream (responses): %v", err)
		s.metrics.observeUpstreamError(err)
		emitResponseFailed()
		return true
	}

	// The turn finished. The buffered path folds inference metrics + the admission
	// accounting into s.complete; this path bypasses it, so account here.
	lease.SettleUsage(comp.Usage) // settle the token-rate window with real usage (#2019)
	s.accountStreamedTurn(ctx, sessionTurn, comp, turn.messages, began, reqModel)

	// Tool-call conformance fail-closed (the buffered counterpart is
	// handleResponses): the upstream announced tool_calls but none survived parsing
	// — proceeding would skip adjudication on a call the model intended to make.
	if comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
		if started {
			s.logf("gateway: upstream announced tool_calls but none parsed mid-stream (responses stream); model=%s", s.model)
			emitResponseFailed()
			return true
		}
		s.logf("gateway: upstream announced tool_calls but none parsed (responses stream); model=%s", s.model)
		writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
		return true
	}

	// A final incomplete rune has no later fragment to complete it. Flush it now so
	// guard.streamed() below is final before the remainder is reconciled.
	if err := fragments.flush(); err != nil {
		return true
	}

	// Adjudicate the proposed tool calls BEFORE the client sees them — the
	// load-bearing invariant, applied whether or not tools were offered (a model can
	// hallucinate a call even with none). Only survivors are emitted.
	kept, adjs, dropped, servedText, servedHits := s.adjudicateProposedServed(ctx, comp.Message.ToolCalls, reqTrace)
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	// #3567 output-side shadow: classify the MODEL's own outbound prose (sampled,
	// observe-only) before fak blanks/appends anything — mirror of the buffered path.
	outputNegframeAudit.observe(comp.Message.Content)

	// Reconcile what the guard withheld: the buffered (post-lift) content beyond the
	// prose already streamed, then the served-inline and adjudication notes, each as
	// final content deltas BEFORE output_text.done closes the item.
	streamed := guard.streamed()
	remaining := liftRemainder(streamed, comp.Message.Content)
	if remaining != "" {
		emitContent(remaining)
		fullText = streamed + remaining
	} else {
		fullText = streamed
	}
	if servedText != "" {
		// vDSO served-inline: the call was dropped from kept, so the client never
		// re-runs it; its answer is folded into assistant text (chat-path parity).
		emitContent("\n" + servedText)
		fullText += "\n" + servedText
	}
	emittedAdjudicationNote := false
	if anyLivelock(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			emitContent(note)
			fullText += note
			emittedAdjudicationNote = true
		}
	}
	if !emittedAdjudicationNote && len(kept) == 0 && dropped > 0 && streamed == "" && remaining == "" && servedText == "" {
		// Buffered-path parity: every proposed call was refused AND the turn carried
		// no content of its own — give even a fak-unaware client an actionable note
		// (which tools were denied and why) rather than an empty turn.
		if note := denySummary(adjs); note != "" {
			emitContent(note)
			fullText += note
		}
	}

	if itemOpened {
		start()
		messageItem.Status = "completed"
		messageItem.Content = []responsesContentPart{{Type: "output_text", Text: fullText}}
		_ = writeSSEEvent(w, "response.output_text.done", outputTextDoneEvent{
			Type:           "response.output_text.done",
			SequenceNumber: nextSeq,
			OutputIndex:    0,
			ContentIndex:   0,
			Text:           fullText,
		})
		nextSeq++
		_ = writeSSEEvent(w, "response.output_item.done", outputItemEvent{
			Type:           "response.output_item.done",
			SequenceNumber: nextSeq,
			OutputIndex:    0,
			Item:           messageItem,
		})
		nextSeq++
	}

	// Kept tool calls: one function_call output item each (added → arguments
	// delta/done → done), mirroring writeResponsesStream's per-item emission, at the
	// output indexes right after the message item.
	fnBase := 0
	if itemOpened {
		fnBase = 1
	}
	for i, tc := range kept {
		idx := fnBase + i
		args := tc.Function.Arguments
		if args == "" {
			args = "{}"
		}
		callItem := responsesOutputItem{
			Type:      "function_call",
			ID:        tc.ID,
			Status:    "completed",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Namespace: tc.Function.Namespace,
			Arguments: args,
		}
		_ = writeSSEEvent(w, "response.output_item.added", outputItemEvent{
			Type:           "response.output_item.added",
			SequenceNumber: nextSeq,
			OutputIndex:    idx,
			Item:           callItem,
		})
		nextSeq++
		_ = writeSSEEvent(w, "response.function_call_arguments.delta", functionCallArgsDeltaEvent{
			Type:           "response.function_call_arguments.delta",
			SequenceNumber: nextSeq,
			OutputIndex:    idx,
			CallID:         tc.ID,
			Delta:          args,
		})
		nextSeq++
		_ = writeSSEEvent(w, "response.function_call_arguments.done", functionCallArgsDoneEvent{
			Type:           "response.function_call_arguments.done",
			SequenceNumber: nextSeq,
			OutputIndex:    idx,
			CallID:         tc.ID,
			Arguments:      args,
		})
		nextSeq++
		_ = writeSSEEvent(w, "response.output_item.done", outputItemEvent{
			Type:           "response.output_item.done",
			SequenceNumber: nextSeq,
			OutputIndex:    idx,
			Item:           callItem,
		})
		nextSeq++
	}

	// Buffered-tail parity: surviving calls make this a tool_calls turn; every call
	// refused/served-inline is a plain stop. The RESPONSE status below still maps the
	// planner's own finish reason (responsesStatusFor), exactly as the buffered tail.
	finish := comp.FinishReason
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 || servedHits > 0 {
		finish = "stop"
	}

	respModel := comp.Model
	if respModel == "" {
		respModel = reqModel
	}
	// Open the stream even for an empty turn so the client always gets a
	// well-formed created → completed sequence.
	start()
	s.logInferenceTurn(reqTrace, "openai_responses", true, comp.Usage, finish, time.Since(began), false)

	resp := responsesResponse{
		ID:        createdID,
		Object:    "response",
		CreatedAt: createdAt,
		Model:     respModel,
		Status:    responsesStatusFor(comp.FinishReason),
		Output:    []responsesOutputItem{},
		Usage:     responsesUsageFrom(comp.Usage),
	}
	if comp.FinishReason == "length" {
		resp.IncompleteDetails = &responsesIncomplete{Reason: "max_output_tokens"}
	}
	if itemOpened {
		resp.Output = append(resp.Output, messageItem)
	}
	for _, tc := range kept {
		args := tc.Function.Arguments
		if args == "" {
			args = "{}"
		}
		resp.Output = append(resp.Output, responsesOutputItem{
			Type:      "function_call",
			ID:        tc.ID,
			Status:    "completed",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Namespace: tc.Function.Namespace,
			Arguments: args,
		})
	}
	resp.OutputText = fullText
	turnAdjs := turnAdjudications(restoreContinuation.adjudications, adjs)
	for i := range turnAdjs {
		AttachOperatorRemedyMetadata(&turnAdjs[i])
	}
	SetOperatorRemedyHeaders(w, turnAdjs)
	if len(turnAdjs) > 0 || len(resultAdmissions) > 0 {
		resp.Fak = &FakExt{Adjudications: turnAdjs, ResultAdmissions: resultAdmissions}
	}
	// Publish the finalized continuation state before response.completed reveals the
	// ID (the buffered path persists before writeResponsesStream for the same reason).
	persistResponse(createdID, agent.Message{Role: agent.RoleAssistant, Content: fullText, ToolCalls: kept})
	_ = writeSSEEvent(w, "response.completed", responseEvent{
		Type:           "response.completed",
		SequenceNumber: nextSeq,
		Response:       resp,
	})
	nextSeq++
	return true
}

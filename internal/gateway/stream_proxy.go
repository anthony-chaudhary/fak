package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// incidentConfig holds the configuration for mid-stream incident packets.
type incidentConfig struct {
	enabled bool
	dir     string
}

// newIncidentConfig creates an incident config from environment.
// FAK_STREAM_INCIDENT_DIR sets the output directory.
func newIncidentConfig() *incidentConfig {
	if dir := os.Getenv("FAK_STREAM_INCIDENT_DIR"); dir != "" {
		return &incidentConfig{enabled: true, dir: dir}
	}
	return &incidentConfig{}
}

// checkpointConfig holds the configuration for partial-output checkpoints.
type checkpointConfig struct {
	enabled bool
	dir     string
}

// newCheckpointConfig creates a checkpoint config from environment.
// FAK_STREAM_CHECKPOINT_DIR sets the output directory.
func newCheckpointConfig() *checkpointConfig {
	if dir := os.Getenv("FAK_STREAM_CHECKPOINT_DIR"); dir != "" {
		return &checkpointConfig{enabled: true, dir: dir}
	}
	return &checkpointConfig{}
}

// writeIncident writes a mid-stream incident packet to the incident directory.
// It captures the correlated slice: phase, counts, durations, upstream status class, bound policy.
func (ic *incidentConfig) writeIncident(wire, phase, statusClass, boundPolicy, cause string, ttft, elapsed, bytesEmitted, eventsEmitted, lastEventAge int64) {
	if !ic.enabled {
		return
	}
	_ = os.MkdirAll(ic.dir, 0o750)
	path := filepath.Join(ic.dir, "incidents.jsonl")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o640)
	if err != nil {
		return
	}
	defer f.Close()
	pkt := map[string]any{
		"wire":                  wire,
		"phase":                 phase,
		"first_token_ms":        ttft,
		"elapsed_ms":            elapsed,
		"bytes_emitted":         bytesEmitted,
		"events_emitted":        eventsEmitted,
		"last_event_age_ms":     lastEventAge,
		"upstream_status_class": statusClass,
		"bound_policy":          boundPolicy,
		"cause":                 cause,
	}
	data, _ := json.Marshal(pkt)
	f.WriteString(string(data) + "\n")
}

// writeCheckpoint writes a durable partial-output checkpoint for mid-stream death.
func (c *checkpointConfig) writeCheckpoint(wire, traceID, model, phase string, elapsedMS int64, text string, estimatedTokens int, boundPolicy, reason string) {
	if !c.enabled {
		return
	}
	ck := map[string]any{
		"schema":           "fak-stream-checkpoint/1",
		"wire":             wire,
		"trace_id":         traceID,
		"model":            model,
		"phase":            phase,
		"elapsed_ms":       elapsedMS,
		"text":             text,
		"estimated_tokens": estimatedTokens,
		"bound_policy":     boundPolicy,
		"reason":           reason,
	}
	data, err := json.Marshal(ck)
	if err != nil {
		return
	}
	filename := fmt.Sprintf("checkpoint-%s-%d.json", traceID, time.Now().UnixNano())
	path := filepath.Join(c.dir, filename)
	_ = os.WriteFile(path, data, 0644)
}

// estimateTokens provides a rough token estimate for the checkpoint (chars/4 proxy).
func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

// heartbeatConfig holds the configuration for stream progress heartbeats (#10672).
type heartbeatConfig struct {
	enabled       bool
	interval      time.Duration
	started       bool
	mu            sync.Mutex
	writeMu       sync.Locker
	streamStart   time.Time
	lastEvent     time.Time
	bytesEmitted  int64
	eventsEmitted int64
}

func (h *heartbeatConfig) setWriteLocker(l sync.Locker) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.writeMu = l
}

// newHeartbeatConfig creates a heartbeat config from environment.
// FAK_STREAM_HEARTBEAT_S sets the interval in seconds (clamped to [1, 60]).
// Zero or unset means heartbeats are disabled.
func newHeartbeatConfig() *heartbeatConfig {
	hb := &heartbeatConfig{}
	if v := os.Getenv("FAK_STREAM_HEARTBEAT_S"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= 60 {
			hb.enabled = true
			hb.interval = time.Duration(n) * time.Second
		}
	}
	return hb
}

// markStreamStart marks the moment the first token was emitted (stream committed).
func (h *heartbeatConfig) markStreamStart() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.started {
		h.started = true
		h.streamStart = time.Now()
		h.lastEvent = h.streamStart
	}
}

// recordEvent records that a content event was emitted.
func (h *heartbeatConfig) recordEvent(byteCount int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.started {
		h.bytesEmitted += int64(byteCount)
		h.eventsEmitted++
		h.lastEvent = time.Now()
	}
}

// emitHeartbeat writes a heartbeat comment frame if heartbeats are enabled and stream has started.
func (h *heartbeatConfig) emitHeartbeat(w http.ResponseWriter) bool {
	h.mu.Lock()
	if !h.enabled || !h.started {
		h.mu.Unlock()
		return false
	}
	now := time.Now()
	elapsed := now.Sub(h.streamStart)
	lastEventAge := now.Sub(h.lastEvent)
	bytesEmitted := h.bytesEmitted
	eventsEmitted := h.eventsEmitted
	h.mu.Unlock()

	hb := map[string]any{
		"elapsed_ms":        elapsed.Milliseconds(),
		"phase":             "mid_stream",
		"bytes_emitted":     bytesEmitted,
		"events_emitted":    eventsEmitted,
		"last_event_age_ms": lastEventAge.Milliseconds(),
	}
	data, _ := json.Marshal(hb)
	h.mu.Lock()
	writeMu := h.writeMu
	h.mu.Unlock()
	if writeMu != nil {
		writeMu.Lock()
		defer writeMu.Unlock()
	}
	_, _ = fmt.Fprintf(w, ": fak-heartbeat %s\n\n", data)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	return true
}

// streamChatLive serves POST /v1/chat/completions as a TRUE token stream: it
// forwards each upstream CONTENT fragment to the client as an OpenAI SSE chunk the
// instant the model emits it, so time-to-first-token tracks the model rather than the
// whole turn. The buffered path (chatStreamWriter) opens the stream up front but can
// only synthesize the CONTENT chunks after the complete turn is generated — so its
// first TOKEN still costs the entire generation, even though its first BYTE no longer
// does (#5399); this is the half that makes fak a real low-latency server in front of
// a streaming upstream (a hosted OpenAI-compatible API, or a local vLLM/SGLang).
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
// Typed progress heartbeats (#10672): when FAK_STREAM_HEARTBEAT_S is set, the gateway
// emits periodic SSE comment frames (`: fak-heartbeat {json}`) during the streaming
// window, carrying only counts and durations (elapsed, phase, bytes/events emitted,
// last-event age). Heartbeats are SILENT before the first token (the response is not
// committed yet) and contain NO prompt or output content.
func (s *Server) streamChatLive(ctx context.Context, w http.ResponseWriter, req ChatRequest, reqModel, reqTrace string, sessionTurn servedSessionTurn, resultAdmissions []ResultAdmission, inputTriggerRoute *InputTriggerRouteReceipt) bool {
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

	// Heartbeat config for typed progress heartbeats (#10672).
	var streamMu sync.Mutex
	hb := newHeartbeatConfig()
	hb.setWriteLocker(&streamMu)
	if hbTicker, hbDone := startHeartbeat(ctx, hb, w); hbTicker != nil {
		defer func() {
			close(hbDone)
			hbTicker.Stop()
		}()
	}

	// Headers + the opening role chunk are written lazily on the first content
	// fragment, so an upstream failure BEFORE any token still lets us return a real
	// HTTP status (a 200 + SSE error is far worse for a client than a clean 502).
	var started bool
	start := func() error {
		streamMu.Lock()
		defer streamMu.Unlock()
		if started {
			return nil
		}
		started = true
		hb.markStreamStart()
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
		hb.recordEvent(len(contentDelta))
		streamMu.Lock()
		defer streamMu.Unlock()
		return writeSSEData(w, chunk(ChatDelta{Content: contentDelta}, nil, nil))
	}
	// The sink streams prose through the lift-guard so a text-form tool-call dialect a
	// model buries in content never reaches the wire before adjudication. Whatever the
	// guard withheld is reconciled against the buffered post-lift content below.
	guard := newLiftGuard(emitContent)
	utf8Fragments := newUTF8FragmentBuffer(guard.write)
	opts := buildStreamSampleOpts(req, sessionTurn)
	lease, ok := s.admitStreamedTurn(ctx, w, "stream", sessionTurn, req.Messages, req.Tools, sampleMaxTokens(opts))
	if !ok {
		return true
	}
	defer lease.Release()

	began := time.Now()
	comp, err := sp.CompleteStream(ctx, utf8Fragments.write, req.Messages, req.Tools, opts...)
	if err != nil {
		s.handleStreamError(w, flusher, hb, err, started, reqTrace, began)
		return true
	}

	// The turn finished. The buffered path records inference metrics inside
	// s.complete; this path bypasses it, so account here.
	lease.SettleUsage(comp.Usage) // settle the token-rate window with real usage (#2019)
	s.accountStreamedTurn(ctx, sessionTurn, comp, req.Messages, began, reqModel)

	// Tool-call conformance fail-closed (the rule itself lives in
	// failClosedOnUnparsedToolCalls; the buffered counterpart is handleChatCompletions).
	// Mid-stream this surface ends the turn in the OpenAI dialect: a data frame carrying
	// a server_error, then [DONE].
	if s.failClosedOnUnparsedToolCalls(w, comp, started, "stream conformance fail-closed", "conformance fail-closed", func() {
		streamMu.Lock()
		defer streamMu.Unlock()
		_ = writeSSEData(w, map[string]any{
			"error": map[string]any{"message": "upstream tool-call format not recognized", "type": "server_error"},
		})
		writeSSEDone(w, flusher)
	}) {
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

	// A final incomplete rune has no later fragment to complete it. Feed it through
	// the existing JSON fallback before reconciling the buffered completion.
	if err := utf8Fragments.flush(); err != nil {
		return true
	}

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
	emittedAdjudicationNote := false
	if anyLivelock(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			if err := emitContent(note); err != nil {
				return true
			}
			emittedAdjudicationNote = true
		}
	}
	// Parity with the buffered path: when every proposed call was refused AND the turn
	// carried no content of its own, give even a fak-unaware client an actionable note
	// (which tools were denied and why) rather than an empty turn.
	if !emittedAdjudicationNote && len(kept) == 0 && dropped > 0 && guard.streamed() == "" && remaining == "" {
		if err := emitContent(denySummary(adjs)); err != nil {
			return true
		}
	}
	streamMu.Lock()
	if len(kept) > 0 {
		if err := writeSSEData(w, chunk(ChatDelta{ToolCalls: streamToolCalls(kept)}, nil, nil)); err != nil {
			streamMu.Unlock()
			return true
		}
	}
	usage := comp.Usage
	final := chunk(ChatDelta{}, &finish, &usage)
	if len(adjs) > 0 || len(resultAdmissions) > 0 || inputTriggerRoute != nil {
		final.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions, InputTriggerRoute: inputTriggerRoute}
	}
	_ = writeSSEData(w, final)
	writeSSEDone(w, flusher)
	streamMu.Unlock()
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

func startHeartbeat(ctx context.Context, hb *heartbeatConfig, w http.ResponseWriter) (*time.Ticker, chan struct{}) {
	if !hb.enabled {
		return nil, nil
	}
	hbTicker := time.NewTicker(hb.interval)
	hbDone := make(chan struct{})
	go func() {
		for {
			select {
			case <-hbTicker.C:
				hb.emitHeartbeat(w)
			case <-hbDone:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	return hbTicker, hbDone
}

func handleMidStreamError(w http.ResponseWriter, flusher http.Flusher, hb *heartbeatConfig, err error, msg string, began time.Time) {
	if hb != nil {
		hb.mu.Lock()
		writeMu := hb.writeMu
		hb.mu.Unlock()
		if writeMu != nil {
			writeMu.Lock()
			defer writeMu.Unlock()
		}
	}
	ic := newIncidentConfig()
	hb.mu.Lock()
	ttft := int64(0)
	if !hb.streamStart.IsZero() {
		ttft = hb.streamStart.Sub(began).Milliseconds()
	}
	elapsed := time.Since(began).Milliseconds()
	bytesEmitted := hb.bytesEmitted
	eventsEmitted := hb.eventsEmitted
	lastEventAge := time.Since(hb.lastEvent).Milliseconds()
	hb.mu.Unlock()

	statusClass := "error"
	boundPolicy := "max-duration=off"
	if v := os.Getenv("FAK_STREAM_MAX_DURATION_S"); v != "" {
		boundPolicy = "max-duration=" + v + "s"
	}
	var cause string
	var ue *agent.UpstreamStalledError
	if errors.As(err, &ue) {
		statusClass = "stall"
		if ue.Kind == "max-duration" {
			statusClass = "bound:max-stream-duration"
		}
		cause = ue.Kind
	} else {
		cause = "upstream_error"
	}
	ic.writeIncident("openai_chat_completions", "mid_stream", statusClass, boundPolicy, cause, ttft, elapsed, bytesEmitted, eventsEmitted, lastEventAge)

	_ = writeSSEData(w, map[string]any{
		"error": map[string]any{"message": msg, "type": "server_error"},
	})
	writeSSEDone(w, flusher)
}

func buildStreamSampleOpts(req ChatRequest, sessionTurn servedSessionTurn) []agent.SampleOpt {
	return []agent.SampleOpt{
		agent.WithModel(req.Model),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithStop(normalizeStop(req.Stop)),
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithToolChoice(req.ToolChoice),
		agent.WithLogitBias(req.LogitBias),
		agent.WithGuidedDecode(req.GuidedDecodeFields()),
	}
}

func (s *Server) handleStreamError(w http.ResponseWriter, flusher http.Flusher, hb *heartbeatConfig, err error, started bool, reqTrace string, began time.Time) {
	if _, _, _, ok := inKernelOOMObservation(err); ok {
		s.observePlannerRequestMemory()
	}
	s.renderTurnDebugError(reqTrace, "openai_chat_completions", err, time.Since(began))
	if !started {
		s.logf("gateway: upstream model error (stream): %v", err)
		s.writeUpstreamErr(w, err)
		return
	}
	_, _, msg := s.plannerErrorStatus(err)
	s.logf("gateway: upstream model error mid-stream: %v", err)
	handleMidStreamError(w, flusher, hb, err, msg, began)
}

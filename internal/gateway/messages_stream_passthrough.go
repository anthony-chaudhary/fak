package gateway

// messages_stream_passthrough.go is the TRUE-streaming half of the Anthropic
// passthrough — the flagship `fak guard -- claude` latency win. The buffered path
// (streamAnthropicPending) asks the real Anthropic API for the WHOLE turn, waits for
// it, then synthesizes the SSE locally, so the client's first token costs the entire
// generation (TTFT == full turn). This path instead forwards the inbound bytes with
// stream:true and relays the upstream Anthropic SSE as it arrives, so the first token
// tracks the model — and the prompt-cache hit's fast prefill is finally FELT instead
// of being buffered away.
//
// The kernel boundary is preserved by construction. Text and thinking deltas are the
// model's own prose/reasoning — the buffered path forwards them verbatim too — so
// streaming them live changes nothing about the trust posture. Every tool_use block,
// the one thing the kernel must gate, is HELD: its input is accumulated off-wire, the
// whole batch runs through adjudicateProposed exactly as the buffered path does, and
// only survivors are emitted (with repaired arguments where the kernel rewrote them).
// A denied call is dropped and named in an in-band note, never shown to the client.

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// errPassthroughResponded is returned from the event callback when the handler has
// already written a terminal HTTP response (e.g. a result-floor error before any SSE
// byte) and the stream read must stop WITHOUT the caller falling back to a second
// upstream request.
var errPassthroughResponded = errors.New("gateway: anthropic passthrough response already written")

// errUpstreamInBandRefusal is returned from the event callback when the upstream refused the
// turn IN-BAND before message_start with an `error` frame this build cannot classify into a
// status. It is deliberately NOT an *agent.UpstreamStatusError: nothing definite is known
// about the refusal, so the caller's pre-start arm treats it like any other unclassified
// non-start and falls back to the buffered path instead of surfacing an invented status.
var errUpstreamInBandRefusal = errors.New("gateway: upstream refused the stream in-band before message_start")

// sseToolAccum buffers one upstream tool_use content block while its input streams in,
// so the full call can be reconstructed and adjudicated before the client sees it.
type sseToolAccum struct {
	id, name string
	args     strings.Builder
}

// anthropicPassthrough holds the per-request state of the live Anthropic /v1/messages
// passthrough stream. It relays upstream SSE events to the client unchanged EXCEPT that
// tool_use blocks are HELD off-wire, adjudicated as one batch at message_delta, and only
// the survivors are emitted (renumbered to contiguous client-facing indices). The methods
// below are the relay's moving parts; streamAnthropicPassthroughLive wires onEvent into
// StreamAnthropicRaw and interprets the outcome.
type anthropicPassthrough struct {
	s        *Server
	w        http.ResponseWriter
	r        *http.Request
	req      *agent.AnthropicMessagesRequest
	reqTrace string
	turn     servedSessionTurn
	flusher  http.Flusher

	send       func(string, any)
	started    bool
	wroteError bool
	// wroteErrCause is the REAL failure behind the terminal HTTP error wroteError marks.
	// onEvent can only hand the caller the errPassthroughResponded sentinel — "a response
	// was already written" is a control-flow fact, not a failure kind — so the cause is
	// carried here for the terminal switch to count and print. Nil means the producer had
	// no upstream failure to report, and the switch then stays silent rather than filing a
	// control-flow value as an upstream failure on /metrics.
	wroteErrCause error

	outIdx    int         // next contiguous client-facing content-block index
	passIdx   map[int]int // upstream index -> client index (relayed blocks)
	toolBuf   map[int]*sseToolAccum
	toolOrder []int // upstream indices of held tool_use blocks, in arrival order

	admitted     bool
	resultAdms   []ResultAdmission
	flushedTools bool
	keptTools    int
	servedTools  int // vDSO served-inline hits this turn (a SUCCESS, excluded from deny-all)
	adjs         []ToolAdjudication

	promptTok, complTok, cacheRead, cacheCreate int
	finishReason                                string

	// firstTokenAt is the wall-clock of the FIRST content delta from the upstream —
	// the prefill→decode boundary (time-to-first-token). Zero until the first delta
	// arrives; streamAnthropicPassthroughLive turns it into the prefill/decode split
	// it reports to observeInferenceTimed. A turn that never produces a delta (an
	// immediate stop) leaves it zero, so prefill is reported as "not measured" rather
	// than as the full turn.
	firstTokenAt time.Time

	// --- warm-continue (replay-as-context) state, #3353 -------------------------
	// asstText accumulates the assistant TEXT already relayed to the client this turn,
	// off-wire, so a mid-stream worker death can replay it as a prefill turn on a fresh
	// worker (dynamo RetryManager.track_response) instead of ending the caller's SSE with
	// a terminal error. See messages_stream_warmcontinue.go for the resume mechanism.
	asstText strings.Builder
	// continuing is set while re-issuing after a death: onEvent then suppresses the
	// continuation's own message_start echo/notes so the client sees ONE unbroken turn.
	continuing bool
	// sawThinking keeps the replay honest: a thinking turn cannot be prefilled (Anthropic
	// rejects an assistant prefill with extended thinking enabled), so warm-continue is not
	// attempted once a thinking block has been relayed.
	sawThinking bool
	// openClientBlock is the client index of the currently open relayed content block, or
	// -1 when none is open. A warm-continue closes a dangling block before the continuation
	// opens a fresh one, so the resumed stream stays well-formed.
	openClientBlock int

	// --- heartbeat & checkpoint (#10672) ---------------------------------------
	// hb holds the heartbeat config for typed progress heartbeats.
	hb *heartbeatConfig
	// hbTicker and hbDone manage the heartbeat goroutine lifecycle.
	hbTicker *time.Ticker
	hbDone   chan struct{}
	writeMu  sync.Mutex
	// checkpointDir is the directory for durable partial-output checkpoints.
	checkpointDir string
	// incidentDir is the directory for incident packets.
	incidentDir string
	// traceID is the trace ID for this turn (for checkpoint/incident correlation).
	traceID string
	// model is the model ID for this turn (for checkpoint).
	model string
	// began is the turn start time (for elapsed_ms in checkpoint/incident).
	began time.Time
}

// reqModel is the model id the client asked for on this relayed turn, or "" when the
// request is unavailable — the routing input servedLocality classifies from.
func (p *anthropicPassthrough) reqModel() string {
	if p == nil || p.req == nil {
		return ""
	}
	return p.req.Model
}

// markFirstToken stamps the time-to-first-token boundary on the first content delta of
// the turn. Idempotent: only the first delta sets it, so later deltas do not move it.
func (p *anthropicPassthrough) markFirstToken(now time.Time) {
	if p.firstTokenAt.IsZero() {
		p.firstTokenAt = now
	}
}

// start opens the client SSE stream exactly once: it writes the event-stream headers and
// the 200 status, then installs the SSE sender. Idempotent so onEvent can call it freely.
func (p *anthropicPassthrough) start() {
	if p.started {
		return
	}
	p.started = true
	p.hb = newHeartbeatConfig()
	p.hb.setWriteLocker(&p.writeMu)
	if p.hb.enabled {
		p.hbTicker = time.NewTicker(p.hb.interval)
		p.hbDone = make(chan struct{})
		go func() {
			for {
				select {
				case <-p.hbTicker.C:
					p.hb.emitHeartbeat(p.w)
				case <-p.hbDone:
					return
				case <-p.r.Context().Done():
					return
				}
			}
		}()
	}
	h := p.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	p.w.WriteHeader(http.StatusOK)
	rawSend := anthropicSSESender(p.w, p.flusher)
	p.send = func(event string, data any) {
		p.writeMu.Lock()
		defer p.writeMu.Unlock()
		rawSend(event, data)
	}
	p.hb.markStreamStart()
}

// flushHeldTools adjudicates every buffered tool_use block as one batch (exact parity with
// the buffered path's single adjudicateProposed call), emits the survivors as tool_use
// blocks with our contiguous indices, and appends an in-band note when the kernel dropped
// or repaired a call. Runs once, just before the terminal message_delta.
func (p *anthropicPassthrough) flushHeldTools() {
	if len(p.toolOrder) == 0 {
		return
	}
	calls := make([]agent.ToolCall, 0, len(p.toolOrder))
	for _, ui := range p.toolOrder {
		ta := p.toolBuf[ui]
		args := strings.TrimSpace(ta.args.String())
		if args == "" {
			args = "{}"
		}
		calls = append(calls, agent.ToolCall{
			ID: ta.id, Type: "function",
			Function: agent.Func{Name: ta.name, Arguments: args},
		})
	}
	kept, adjs, dropped, servedText, servedHits := p.s.adjudicateProposedServed(p.r.Context(), calls, p.reqTrace)
	p.keptTools = len(kept)
	p.servedTools = servedHits
	p.adjs = adjs
	// Stash this turn's SAFETY delta (blocked/repaired calls + quarantined inbound results) so the
	// per-turn fak-turn debug line, rendered just after the terminal inference observation, shows
	// what the kernel refused the MOMENT it happened — not only in the exit summary. p.resultAdms
	// was set at message_start; together they are the whole turn's adjudication outcome.
	p.s.recordTurnSafety(p.reqTrace, adjs, p.resultAdms)
	// Render survivors through the SAME helper the buffered path uses, so the tool_use
	// blocks (id preserved, input as a normalized object) are byte-shaped identically —
	// only the framing differs.
	for _, blk := range agent.AnthropicResponseBlocks(agent.Message{Role: agent.RoleAssistant, ToolCalls: kept}) {
		oi := p.outIdx
		p.outIdx++
		p.send("content_block_start", map[string]any{
			"type": "content_block_start", "index": oi,
			"content_block": map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": map[string]any{}},
		})
		p.send("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": oi,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": string(blk.Input)},
		})
		p.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": oi})
	}
	if dropped > 0 || anyRepaired(adjs) || anyLivelock(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			emitAnthropicTextBlock(p.send, &p.outIdx, note)
		}
	}
	// vDSO served-inline (vDSO live in the hot path): fold a fresh cache hit into a
	// synthetic assistant text block; the call was dropped from kept so no tool_use is
	// emitted and the client never re-runs it. relayMessageDelta already rewrites a
	// fully-served turn (hadTools, keptTools==0) tool_use -> end_turn.
	if servedText != "" {
		emitAnthropicTextBlock(p.send, &p.outIdx, servedText)
	}
	if servedHits > 0 {
		p.s.metrics.recordServedInline(servedHits)
	}
}

// relayBlockEvent forwards one upstream content-block event under the CLIENT-side index
// this relay assigned to that upstream block, and reports that index. The renumbering is
// the whole point: held tool_use blocks never reach the client, so upstream indices are
// not contiguous downstream and an event replayed with its upstream index would address
// the wrong block. An upstream index with no client index is a held block — nothing is
// relayed and ok is false.
func (p *anthropicPassthrough) relayBlockEvent(event string, data json.RawMessage, upstreamIdx int) (clientIdx int, ok bool) {
	oi, ok := p.passIdx[upstreamIdx]
	if !ok {
		return 0, false
	}
	relayWithIndex(p.send, event, data, oi)
	return oi, true
}

// onEvent is the per-SSE-event relay callback handed to StreamAnthropicRaw. It opens the
// client stream on message_start (after arming the inbound result floor once), holds and
// renumbers content blocks, batches held tool_use blocks at message_delta, and forwards
// terminal frames. A returned error stops the upstream read; errPassthroughResponded marks
// that a terminal HTTP response was already written.
func (p *anthropicPassthrough) onEvent(ev agent.AnthropicSSEEvent) error {
	switch ev.Event {
	case "message_start":
		// A warm-continue re-issue (#3353) opens its OWN upstream stream with its own
		// message_start; the client is already mid-turn, so swallow the duplicate frame
		// (and the start-of-turn notes) and keep the single unbroken client turn.
		if p.continuing {
			p.start() // idempotent no-op; the client stream is already open
			return nil
		}
		// First event from the real API. Arm the result-side floor ONCE (so a
		// tainted inbound result still refuses a later exfil call, exactly as the
		// buffered path does), then open the client stream. Running admit here —
		// only after the upstream opened — means an open failure falls back to the
		// buffered path WITHOUT a double-admit on the same trace.
		if !p.admitted {
			p.admitted = true
			adms, aerr := p.s.admitInboundResults(p.r.Context(), p.req.Messages, p.req.Tools, p.reqTrace)
			if aerr != nil {
				p.s.logf("gateway: result-floor error (messages stream): %v", aerr)
				writeErr(p.w, http.StatusBadGateway, "upstream model error")
				// Carry the cause out with the flag: the sentinel below is all the caller
				// would otherwise have, and it says nothing about WHY the turn failed.
				p.wroteError, p.wroteErrCause = true, aerr
				return errPassthroughResponded
			}
			p.resultAdms = adms
		}
		p.start()
		p.promptTok, p.cacheRead, p.cacheCreate = anthropicStartUsage(ev.Data)
		p.send("message_start", ev.Data)
		// The model is about to read kernel-originated diagnostics for inbound tool
		// results before its prose: executor failures first, then quarantine stubs.
		if note := p.s.toolFailureNoteOnce(p.reqTrace, p.req.Messages); note != "" {
			emitAnthropicTextBlock(p.send, &p.outIdx, note)
		}
		if note := resultAdmissionNote(freshAdmissionNotes(p.resultAdms)); note != "" {
			emitAnthropicTextBlock(p.send, &p.outIdx, note)
		}
		// Unusually-expensive-session advisory (once/session, gate-armed only).
		if note := p.s.ctxExpenseNoteOnce(p.reqTrace); note != "" {
			emitAnthropicTextBlock(p.send, &p.outIdx, note)
		}

	case "content_block_start":
		var d struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
		}
		if json.Unmarshal(ev.Data, &d) != nil {
			return nil
		}
		if d.ContentBlock.Type == "tool_use" {
			p.toolBuf[d.Index] = &sseToolAccum{id: d.ContentBlock.ID, name: d.ContentBlock.Name}
			p.toolOrder = append(p.toolOrder, d.Index)
			return nil // HELD until adjudicated
		}
		if d.ContentBlock.Type == "thinking" || d.ContentBlock.Type == "redacted_thinking" {
			// A thinking turn cannot be replayed as an assistant prefill, so once one is
			// relayed warm-continue (#3353) stands down for this turn (canWarmContinue).
			p.sawThinking = true
		}
		oi := p.outIdx
		p.outIdx++
		p.passIdx[d.Index] = oi
		p.openClientBlock = oi // a warm-continue closes this if it is still open on death
		relayWithIndex(p.send, "content_block_start", ev.Data, oi)

	case "content_block_delta":
		var d struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal(ev.Data, &d) != nil {
			return nil
		}
		// First content delta of the turn = the model's first produced token, whether
		// it lands in a relayed text block or a held tool_use block. Stamp the TTFT
		// boundary here so prefill (prompt ingest) is separated from decode below.
		p.markFirstToken(time.Now())
		if ta, held := p.toolBuf[d.Index]; held {
			ta.args.WriteString(d.Delta.PartialJSON) // accumulate off-wire
			return nil
		}
		if _, ok := p.relayBlockEvent("content_block_delta", ev.Data, d.Index); ok {
			// Accumulate relayed assistant TEXT off-wire so a mid-stream worker death can
			// replay exactly what the client already saw as a prefill turn (#3353). Only
			// text_delta is captured — thinking deltas are not prefill-replayable.
			if d.Delta.Type == "text_delta" {
				p.asstText.WriteString(d.Delta.Text)
				// Record heartbeat event for typed progress heartbeats (#10672).
				if p.hb != nil {
					p.hb.recordEvent(len(d.Delta.Text))
				}
			}
		}

	case "content_block_stop":
		var d struct {
			Index int `json:"index"`
		}
		if json.Unmarshal(ev.Data, &d) != nil {
			return nil
		}
		if _, held := p.toolBuf[d.Index]; held {
			return nil // emitted (or dropped) as a batch at message_delta
		}
		if oi, ok := p.relayBlockEvent("content_block_stop", ev.Data, d.Index); ok {
			if p.openClientBlock == oi {
				p.openClientBlock = -1 // block closed cleanly; nothing dangling to close on death
			}
		}

	case "message_delta":
		if !p.flushedTools {
			p.flushedTools = true
			p.flushHeldTools()
		}
		// Fold this turn's adjudication SHAPE into separate turn-control signals BEFORE relaying
		// the (possibly end_turn-rewritten) message_delta. Hard deny-all remains the bounded
		// stop-policy path; retryable tool feedback continues without counting as a session stop.
		// On a deny-all turn the fingerprint (same tool+reason) is the same-issue signal the guard
		// Stop hook keys its give-up on.
		signal := adjudicationOutcomeForTurn(p.adjs, p.keptTools, p.servedTools)
		denyFP := ""
		if signal == adjudicationOutcomeDenyAll {
			denyFP = denyAllFingerprint(p.adjs)
		}
		p.s.recordAdjudicationOutcome(signal, denyFP)
		p.complTok, p.finishReason = relayMessageDelta(p.send, ev.Data, p.complTok, len(p.toolOrder) > 0, p.keptTools)

	case "message_stop":
		if p.started {
			p.send("message_stop", ev.Data)
		}

	case "ping":
		if p.started {
			p.send("ping", ev.Data)
		}

	case "error":
		if p.started {
			p.send("error", ev.Data)
			return nil
		}
		// Pre-start: the client still has nothing, so the response is STILL OURS to choose —
		// which is exactly what writing a hardcoded 502 here used to throw away (#5491). The
		// classified in-band refusals (overloaded_error, rate_limit_error, …) never reach
		// this arm at all: StreamAnthropicRaw intercepts them before message_start, retries
		// the transient ones invisibly, and surfaces the rest as a real *UpstreamStatusError
		// carrying the upstream's own status. What is left here is a frame this build cannot
		// classify, and the honest answer for it is the SAME door every other pre-start
		// failure takes — the caller's default arm, which counts the failure and falls back
		// to the buffered path — rather than a terminal 502 that also cut off the fallback by
		// setting wroteError.
		p.s.logf("gateway: unclassified upstream error frame before stream start (messages)")
		return errUpstreamInBandRefusal
	}
	return nil
}

// streamAnthropicPassthroughLive relays a live Anthropic Messages SSE stream from the
// real upstream to the client, holding tool_use blocks for kernel adjudication. It
// returns true once it owns the response (streamed a turn, or wrote a clean terminal
// error). It returns false ONLY when the upstream stream never opened and NOTHING was
// written to the client — so the caller can fall back to the buffered path with exactly
// one upstream generation having been attempted.
func (s *Server) streamAnthropicPassthroughLive(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn, upstreamKey, upstreamBeta string, compacted, contextEvent bool, hcoh harnessCoherenceInputs) bool {
	// Unwrap a dual planner to its proxy side: an API-bound request in dual mode rides
	// the same byte-preserving relay as proxy-only mode. (The caller has already
	// excluded dual-mode requests addressed to the LOCAL model via anthropicPassthroughFor.)
	hp := unwrapHTTPPlanner(s.planner)
	if hp == nil {
		return false
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return false // this writer cannot stream; let the caller use the buffered path
	}

	p := &anthropicPassthrough{
		s:               s,
		w:               w,
		r:               r,
		req:             req,
		reqTrace:        reqTrace,
		turn:            sessionTurn,
		flusher:         flusher,
		passIdx:         map[int]int{},
		toolBuf:         map[int]*sseToolAccum{},
		openClientBlock: -1,
		checkpointDir:   "",
		incidentDir:     "",
		traceID:         reqTrace,
		model:           req.Model,
		began:           time.Now(),
	}
	began := time.Now()

	err := hp.StreamAnthropicRaw(r.Context(), req.Raw, upstreamKey, upstreamBeta, p.onEvent)
	// Ensure heartbeat ticker is cleaned up on all exit paths.
	defer func() {
		if p.hbTicker != nil {
			p.hbTicker.Stop()
		}
		if p.hbDone != nil {
			close(p.hbDone)
		}
	}()
	// #3353 warm-continue: a worker that dies mid-turn (client bytes already flowing) is
	// recovered by replaying the already-delivered assistant text as a prefill turn on a
	// fresh worker with the token budget decremented, so the caller sees ONE unbroken turn
	// instead of a terminal error + cold session restart. Gated OFF by default; only the
	// text-only pre-tool prefix is replayable (see canWarmContinue).
	if err != nil && p.started && p.canWarmContinue(err) {
		s.logf("gateway: warm-continue after mid-stream death (messages): %v", err)
		err = p.warmContinue(hp, upstreamKey, upstreamBeta)
	}
	if err != nil {
		switch {
		case p.started:
			// Client bytes already flowed; we cannot change the status. Emit a terminal
			// error frame so the client's SSE parser ends cleanly, then own the response.
			// Count + surface it: the s.logf below only reaches the --log stream (OFF by
			// default), so emit the FAILED line to the default --debug-stats stderr and bump
			// the upstream-error counter — without this a mid-stream stall is a silent freeze.
			s.metrics.observeUpstreamError(err)
			s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(began))
			s.logf("gateway: upstream model error mid-stream (messages): %v", err)
			// Carry the distinct error type to the client when we can classify it (a stall),
			// so a harness sees "upstream_stalled" rather than an opaque api_error.
			errType := "api_error"
			statusClass := "error"
			boundPolicy := "max-duration=off"
			var ue *agent.UpstreamStalledError
			if errors.As(err, &ue) {
				errType = "upstream_stalled"
				statusClass = "stall"
				if ue.Kind == "max-duration" {
					statusClass = "bound:max-stream-duration"
					boundPolicy = "max-duration=5s"
				}
			}
			p.send("error", map[string]any{
				"type":  "error",
				"error": map[string]any{"type": errType, "message": "upstream model error"},
			})

			// Write partial-output checkpoint (#10672): capture the exact assistant text
			// already relayed to the client so a resume can continue from the last good state.
			cc := newCheckpointConfig()
			elapsedMS := time.Since(began).Milliseconds()
			var ttftMS int64
			if !p.firstTokenAt.IsZero() {
				ttftMS = p.firstTokenAt.Sub(began).Milliseconds()
			}
			cc.writeCheckpoint("anthropic_messages", reqTrace, p.reqModel(), "mid_stream", elapsedMS, p.asstText.String(), estimateTokens(p.asstText.String()), boundPolicy, "stream-death")

			// Write mid-stream incident packet (#10672).
			ic := newIncidentConfig()
			ic.writeIncident("anthropic_messages", "mid_stream", statusClass, boundPolicy, errType, ttftMS, elapsedMS, int64(len(p.asstText.String())), 1, elapsedMS-ttftMS)

			return true
		case p.wroteError:
			// A clean terminal HTTP error is already on the wire, so the CLIENT has its
			// answer and must not be written to twice — nothing below touches the response.
			// But "the client was told" and "the operator was told" are different questions,
			// and only the first was ever satisfied here: the producer's s.logf reaches the
			// --log stream (OFF by default), so a result-floor 502 was invisible on /metrics
			// and printed no FAILED line — an under-count nobody notices, because the number
			// is too SMALL. Count + print the real cause the producer stashed, never the
			// errPassthroughResponded value in `err`: that one means "we already responded",
			// not "the turn failed", and would file a pure control-flow artifact in the
			// coarse `other` bucket. No producer observes before setting the flag, so the
			// failure is counted exactly once.
			if p.wroteErrCause != nil {
				s.metrics.observeUpstreamError(p.wroteErrCause)
				s.renderTurnDebugError(reqTrace, "anthropic_messages", p.wroteErrCause, time.Since(began))
			}
			return true
		default:
			// The stream never opened and nothing was written. Count + surface the failure on
			// the default debug line (the s.logf calls below only reach the --log stream, OFF
			// by default), so an operator sees WHY a turn took a non-streaming path.
			s.metrics.observeUpstreamError(err)
			s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(began))
			// If the upstream answered with a DEFINITE HTTP status — a retryable one that
			// StreamAnthropicRaw already retried to exhaustion (a persistent 429/529), or a
			// non-retryable 4xx no retry can fix — re-issuing it through the buffered fallback
			// would only hit the SAME status a second time, doubling the load on a (commonly
			// shared) upstream account on the very rate limit it just refused. Surface it
			// straight to the client with its distinct status/code + any Retry-After (the
			// classify here is the PURE mapper; the metric was already observed just above, so
			// the failure is counted exactly once), exactly as the planner-live sibling does.
			// The buffered fall-back is kept only for the genuine "this wire can't stream"
			// cases — ErrStreamingUnsupported, an unreachable dial, or a nil-but-no-events
			// open — handled by the return false below.
			var statusErr *agent.UpstreamStatusError
			if errors.As(err, &statusErr) {
				s.surfaceUpstreamStatus(w, err, "upstream model error (messages passthrough; surfaced, not re-tried via buffered)")
				return true
			}
			s.logf("gateway: anthropic passthrough stream did not open (%v); falling back to buffered", err)
			return false
		}
	}
	if p.started {
		dur := time.Since(began)
		// TTFT = first-content-delta time minus turn start (zero if no delta arrived,
		// e.g. an immediate stop). This is the prefill phase; observeInferenceTimed
		// splits decode = dur - ttft from it.
		var ttft time.Duration
		if !p.firstTokenAt.IsZero() {
			ttft = p.firstTokenAt.Sub(began)
		}
		// The requested model id, which is what the planner routes on — so this relay's
		// turns are attributed by the same rule as every other path, rather than being
		// assumed vendor because the byte-preserving relay happens to talk to a vendor
		// today.
		s.metrics.observeInferenceServedTimed(s.servedLocality(p.reqModel()), p.promptTok, p.complTok, p.cacheRead, p.cacheCreate, p.finishReason, dur, ttft)
		if compacted {
			s.metrics.recordCompactionCacheRead(p.cacheRead) // OBSERVED provider cache_read on a compacted streamed turn
			s.observeResetHealth(reqTrace, p.promptTok, p.cacheRead, p.cacheCreate)
		}
		// Harness-coherence (#1132): fold this streamed turn with the content-free inbound-prefix
		// digest captured before transforms and the provider's relayed cache counters (known now at
		// stream end). Same observation the buffered path makes; the family stays path-agnostic.
		s.observeHarnessCoherenceAndArm(reqTrace, time.Now(), hcoh.inboundPrefixDigest, compacted, hcoh.fakBail,
			false /*fakWorldBreak*/, false /*sealed*/, int64(p.cacheRead), int64(p.cacheCreate), int64(p.promptTok))
		s.logInferenceTurnWithContextEvent(reqTrace, "anthropic_messages", true, agent.Usage{
			PromptTokens:             p.promptTok,
			CompletionTokens:         p.complTok,
			CacheReadInputTokens:     p.cacheRead,
			CacheCreationInputTokens: p.cacheCreate,
		}, p.finishReason, dur, compacted, contextEvent)
		s.debitServedSessionTurn(r.Context(), p.turn, agent.Usage{
			PromptTokens:             p.promptTok,
			CompletionTokens:         p.complTok,
			CacheReadInputTokens:     p.cacheRead,
			CacheCreationInputTokens: p.cacheCreate,
		}, dur, p.req.Messages)
		return true
	}
	// StreamAnthropicRaw returned nil but produced no events (no message_start) — treat
	// as a non-start so the caller can fall back rather than leave the client hanging.
	return false
}

// emitAnthropicTextBlock streams one synthetic text content block (the kernel's in-band
// note) at the next contiguous index, advancing it.
func emitAnthropicTextBlock(send func(string, any), outIdx *int, text string) {
	oi := *outIdx
	*outIdx++
	send("content_block_start", map[string]any{
		"type": "content_block_start", "index": oi,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	send("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": oi,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
	send("content_block_stop", map[string]any{"type": "content_block_stop", "index": oi})
}

// relayWithIndex relays a content_block_* event verbatim except for its `index`, which
// is rewritten to the client-facing value (held tool_use blocks leave gaps in the
// upstream numbering, so emitted blocks are renumbered contiguously). All nested
// fidelity — thinking signatures, citations, partial_json formatting — is preserved
// because only the top-level index field is touched.
func relayWithIndex(send func(string, any), event string, data json.RawMessage, idx int) {
	var m map[string]json.RawMessage
	if json.Unmarshal(data, &m) != nil {
		send(event, data) // not an object — relay verbatim
		return
	}
	if b, err := json.Marshal(idx); err == nil {
		m["index"] = b
	}
	send(event, m)
}

// relayMessageDelta relays the terminal message_delta, rewriting stop_reason to
// "end_turn" when the model asked for tools but the kernel dropped EVERY one (else the
// client hunts for a tool_use block that isn't there). It returns the turn's output
// token count and the (possibly rewritten) stop_reason for metrics.
func relayMessageDelta(send func(string, any), data json.RawMessage, complTok int, hadTools bool, keptTools int) (int, string) {
	var m map[string]json.RawMessage
	if json.Unmarshal(data, &m) != nil {
		send("message_delta", data)
		return complTok, ""
	}
	if u, ok := m["usage"]; ok {
		var usage struct {
			OutputTokens int `json:"output_tokens"`
		}
		if json.Unmarshal(u, &usage) == nil && usage.OutputTokens > 0 {
			complTok = usage.OutputTokens
		}
	}
	finish := ""
	if dm, ok := m["delta"]; ok {
		var dd map[string]json.RawMessage
		if json.Unmarshal(dm, &dd) == nil {
			if sr, ok := dd["stop_reason"]; ok {
				var s string
				_ = json.Unmarshal(sr, &s)
				finish = s
				if hadTools && keptTools == 0 && s == "tool_use" {
					dd["stop_reason"] = json.RawMessage(`"end_turn"`)
					finish = "end_turn"
					if nb, err := json.Marshal(dd); err == nil {
						m["delta"] = nb
					}
				}
			}
		}
	}
	send("message_delta", m)
	return complTok, finish
}

// anthropicStartUsage extracts the input/cache token counts from a message_start
// event's usage block (Anthropic reports input_tokens as the uncached remainder and
// cache_read/cache_creation separately), for inference metrics and session budgets.
func anthropicStartUsage(data json.RawMessage) (input, cacheRead, cacheCreate int) {
	var ms struct {
		Message struct {
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if json.Unmarshal(data, &ms) != nil {
		return 0, 0, 0
	}
	return ms.Message.Usage.InputTokens, ms.Message.Usage.CacheReadInputTokens, ms.Message.Usage.CacheCreationInputTokens
}

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
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// errPassthroughResponded is returned from the event callback when the handler has
// already written a terminal HTTP response (e.g. a result-floor error before any SSE
// byte) and the stream read must stop WITHOUT the caller falling back to a second
// upstream request.
var errPassthroughResponded = errors.New("gateway: anthropic passthrough response already written")

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

	outIdx    int         // next contiguous client-facing content-block index
	passIdx   map[int]int // upstream index -> client index (relayed blocks)
	toolBuf   map[int]*sseToolAccum
	toolOrder []int // upstream indices of held tool_use blocks, in arrival order

	admitted     bool
	resultAdms   []ResultAdmission
	flushedTools bool
	keptTools    int

	promptTok, complTok, cacheRead, cacheCreate int
	finishReason                                string

	// firstTokenAt is the wall-clock of the FIRST content delta from the upstream —
	// the prefill→decode boundary (time-to-first-token). Zero until the first delta
	// arrives; streamAnthropicPassthroughLive turns it into the prefill/decode split
	// it reports to observeInferenceTimed. A turn that never produces a delta (an
	// immediate stop) leaves it zero, so prefill is reported as "not measured" rather
	// than as the full turn.
	firstTokenAt time.Time
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
	h := p.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	p.w.WriteHeader(http.StatusOK)
	p.send = anthropicSSESender(p.w, p.flusher)
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
	kept, adjs, dropped := p.s.adjudicateProposed(p.r.Context(), calls, p.reqTrace)
	p.keptTools = len(kept)
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
	if dropped > 0 || anyRepaired(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			emitAnthropicTextBlock(p.send, &p.outIdx, note)
		}
	}
}

// onEvent is the per-SSE-event relay callback handed to StreamAnthropicRaw. It opens the
// client stream on message_start (after arming the inbound result floor once), holds and
// renumbers content blocks, batches held tool_use blocks at message_delta, and forwards
// terminal frames. A returned error stops the upstream read; errPassthroughResponded marks
// that a terminal HTTP response was already written.
func (p *anthropicPassthrough) onEvent(ev agent.AnthropicSSEEvent) error {
	switch ev.Event {
	case "message_start":
		// First event from the real API. Arm the result-side floor ONCE (so a
		// tainted inbound result still refuses a later exfil call, exactly as the
		// buffered path does), then open the client stream. Running admit here —
		// only after the upstream opened — means an open failure falls back to the
		// buffered path WITHOUT a double-admit on the same trace.
		if !p.admitted {
			p.admitted = true
			adms, aerr := p.s.admitInboundResults(p.r.Context(), p.req.Messages, p.reqTrace)
			if aerr != nil {
				p.s.logf("gateway: result-floor error (messages stream): %v", aerr)
				writeErr(p.w, http.StatusBadGateway, "upstream model error")
				p.wroteError = true
				return errPassthroughResponded
			}
			p.resultAdms = adms
		}
		p.start()
		p.promptTok, p.cacheRead, p.cacheCreate = anthropicStartUsage(ev.Data)
		p.send("message_start", ev.Data)
		// The model is about to read a quarantine stub where an inbound tool result
		// was paged out — say so in-band, as a LEADING text block, before its prose.
		if note := resultAdmissionNote(p.resultAdms); note != "" {
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
		oi := p.outIdx
		p.outIdx++
		p.passIdx[d.Index] = oi
		relayWithIndex(p.send, "content_block_start", ev.Data, oi)

	case "content_block_delta":
		var d struct {
			Index int `json:"index"`
			Delta struct {
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
		if oi, ok := p.passIdx[d.Index]; ok {
			relayWithIndex(p.send, "content_block_delta", ev.Data, oi)
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
		if oi, ok := p.passIdx[d.Index]; ok {
			relayWithIndex(p.send, "content_block_stop", ev.Data, oi)
		}

	case "message_delta":
		if !p.flushedTools {
			p.flushedTools = true
			p.flushHeldTools()
		}
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
		p.s.logf("gateway: upstream error before stream start (messages)")
		writeErr(p.w, http.StatusBadGateway, "upstream model error")
		p.wroteError = true
		return errPassthroughResponded
	}
	return nil
}

// streamAnthropicPassthroughLive relays a live Anthropic Messages SSE stream from the
// real upstream to the client, holding tool_use blocks for kernel adjudication. It
// returns true once it owns the response (streamed a turn, or wrote a clean terminal
// error). It returns false ONLY when the upstream stream never opened and NOTHING was
// written to the client — so the caller can fall back to the buffered path with exactly
// one upstream generation having been attempted.
func (s *Server) streamAnthropicPassthroughLive(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn, upstreamKey, upstreamBeta string, compacted bool) bool {
	hp, ok := s.planner.(*agent.HTTPPlanner)
	if !ok {
		return false
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return false // this writer cannot stream; let the caller use the buffered path
	}

	p := &anthropicPassthrough{
		s: s, w: w, r: r, req: req, reqTrace: reqTrace, turn: sessionTurn, flusher: flusher,
		passIdx: map[int]int{},
		toolBuf: map[int]*sseToolAccum{},
	}
	began := time.Now()

	err := hp.StreamAnthropicRaw(r.Context(), req.Raw, upstreamKey, upstreamBeta, p.onEvent)
	if err != nil {
		switch {
		case p.started:
			// Client bytes already flowed; we cannot change the status. Emit a terminal
			// error frame so the client's SSE parser ends cleanly, then own the response.
			s.logf("gateway: upstream model error mid-stream (messages): %v", err)
			p.send("error", map[string]any{
				"type":  "error",
				"error": map[string]any{"type": "api_error", "message": "upstream model error"},
			})
			return true
		case p.wroteError:
			return true // a clean terminal HTTP error was already written
		default:
			// The stream never opened and nothing was written — let the caller fall back
			// to the buffered path (exactly one upstream generation total).
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
		s.metrics.observeInferenceTimed(p.promptTok, p.complTok, p.cacheRead, p.finishReason, dur, ttft)
		if compacted {
			s.metrics.recordCompactionCacheRead(p.cacheRead) // OBSERVED provider cache_read on a compacted streamed turn
			s.observeResetHealth(reqTrace, p.promptTok, p.cacheRead, p.cacheCreate)
		}
		s.logInferenceTurn(reqTrace, "anthropic_messages", true, agent.Usage{
			PromptTokens:             p.promptTok,
			CompletionTokens:         p.complTok,
			CacheReadInputTokens:     p.cacheRead,
			CacheCreationInputTokens: p.cacheCreate,
		}, p.finishReason, dur, compacted)
		s.debitServedSessionTurn(r.Context(), p.turn, agent.Usage{
			PromptTokens:             p.promptTok,
			CompletionTokens:         p.complTok,
			CacheReadInputTokens:     p.cacheRead,
			CacheCreationInputTokens: p.cacheCreate,
		}, p.req.Messages)
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

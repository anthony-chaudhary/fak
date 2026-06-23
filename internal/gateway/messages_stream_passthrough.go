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

// streamAnthropicPassthroughLive relays a live Anthropic Messages SSE stream from the
// real upstream to the client, holding tool_use blocks for kernel adjudication. It
// returns true once it owns the response (streamed a turn, or wrote a clean terminal
// error). It returns false ONLY when the upstream stream never opened and NOTHING was
// written to the client — so the caller can fall back to the buffered path with exactly
// one upstream generation having been attempted.
func (s *Server) streamAnthropicPassthroughLive(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace, upstreamKey, upstreamBeta string) bool {
	hp, ok := s.planner.(*agent.HTTPPlanner)
	if !ok {
		return false
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return false // this writer cannot stream; let the caller use the buffered path
	}

	var (
		send       func(string, any)
		started    bool
		wroteError bool

		outIdx    int             // next contiguous client-facing content-block index
		passIdx   = map[int]int{} // upstream index -> client index (relayed blocks)
		toolBuf   = map[int]*sseToolAccum{}
		toolOrder []int // upstream indices of held tool_use blocks, in arrival order

		admitted     bool
		resultAdms   []ResultAdmission
		flushedTools bool
		keptTools    int

		promptTok, complTok, cacheRead int
		finishReason                   string
		began                          = time.Now()
	)

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
	}

	// flushHeldTools adjudicates every buffered tool_use block as one batch (exact
	// parity with the buffered path's single adjudicateProposed call), emits the
	// survivors as tool_use blocks with our contiguous indices, and appends an in-band
	// note when the kernel dropped or repaired a call. Runs once, just before the
	// terminal message_delta.
	flushHeldTools := func() {
		if len(toolOrder) == 0 {
			return
		}
		calls := make([]agent.ToolCall, 0, len(toolOrder))
		for _, ui := range toolOrder {
			ta := toolBuf[ui]
			args := strings.TrimSpace(ta.args.String())
			if args == "" {
				args = "{}"
			}
			calls = append(calls, agent.ToolCall{
				ID: ta.id, Type: "function",
				Function: agent.Func{Name: ta.name, Arguments: args},
			})
		}
		kept, adjs, dropped := s.adjudicateProposed(r.Context(), calls, reqTrace)
		keptTools = len(kept)
		// Render survivors through the SAME helper the buffered path uses, so the
		// tool_use blocks (id preserved, input as a normalized object) are byte-shaped
		// identically — only the framing differs.
		for _, blk := range agent.AnthropicResponseBlocks(agent.Message{Role: agent.RoleAssistant, ToolCalls: kept}) {
			oi := outIdx
			outIdx++
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
		if dropped > 0 || anyRepaired(adjs) {
			if note := adjudicationNote(adjs); note != "" {
				emitAnthropicTextBlock(send, &outIdx, note)
			}
		}
	}

	onEvent := func(ev agent.AnthropicSSEEvent) error {
		switch ev.Event {
		case "message_start":
			// First event from the real API. Arm the result-side floor ONCE (so a
			// tainted inbound result still refuses a later exfil call, exactly as the
			// buffered path does), then open the client stream. Running admit here —
			// only after the upstream opened — means an open failure falls back to the
			// buffered path WITHOUT a double-admit on the same trace.
			if !admitted {
				admitted = true
				adms, aerr := s.admitInboundResults(r.Context(), req.Messages, reqTrace)
				if aerr != nil {
					s.logf("gateway: result-floor error (messages stream): %v", aerr)
					writeErr(w, http.StatusBadGateway, "upstream model error")
					wroteError = true
					return errPassthroughResponded
				}
				resultAdms = adms
			}
			start()
			promptTok, cacheRead = anthropicStartUsage(ev.Data)
			send("message_start", ev.Data)
			// The model is about to read a quarantine stub where an inbound tool result
			// was paged out — say so in-band, as a LEADING text block, before its prose.
			if note := resultAdmissionNote(resultAdms); note != "" {
				emitAnthropicTextBlock(send, &outIdx, note)
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
				toolBuf[d.Index] = &sseToolAccum{id: d.ContentBlock.ID, name: d.ContentBlock.Name}
				toolOrder = append(toolOrder, d.Index)
				return nil // HELD until adjudicated
			}
			oi := outIdx
			outIdx++
			passIdx[d.Index] = oi
			relayWithIndex(send, "content_block_start", ev.Data, oi)

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
			if ta, held := toolBuf[d.Index]; held {
				ta.args.WriteString(d.Delta.PartialJSON) // accumulate off-wire
				return nil
			}
			if oi, ok := passIdx[d.Index]; ok {
				relayWithIndex(send, "content_block_delta", ev.Data, oi)
			}

		case "content_block_stop":
			var d struct {
				Index int `json:"index"`
			}
			if json.Unmarshal(ev.Data, &d) != nil {
				return nil
			}
			if _, held := toolBuf[d.Index]; held {
				return nil // emitted (or dropped) as a batch at message_delta
			}
			if oi, ok := passIdx[d.Index]; ok {
				relayWithIndex(send, "content_block_stop", ev.Data, oi)
			}

		case "message_delta":
			if !flushedTools {
				flushedTools = true
				flushHeldTools()
			}
			complTok, finishReason = relayMessageDelta(send, ev.Data, complTok, len(toolOrder) > 0, keptTools)

		case "message_stop":
			if started {
				send("message_stop", ev.Data)
			}

		case "ping":
			if started {
				send("ping", ev.Data)
			}

		case "error":
			if started {
				send("error", ev.Data)
				return nil
			}
			s.logf("gateway: upstream error before stream start (messages)")
			writeErr(w, http.StatusBadGateway, "upstream model error")
			wroteError = true
			return errPassthroughResponded
		}
		return nil
	}

	err := hp.StreamAnthropicRaw(r.Context(), req.Raw, upstreamKey, upstreamBeta, onEvent)
	if err != nil {
		switch {
		case started:
			// Client bytes already flowed; we cannot change the status. Emit a terminal
			// error frame so the client's SSE parser ends cleanly, then own the response.
			s.logf("gateway: upstream model error mid-stream (messages): %v", err)
			send("error", map[string]any{
				"type":  "error",
				"error": map[string]any{"type": "api_error", "message": "upstream model error"},
			})
			return true
		case wroteError:
			return true // a clean terminal HTTP error was already written
		default:
			// The stream never opened and nothing was written — let the caller fall back
			// to the buffered path (exactly one upstream generation total).
			s.logf("gateway: anthropic passthrough stream did not open (%v); falling back to buffered", err)
			return false
		}
	}
	if started {
		s.metrics.observeInference(promptTok, complTok, cacheRead, finishReason, time.Since(began))
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

// anthropicStartUsage extracts the input + cache-read token counts from a message_start
// event's usage block (Anthropic reports input_tokens as the uncached remainder and
// cache_read_input_tokens separately), for inference metrics.
func anthropicStartUsage(data json.RawMessage) (input, cacheRead int) {
	var ms struct {
		Message struct {
			Usage struct {
				InputTokens          int `json:"input_tokens"`
				CacheReadInputTokens int `json:"cache_read_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if json.Unmarshal(data, &ms) != nil {
		return 0, 0
	}
	return ms.Message.Usage.InputTokens, ms.Message.Usage.CacheReadInputTokens
}

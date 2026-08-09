package gateway

// sse_stream_preamble.go — the WIRE-AGNOSTIC half of the early SSE preamble, shared by
// every buffered streaming surface the gateway serves (#5514, generalizing #5399).
//
// #5399 fixed the buffered chat stream: a stream:true turn served by a Complete-only
// planner (agent.InKernelPlanner is not an agent.StreamingPlanner, so streamChatLive
// declines every in-kernel serve) used to put its FIRST byte — status line, headers,
// opening chunk — on the wire only after completeServed returned the whole turn. For the
// duration of a sharded multi-rank decode the client could not tell an accepted
// streaming request from a dead socket; 1120+ seconds on the original 8-rank GLM-5.2 EP
// report.
//
// The legacy OpenAI text-completion wire (POST /v1/completions) rides the SAME served
// path and the SAME Complete-only planners, so it had the SAME stall (#5514). What the
// two wires do NOT share is their frame shape: chat emits `chat.completion.chunk`
// objects carrying a `delta`, the legacy wire emits `text_completion` objects carrying a
// bare `text`, and collapsing them would break real legacy clients. The preamble TIMING
// is the bug; the schema is not.
//
// So the timing lives here, once, and each wire supplies only its own frames:
//
//   - sseStreamWriter owns the stream identity (id + created, minted at REQUEST time
//     because the opening frame now leaves before the turn exists and every later frame
//     has to agree with it), the SSE headers, the idempotent open-and-flush, and the
//     in-band failure path taken once the status line is spent.
//   - chatStreamWriter (chat_stream_preamble.go) and completionStreamWriter
//     (completions.go) embed it and add their own chunk builder + opening frame.

import (
	"net/http"
	"time"
)

// sseStreamWriter is the transport half of one buffered SSE turn. It deliberately
// carries no wire shape: `model` is the REQUEST model, announced before the upstream can
// report what it served and therefore held CONSTANT for the life of the stream (the rule
// OpenAI follows and streamChatLive already followed), and `id`/`created` are minted once
// so every frame agrees with the preamble.
type sseStreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	id      string
	created int64
	model   string
	opened  bool
}

// newSSEStreamWriter mints the stream identity at REQUEST time. idPrefix is the wire's
// own id namespace ("chatcmpl-fak-" for chat, "cmpl-fak-" for the legacy text wire), so a
// client that keys on the prefix still sees the right one.
func newSSEStreamWriter(w http.ResponseWriter, idPrefix, model string) sseStreamWriter {
	f, _ := w.(http.Flusher)
	now := time.Now()
	return sseStreamWriter{
		w:       w,
		flusher: f,
		id:      idPrefix + itoa(uint64(now.UnixNano())),
		created: now.Unix(),
		model:   model,
	}
}

// openWith writes the status line, the SSE headers, and the caller's opening frame, then
// FLUSHES (writeSSEData flushes) so the bytes actually leave the socket instead of
// sitting in the response buffer until the handler returns. It is idempotent, and it must
// be called only once every PRE-decode refusal has been passed: after it returns the HTTP
// status is spent and errors can only be reported in-band (see fail).
//
// It returns the write error when the client has already gone away, so the caller can
// abandon the turn instead of decoding for a socket nobody is reading.
func (p *sseStreamWriter) openWith(first any) error {
	if p.opened {
		return nil
	}
	p.opened = true
	h := p.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	p.w.WriteHeader(http.StatusOK)
	return writeSSEData(p.w, first)
}

// fail terminates an OPENED stream on a failure the buffered path would otherwise have
// answered with a real HTTP status. The 200 is already on the wire, so the classified
// status/code/message ride an OpenAI-shaped SSE error event, followed by [DONE] so the
// client's SSE parser ends cleanly rather than hanging on a truncated stream. msg is the
// same client-facing string writeErrCode would have sent — the upstream's raw body never
// crosses this boundary. The error envelope is not a wire-shaped chunk, so both surfaces
// report failures identically.
func (p *sseStreamWriter) fail(status int, code, msg string) {
	var codeVal any
	if code != "" {
		codeVal = code
	}
	_ = writeSSEData(p.w, map[string]any{
		"error": map[string]any{"message": msg, "type": errType(status), "code": codeVal, "param": nil},
	})
	writeSSEDone(p.w, p.flusher)
}

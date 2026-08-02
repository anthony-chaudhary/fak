package gateway

// chat_stream_preamble.go — the EARLY SSE preamble for the BUFFERED chat stream
// (#5399, the remaining half of #4855).
//
// POST /v1/chat/completions with stream:true has two servers. The LIVE one
// (streamChatLive) only exists for a planner that implements agent.StreamingPlanner;
// agent.InKernelPlanner implements Complete ONLY, so the pure-fak in-kernel serve —
// the exact topology the 8-rank GLM-5.2 EP report came from — declines it and
// falls through to the buffered path. That path used to write its FIRST byte (status
// line, headers, opening chunk, everything) only after completeServed returned the
// whole turn. For the full duration of a sharded multi-rank decode the client could
// not distinguish an accepted streaming request from a dead socket; on the original
// report that was 1120+ seconds of silence.
//
// chatStreamWriter splits that single write into two halves so the socket can prove
// itself alive immediately:
//
//   - open() puts the 200, the SSE headers, and the opening role chunk on the wire
//     BEFORE the decode starts. It is called strictly AFTER every pre-decode refusal
//     (method/body/sampling 400s, the session refusal, the inbound-result 502) so no
//     refusal ever loses its real HTTP status to a premature 200.
//   - finish() emits the adjudicated remainder — content deltas, the surviving tool
//     calls, the terminal finish/usage chunk, [DONE] — once the turn exists.
//   - fail() ends a stream that has ALREADY announced 200: the status line is spent,
//     so a post-preamble upstream failure surfaces as an SSE error event + [DONE]
//     (what OpenAI does for a mid-stream failure) instead of truncating the socket.
//
// Two shape consequences follow, and both are deliberate:
//
//  1. The opening chunk can no longer carry the adjudicated tool calls — they are not
//     known before the decode. They move into their own later delta. OpenAI clients
//     accumulate tool_calls by index ACROSS deltas, so this is wire-compatible, and it
//     is the shape streamChatLive already emits on trunk (content, then a dedicated
//     tool-call delta).
//  2. `model` is announced before the upstream can report what it served, so the
//     streamed wire carries the REQUEST model, constant for the life of the stream —
//     again the rule streamChatLive already follows, and the one OpenAI follows (the
//     field never changes mid-stream). The #82 served-model echo therefore stays on
//     the non-streaming JSON response, where it can still be truthful; when the two
//     diverge the gateway LOGS it rather than silently flipping the field mid-stream.
//
// The wire-INDEPENDENT mechanics behind open()/fail() — the stream identity, the SSE
// headers, the flush, the in-band error envelope — moved to sse_stream_preamble.go when
// the legacy /v1/completions wire adopted the same timing (#5514). chatStreamWriter now
// embeds that writer and supplies only chat's own frames; the behavior above is
// unchanged.

import (
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// chatStreamWriter owns the client half of one stream:true chat completion served by
// the buffered path. It is created (and opened) before the decode and finished after
// it, so the id/created/model a client sees are stable across every chunk. Everything
// wire-INDEPENDENT — that identity, the SSE headers, the idempotent open-and-flush, the
// in-band failure path — lives in the embedded sseStreamWriter, which the legacy
// text-completion wire shares (#5514); only the chunk shape below is chat's own.
type chatStreamWriter struct {
	sseStreamWriter
}

// newChatStreamWriter mints the stream identity (id + created) at REQUEST time rather
// than at completion time, because the opening chunk now leaves before the turn
// exists and every later chunk has to agree with it. model is the request model (see
// the file comment on the #82 seam).
func newChatStreamWriter(w http.ResponseWriter, model string) *chatStreamWriter {
	return &chatStreamWriter{sseStreamWriter: newSSEStreamWriter(w, "chatcmpl-fak-", model)}
}

// chunk builds one OpenAI `chat.completion.chunk` carrying this stream's identity.
// Index is 0 because the gateway normalizes every served turn to a single choice.
func (p *chatStreamWriter) chunk(d ChatDelta, finish *string, usage *agent.Usage) ChatStreamResponse {
	return ChatStreamResponse{
		ID:      p.id,
		Object:  "chat.completion.chunk",
		Created: p.created,
		Model:   p.model,
		Choices: []ChatStreamChoice{{Index: 0, Delta: d, FinishReason: finish}},
		Usage:   usage,
	}
}

// open writes the status line, the SSE headers, and the opening ROLE chunk — chat's
// own opening frame — through the shared preamble, which flushes so the bytes actually
// leave the socket instead of sitting in the response buffer until the handler returns.
// It is idempotent, and it must be called only once every pre-decode refusal has been
// passed: after it returns, the HTTP status is spent and errors can only be reported
// in-band (see sseStreamWriter.fail).
//
// It returns the write error when the client has already gone away, so the caller can
// abandon the turn instead of decoding for a socket nobody is reading.
func (p *chatStreamWriter) open() error {
	return p.openWith(p.chunk(ChatDelta{Role: agent.RoleAssistant}, nil, nil))
}

// writeChatCompletionStream emits the buffered, adjudicated turn onto an ALREADY-OPENED
// chat SSE stream: the incremental content deltas, then the surviving tool calls in
// their own delta, then the terminal finish/usage chunk and [DONE]. The opening role
// chunk left in open(), before the decode — that split is the whole point of #5399.
func writeChatCompletionStream(p *chatStreamWriter, resp ChatResponse) {
	if err := p.open(); err != nil {
		return
	}
	choice := resp.Choices[0]

	// Content chunks: stream the adjudicated content as incremental fragments, one
	// SSE event per fragment, the way a real OpenAI stream delivers tokens — rather
	// than collapsing the whole reply into a single delta. segmentContent preserves
	// every byte, so concatenating the content deltas reproduces the reply exactly.
	for _, seg := range segmentContent(choice.Message.Content) {
		if err := writeSSEData(p.w, p.chunk(ChatDelta{Content: seg}, nil, nil)); err != nil {
			return
		}
	}

	// The surviving (adjudicated) tool calls in a dedicated delta. They cannot ride the
	// opening chunk any more — that left before the decode — and a standard OpenAI client
	// accumulates tool_calls by index across deltas, so the wire contract is preserved.
	if calls := streamToolCalls(choice.Message.ToolCalls); len(calls) > 0 {
		if err := writeSSEData(p.w, p.chunk(ChatDelta{ToolCalls: calls}, nil, nil)); err != nil {
			return
		}
	}

	finish := choice.FinishReason
	final := p.chunk(ChatDelta{}, &finish, &resp.Usage)
	final.Fak = resp.Fak
	if err := writeSSEData(p.w, final); err != nil {
		return
	}
	writeSSEDone(p.w, p.flusher)
}

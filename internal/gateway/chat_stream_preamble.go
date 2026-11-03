package gateway

// chat_stream_preamble.go — the EARLY SSE preamble for the BUFFERED chat stream
// (#5399, the remaining half of #4855).
//
// POST /v1/chat/completions with stream:true has two servers. The LIVE one
// (streamChatLive) serves any planner that implements agent.StreamingPlanner;
// agent.InKernelPlanner now implements StreamingPlanner (its CompleteStream lives in
// internal/agent/inkernel_planner.go), so the pure-fak in-kernel serve — the exact
// topology the 8-rank GLM-5.2 EP report came from — takes the live path. The buffered
// preamble below remains the fallback for Complete-ONLY planners (e.g. the modeled
// collectivePlanner), whose first byte used to be written (status line, headers,
// opening chunk, everything) only after completeServed returned the whole turn. For
// the full duration of a sharded multi-rank decode such a client could not distinguish
// an accepted streaming request from a dead socket; on the original report that was
// 1120+ seconds of silence.
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
//
// Opt-in route-decision preamble (lemonade borrow, ADAPT verdict — re-expressed, no
// copied code; source lemonade-sdk/lemonade src/cpp/server/route_decision_response.cpp
// @bc8d99ccfee301742f1dd86a3f6ec406bb8a863d, Apache-2.0): streaming clients behind
// proxies that drop custom headers cannot see which upstream and model served them,
// because every routing fact this gateway publishes otherwise rides response HEADERS.
// When FAK_GATEWAY_STREAM_ROUTE_PREAMBLE names the serve's chosen upstream, open()
// emits exactly ONE SSE comment line (": " prefix — spec-legal between events, ignored
// by OpenAI-compatible parsers) carrying that upstream plus the stream's announced
// model, after the status line and SSE headers and strictly before the first data
// chunk. It is a comment, never a data event: the first data frame stays byte-identical
// with the flag off, and the wire gains no non-standard event shape.

import (
	"fmt"
	"net/http"
	"os"
	"strings"

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
	// includeUsage is the client's stream_options.include_usage opt-in, captured once
	// at construction from the SAME req.DeclaredStreamUsage() reader the live emitter
	// uses, so the two native emitters can never disagree. When true every non-usage
	// chunk — including the opening role chunk — carries `usage:null`, and the turn is
	// closed by exactly one usage-only chunk; when false the field is never written.
	includeUsage bool
	// routePreamble is the opt-in route-decision comment payload (see the file
	// header), or "" when the flag is off. When non-empty, open() installs a one-shot
	// transport shim so the line ": <routePreamble>\n\n" lands between the SSE headers
	// and the first data frame without touching the shared openWith mechanics.
	routePreamble string
}

// newChatStreamWriter mints the stream identity (id + created) at REQUEST time rather
// than at completion time, because the opening chunk now leaves before the turn
// exists and every later chunk has to agree with it. model is the request model (see
// the file comment on the #82 seam). includeUsage is the client's stream_options
// opt-in (see the field comment). The route-decision preamble payload is resolved
// here, at request time, from the operator's opt-in flag (same per-request env read
// as the FAK_GATEWAY_JSON_SCHEMA_* caps).
func newChatStreamWriter(w http.ResponseWriter, model string, includeUsage bool) *chatStreamWriter {
	return &chatStreamWriter{
		sseStreamWriter: newSSEStreamWriter(w, "chatcmpl-fak-", model),
		includeUsage:    includeUsage,
		routePreamble:   streamRoutePreamblePayload(os.Getenv(streamRoutePreambleEnv), model),
	}
}

// streamRoutePreambleEnv opts the buffered chat stream into the in-band route-decision
// preamble. Its value NAMES the serve's chosen upstream — the backend identity the
// operator declares for this process ("inkernel", "proxy", ...). The value is the name
// rather than a bare switch because the per-request planner choice is not known before
// the decode (the preamble leaves before it) and no process-wide registry of the
// selected planner exists at this seam; the operator who stands up the serve knows the
// topology they configured. Empty, "0", "false", "off", and "no" (any case) leave the
// preamble off, which is the default.
const streamRoutePreambleEnv = "FAK_GATEWAY_STREAM_ROUTE_PREAMBLE"

// streamRoutePreamblePayload builds the comment payload "fak-route upstream=<label>
// model=<model>", or "" when the flag disables the preamble. label is the operator's
// declared upstream (sanitized); model is the stream's ANNOUNCED model — the constant
// every chunk on this wire carries — not the post-decode served-model echo, which stays
// on the non-streaming JSON response per the #82 seam. Sanitization keeps the value to
// one physical line so a misconfigured flag can never terminate or inject into the
// comment frame.
func streamRoutePreamblePayload(raw, model string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "false", "off", "no":
		return ""
	}
	upstream := sseCommentField(raw)
	if upstream == "" {
		return ""
	}
	m := sseCommentField(model)
	if m == "" {
		m = "unknown"
	}
	return "fak-route upstream=" + upstream + " model=" + m
}

// sseCommentField strips CR/LF and all other ASCII control characters (and trims
// surrounding spaces) so a configured value can never break out of the single-line
// SSE comment frame it rides in.
func sseCommentField(v string) string {
	var b strings.Builder
	for _, r := range v {
		if r >= ' ' && r != '\u007f' {
			b.WriteRune(r)
		}
	}
	return strings.TrimSpace(b.String())
}

// routePreambleWriter is the one-shot transport shim open() installs when the
// route-decision preamble is opted in. Its FIRST Write emits the SSE comment line
// (then flushes) before delegating, so the comment lands after the status line and
// headers the shared preamble has already committed and strictly before the first
// data frame; every later write passes through untouched. Header/WriteHeader forward
// verbatim, and Flush implements http.Flusher so the shared preamble's flush-on-write
// behavior (writeSSEData asserts the concrete writer) is unchanged with the shim in
// place.
type routePreambleWriter struct {
	inner   http.ResponseWriter
	comment string
	armed   bool
}

func (r *routePreambleWriter) Header() http.Header { return r.inner.Header() }

func (r *routePreambleWriter) WriteHeader(statusCode int) { r.inner.WriteHeader(statusCode) }

func (r *routePreambleWriter) Write(b []byte) (int, error) {
	if r.armed {
		r.armed = false
		if _, err := fmt.Fprintf(r.inner, ": %s\n\n", r.comment); err != nil {
			return 0, err
		}
		r.Flush()
	}
	return r.inner.Write(b)
}

func (r *routePreambleWriter) Flush() {
	if f, ok := r.inner.(http.Flusher); ok {
		f.Flush()
	}
}

// chunk builds one OpenAI `chat.completion.chunk` carrying this stream's identity.
// Index is 0 because the gateway normalizes every served turn to a single choice.
// UsageEmitted marks a non-usage chunk of an opted-in stream so MarshalJSON renders
// `usage:null`; a non-opted stream leaves it false and omits the field entirely.
func (p *chatStreamWriter) chunk(d ChatDelta, finish *string, usage *agent.Usage) ChatStreamResponse {
	return ChatStreamResponse{
		ID:           p.id,
		Object:       "chat.completion.chunk",
		Created:      p.created,
		Model:        p.model,
		Choices:      []ChatStreamChoice{{Index: 0, Delta: d, FinishReason: finish}},
		Usage:        usage,
		UsageEmitted: p.includeUsage && usage == nil,
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
//
// With the route-decision preamble opted in, the one-shot comment shim is installed
// BEFORE openWith runs, so the comment frame is injected on the first write — after
// the status line and SSE headers, before the first data frame — while the shared
// openWith mechanics stay untouched (they live in sse_stream_preamble.go, outside
// this file's seam).
func (p *chatStreamWriter) open() error {
	if p.routePreamble != "" && !p.opened {
		p.w = &routePreambleWriter{inner: p.w, comment: p.routePreamble, armed: true}
	}
	return p.openWith(p.chunk(ChatDelta{Role: agent.RoleAssistant}, nil, nil))
}

// writeChatCompletionStream emits the buffered, adjudicated turn onto an ALREADY-OPENED
// chat SSE stream: the incremental content deltas, then the surviving tool calls in
// their own delta, then the terminal finish chunk, the optional opted-in usage-only
// chunk, and [DONE]. The opening role chunk left in open(), before the decode — that
// split is the whole point of #5399.
//
// The client's stream_options.include_usage opt-in was captured on the writer at
// construction, so every chunk carries the correct `usage:null` / omitted / usage
// tri-state and the terminal usage-only frame is emitted only when requested.
func writeChatCompletionStream(p *chatStreamWriter, resp ChatResponse) {
	if err := p.open(); err != nil {
		return
	}
	choice := resp.Choices[0]
	for _, seg := range segmentContent(choice.Message.ReasoningContent) {
		if err := writeSSEData(p.w, p.chunk(ChatDelta{ReasoningContent: seg}, nil, nil)); err != nil {
			return
		}
	}

	// Content chunks: stream the adjudicated content as incremental fragments, one
	// SSE event per fragment, the way a real OpenAI stream delivers tokens — rather
	// than collapsing the whole reply into a single delta. segmentContent preserves
	// every byte, so concatenating the content deltas reproduces the reply exactly.
	// Every non-usage chunk of an opted-in stream nulls `usage` (UsageEmitted); a
	// non-opted stream sets nothing, so the field is omitted entirely.
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

	// The finish chunk never carries usage: when the option is off the usage field is
	// omitted, and when it is on the field is null here and the object rides its own
	// terminal frame below (the OpenAI stream_options.include_usage contract).
	finish := choice.FinishReason
	final := p.chunk(ChatDelta{}, &finish, nil)
	final.Fak = resp.Fak
	// Upstream timings ride the terminal chunk, matching streamChatLive.
	if !p.includeUsage {
		final.Timings = resp.Timings
	}
	if err := writeSSEData(p.w, final); err != nil {
		return
	}
	// Opted-in usage arrives in its OWN terminal frame with an EMPTY choices array,
	// never attached to the finish-bearing choice — the same shape streamChatLive emits.
	if p.includeUsage {
		usage := resp.Usage
		usageChunk := usageOnlyChunk(p.id, p.model, p.created, &usage)
		usageChunk.Timings = resp.Timings
		if err := writeSSEData(p.w, usageChunk); err != nil {
			return
		}
	}
	writeSSEDone(p.w, p.flusher)
}

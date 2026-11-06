package anthropicbridge

// stream.go is the streaming half of the translator: it converts a live OpenAI
// Chat Completions SSE stream into an Anthropic Messages SSE event stream, and
// carries the block/terminal emitters the gateway's Messages SSE wire renders.
// Everything here is the public promotion of code the in-fak path already runs
// (internal/agent/stream.go + internal/gateway stop_buffer.go /
// messages_stream_planner.go / messages.go), ported verbatim so a routed turn
// and an in-fak turn see the same wire shapes.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Event is one Anthropic Messages SSE event: Name is the SSE event name
// ("message_start", "content_block_start", "content_block_delta",
// "content_block_stop", "message_delta", "message_stop", "ping", "error"); Data
// is the pre-marshaled JSON payload.
type Event struct {
	Name string
	Data []byte
}

// Frame renders the SSE wire bytes for one event — the same framing the
// gateway's anthropicSSESender emits: "event: <name>\ndata: <data>\n\n".
func (e Event) Frame() []byte {
	buf := make([]byte, 0, len("event: ")+len(e.Name)+len("\ndata: ")+len(e.Data)+len("\n\n"))
	buf = append(buf, "event: "...)
	buf = append(buf, e.Name...)
	buf = append(buf, "\ndata: "...)
	buf = append(buf, e.Data...)
	buf = append(buf, "\n\n"...)
	return buf
}

// ErrorEvent renders the in-band SSE error event the gateway emits on a failed
// streamed turn: {"type":"error","error":{"type":"api_error","message":msg}}.
func ErrorEvent(message string) Event {
	data, err := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "api_error", "message": message},
	})
	if err != nil {
		// Unreachable for a fixed map of strings; keep the wire shape anyway.
		data = []byte(`{"type":"error","error":{"type":"api_error","message":""}}`)
	}
	return Event{Name: "error", Data: data}
}

// StopHoldbackBuffer is an incremental stop-string holdback buffer for streaming generation
// (#10719, borrowed from vLLM detokenizer).
// It delays emitting up to max(len(stop)) - 1 characters during streaming generation until
// either the stop condition resolves (and the stop sequence is trimmed) or generation finishes
// (and the held buffer is flushed) — preventing partial stop markers from prematurely leaking
// across SSE chunk boundaries to streaming clients.
type StopHoldbackBuffer struct {
	stops       []string
	maxHoldback int
	buffered    strings.Builder
	stopped     bool
	matchedStop string
}

// NewStopHoldbackBuffer creates an incremental holdback buffer for the specified stop sequences.
func NewStopHoldbackBuffer(stops []string) *StopHoldbackBuffer {
	maxLen := 0
	cleanStops := make([]string, 0, len(stops))
	for _, s := range stops {
		if s != "" {
			cleanStops = append(cleanStops, s)
			if len(s) > maxLen {
				maxLen = len(s)
			}
		}
	}
	holdback := 0
	if maxLen > 1 {
		holdback = maxLen - 1
	}
	return &StopHoldbackBuffer{
		stops:       cleanStops,
		maxHoldback: holdback,
	}
}

// Append processes an incoming text chunk delta during streaming.
// It returns any text that is safe to emit immediately.
// If a stop sequence is encountered, it sets Stopped() = true, records MatchedStop(),
// and returns the text prior to the stop sequence (never emitting the stop sequence).
// Any trailing text that could be a partial prefix of a stop sequence is held in buffer.
func (b *StopHoldbackBuffer) Append(chunk string) string {
	if b == nil || chunk == "" || b.stopped {
		return ""
	}
	b.buffered.WriteString(chunk)
	current := b.buffered.String()

	// Find the earliest matching stop string in the accumulated text
	earliestIdx := -1
	var earliestStop string
	for _, stop := range b.stops {
		idx := strings.Index(current, stop)
		if idx >= 0 {
			if earliestIdx == -1 || idx < earliestIdx || (idx == earliestIdx && len(stop) > len(earliestStop)) {
				earliestIdx = idx
				earliestStop = stop
			}
		}
	}

	if earliestIdx >= 0 {
		// Stop sequence found: terminate stream holdback
		b.stopped = true
		b.matchedStop = earliestStop
		toEmit := current[:earliestIdx]
		b.buffered.Reset()
		return toEmit
	}

	// No full stop string matched yet.
	// Emit text up to (len(current) - maxHoldback), keeping the tail buffered in case it forms a partial stop.
	if b.maxHoldback > 0 && len(current) > b.maxHoldback {
		safeLen := len(current) - b.maxHoldback
		toEmit := current[:safeLen]
		tail := current[safeLen:]
		b.buffered.Reset()
		b.buffered.WriteString(tail)
		return toEmit
	} else if b.maxHoldback == 0 {
		toEmit := current
		b.buffered.Reset()
		return toEmit
	}

	return ""
}

// Flush is called when streaming finishes. If generation was not terminated by a stop
// sequence, Flush emits all remaining buffered characters.
func (b *StopHoldbackBuffer) Flush() string {
	if b == nil || b.stopped {
		return ""
	}
	tail := b.buffered.String()
	b.buffered.Reset()
	return tail
}

// Stopped reports whether streaming was halted by a matched stop sequence.
func (b *StopHoldbackBuffer) Stopped() bool {
	if b == nil {
		return false
	}
	return b.stopped
}

// MatchedStop returns the matched stop sequence that halted generation, if any.
func (b *StopHoldbackBuffer) MatchedStop() string {
	if b == nil {
		return ""
	}
	return b.matchedStop
}

// StreamTranslator converts a live OpenAI Chat Completions SSE stream into an
// Anthropic Messages SSE event stream — the streaming variant of the request
// and response translation. Feed each upstream SSE `data:` payload (the JSON
// chunk body, or the literal "[DONE]"); each call returns the Anthropic events
// to forward so far. Text deltas stream live through a StopHoldbackBuffer built
// from the request's stop_sequences (partial stop markers never leak across
// chunk boundaries); tool-call argument fragments are HELD until the stream
// completes (the in-fak held-tool-calls shape) and emitted by Finish.
//
// The seam serves the default single-choice Messages body (choice zero). The
// planner-owned concerns of the in-fak CompleteStream loop stay there — the
// stall reader and its idle/progress/deadline windows, transport retries and
// auth refresh, prompt-progress plumbing, the multi-choice (n>1) completion
// proof, Timings parsing, provider-reported cost attachment, cache telemetry,
// and the Raw transcript reconstruction — none of them are re-implemented here.
type StreamTranslator struct {
	id          string
	model       string
	inputTokens int
	stopBuf     *StopHoldbackBuffer

	started  bool
	done     bool
	finished bool // Finish succeeded once; Finish is idempotent after success

	// The emitText/closeText index bookkeeping, as the gateway's live Messages
	// arm keeps it: outIdx is the next client-facing content-block index,
	// textOpen/textIdx track the single streamed text block.
	outIdx   int
	textOpen bool
	textIdx  int

	content    strings.Builder
	reasoning  strings.Builder
	toolAcc    map[int]*streamToolCallAcc
	usage      *agent.Usage
	chunkModel string
	finish     string
	selected   bool
	choiceErr  error

	pending []Event
}

// NewStreamTranslator builds a translator for one streamed turn. id and model
// are echoed in message_start verbatim; inputTokens is the pre-completion estimate
// message_start reports (routers compute it with EstimateInputTokens);
// stopSequences feed the holdback buffer. The translator is safe for the single-
// goroutine Feed loop a router runs; it is NOT for concurrent use.
func NewStreamTranslator(id, model string, inputTokens int, stopSequences []string) *StreamTranslator {
	return &StreamTranslator{
		id:          id,
		model:       model,
		inputTokens: inputTokens,
		stopBuf:     NewStopHoldbackBuffer(stopSequences),
		toolAcc:     map[int]*streamToolCallAcc{},
		textIdx:     -1,
	}
}

// Start emits the message_start event (idempotent; Feed auto-starts).
func (t *StreamTranslator) Start() []Event {
	t.start()
	return t.drain()
}

// start is the internal auto-start: it buffers message_start without draining,
// so the caller's Feed/Finish returns the whole turn's events in one slice.
func (t *StreamTranslator) start() {
	if t.started {
		return
	}
	t.started = true
	t.emit("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": t.id, "type": "message", "role": "assistant", "model": t.model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": t.inputTokens, "output_tokens": 0},
		},
	})
}

// Feed consumes one upstream data payload; it returns the Anthropic events to
// forward and an error. An error ends the turn: ErrStreamError for an in-band
// error frame (returned at the Feed that saw it), or the latched invalid-choice-
// index error (surfaced at Finish). Heartbeat and keep-alive payloads are
// tolerated and produce no events.
func (t *StreamTranslator) Feed(data string) ([]Event, error) {
	if t.done {
		return nil, nil
	}
	trimmed := strings.TrimSpace(data)
	if trimmed == "" {
		return nil, nil
	}
	if trimmed == "[DONE]" {
		t.done = true
		return t.drain(), nil
	}
	var chunk openAIStreamChunk
	if err := json.Unmarshal([]byte(trimmed), &chunk); err != nil {
		// Keep heartbeat tolerance, but a valid choice envelope with an invalid
		// index must not disappear and let a later DONE certify partial output.
		var envelope struct {
			Choices []struct {
				Index json.RawMessage `json:"index"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(trimmed), &envelope) == nil {
			for _, ch := range envelope.Choices {
				var index *int
				if len(ch.Index) > 0 && json.Unmarshal(ch.Index, &index) != nil && t.choiceErr == nil {
					t.choiceErr = fmt.Errorf("anthropicbridge: %w: invalid choice index", ErrIncompleteStream)
				}
			}
		}
		return t.drain(), nil // tolerate a keep-alive or non-JSON heartbeat line
	}
	if serr := parseStreamErrorFrame(chunk.Error); serr != nil {
		return nil, serr
	}
	// A Completion represents choice zero, never a concatenation of alternatives.
	// Validate the entire frame before releasing any selected fragment.
	seen := make(map[int]bool, len(chunk.Choices))
	for _, ch := range chunk.Choices {
		i := 0 // Legacy singleton chunks may omit the index.
		if ch.Index != nil {
			i = *ch.Index
		}
		if i < 0 || seen[i] || (ch.Index == nil && len(chunk.Choices) > 1) {
			if t.choiceErr == nil {
				t.choiceErr = fmt.Errorf("anthropicbridge: %w: invalid or ambiguous choice index", ErrIncompleteStream)
			}
		}
		seen[i] = true
	}
	if t.choiceErr != nil {
		return t.drain(), nil
	}
	if chunk.Model != "" {
		t.chunkModel = chunk.Model
	}
	if chunk.Usage != nil {
		t.usage = chunk.Usage
	}
	for _, ch := range chunk.Choices {
		if ch.Index != nil && *ch.Index != 0 {
			continue
		}
		t.selected = true
		if ch.FinishReason != "" {
			t.finish = ch.FinishReason
		}
		if ch.Delta.Content != "" {
			t.content.WriteString(ch.Delta.Content)
			if safe := t.stopBuf.Append(ch.Delta.Content); safe != "" {
				t.start()
				if !t.textOpen {
					t.textIdx = t.outIdx
					t.outIdx++
					t.textOpen = true
					t.emit("content_block_start", map[string]any{
						"type": "content_block_start", "index": t.textIdx,
						"content_block": map[string]any{"type": "text", "text": ""},
					})
				}
				t.emit("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": t.textIdx,
					"delta": map[string]any{"type": "text_delta", "text": safe},
				})
			}
		}
		if ch.Delta.ReasoningContent != "" {
			t.reasoning.WriteString(ch.Delta.ReasoningContent)
		}
		for _, tcd := range ch.Delta.ToolCalls {
			acc := t.toolAcc[tcd.Index]
			if acc == nil {
				acc = &streamToolCallAcc{callType: "function"}
				t.toolAcc[tcd.Index] = acc
			}
			if tcd.ID != "" {
				acc.id = tcd.ID
			}
			if tcd.Type != "" {
				acc.callType = tcd.Type
			}
			if tcd.Function.Name != "" {
				acc.name = tcd.Function.Name
			}
			acc.arguments.WriteString(tcd.Function.Arguments)
		}
	}
	return t.drain(), nil
}

// Finish closes the stream: closes any open text block, emits the held tool_use
// blocks, then the terminal message_delta (real usage + stop_reason) and
// message_stop, and returns the fully accumulated canonical Completion (with
// NormalizeStreamedToolCalls applied, so an upstream that omits tool_calls[].id
// still yields answerable call_fak_N ids). Idempotent after success. Errors:
// ErrIncompleteStream (no [DONE] and no finish reason, or no choice ever
// selected), ErrStreamError (in-band error frame, returned at the Feed that saw
// it), or the latched invalid-choice-index error (negative, duplicate, or
// ambiguous missing index — the planner's wording, "invalid or ambiguous choice
// index"; unlike the planner, an out-of-range index is tolerated and skipped,
// the right tolerance for a router fanning out to heterogeneous upstreams).
func (t *StreamTranslator) Finish() ([]Event, *Completion, error) {
	if t.finished {
		return nil, nil, nil
	}
	if t.choiceErr != nil {
		return nil, nil, t.choiceErr
	}
	if !t.selected || (!t.done && t.finish == "") {
		return nil, nil, fmt.Errorf("anthropicbridge: %w", ErrIncompleteStream)
	}
	t.start()
	t.closeText()
	calls := make([]ToolCall, 0, len(t.toolAcc))
	for _, idx := range sortedIndices(t.toolAcc) {
		calls = append(calls, t.toolAcc[idx].toolCall())
	}
	comp := &Completion{
		Message: Message{
			Role:             agent.RoleAssistant,
			Content:          t.content.String(),
			ReasoningContent: t.reasoning.String(),
			ToolCalls:        calls,
		},
		FinishReason: t.finish,
	}
	comp = agent.NormalizeStreamedToolCalls(comp)
	if t.usage != nil {
		comp.Usage = *t.usage
	}
	if t.chunkModel != "" {
		comp.Model = t.chunkModel
	}
	StreamToolUseBlocks(t.emit, &t.outIdx, calls)
	usage := MessagesUsageFromCanonical(comp.Usage)
	stop := StopReason(comp.FinishReason, len(calls) > 0)
	StreamTerminal(t.emit, stop, usage)
	t.finished = true
	return t.drain(), comp, nil
}

// emit marshals one event payload and buffers it; Feed/Start/Finish return the
// drained pending slice. The shapes are fixed maps, so a marshal error is
// unreachable in practice and drops the event rather than corrupting the stream.
func (t *StreamTranslator) emit(event string, data any) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	t.pending = append(t.pending, Event{Name: event, Data: encoded})
}

// drain returns the pending events and resets the buffer.
func (t *StreamTranslator) drain() []Event {
	if len(t.pending) == 0 {
		return nil
	}
	out := t.pending
	t.pending = nil
	return out
}

// closeText flushes the holdback tail and closes the open text block — the
// gateway's closeText closure, verbatim.
func (t *StreamTranslator) closeText() {
	if t.textOpen {
		if tail := t.stopBuf.Flush(); tail != "" {
			t.emit("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": t.textIdx,
				"delta": map[string]any{"type": "text_delta", "text": tail},
			})
		}
		t.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": t.textIdx})
		t.textOpen = false
		t.textIdx = -1
	}
}

// StreamTerminal emits the terminal message_delta (stop_reason + the REAL turn
// usage — the message_start figure was a pre-completion estimate; cache counters
// included only when nonzero) and message_stop.
func StreamTerminal(send func(event string, data any), stop string, usage MessagesUsage) {
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

// StreamBlocks emits one finished turn's content blocks as Anthropic SSE:
// a text block renders start+one text_delta+stop; a tool_use block renders
// start+one whole-argument input_json_delta+stop.
func StreamBlocks(send func(event string, data any), blocks []Block, stop string, usage MessagesUsage) {
	for i, blk := range blocks {
		switch blk.Type {
		case "tool_use":
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": map[string]any{}},
			})
			// The whole (already-validated) argument object as one input_json_delta.
			send("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(blk.Input)},
			})
		default: // text
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			send("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "text_delta", "text": blk.Text},
			})
		}
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}

	// The terminal message_delta carries the REAL usage from the finished turn — the
	// message_start figure was a pre-completion estimate. Report the upstream's true
	// input_tokens (the uncached remainder) plus the cache counters, so a passthrough
	// turn's cache hit reaches the client's accounting. Counters are omitted when zero
	// (a local-model turn streams the same shape as before).
	StreamTerminal(send, stop, usage)
}

// StreamToolUseBlocks streams tool calls as tool_use content blocks
// (content_block_start/input_json_delta/content_block_stop triples), advancing
// *nextIndex for each so client-facing index numbering stays contiguous.
func StreamToolUseBlocks(send func(event string, data any), nextIndex *int, calls []ToolCall) {
	for _, blk := range ResponseBlocks(Message{Role: agent.RoleAssistant, ToolCalls: calls}) {
		oi := *nextIndex
		*nextIndex++
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
}

// openAIStreamChunk is one OpenAI-compatible `chat.completion.chunk` SSE event. Each
// `data:` line carries one of these; the terminal `data: [DONE]` carries none. The
// delta fields are all optional and additive across chunks: content fragments
// concatenate into the final text, and tool-call fragments accumulate by index (id +
// name on first sight, arguments concatenated). Usage rides the final chunk only when
// the request asked for it (stream_options.include_usage).
type openAIStreamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Index *int `json:"index"`
		Delta struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage          *agent.Usage    `json:"usage"`
	Timings        json.RawMessage `json:"timings,omitempty"`
	Error          json.RawMessage `json:"error,omitempty"`
	PromptProgress json.RawMessage `json:"prompt_progress,omitempty"`
}

// streamToolCallAcc accumulates one streamed tool call's fragments by delta index.
type streamToolCallAcc struct {
	id        string
	callType  string
	name      string
	arguments strings.Builder
}

func (a *streamToolCallAcc) toolCall() ToolCall {
	return ToolCall{
		ID:   a.id,
		Type: a.callType,
		Function: agent.Func{
			Name:      a.name,
			Arguments: a.arguments.String(),
		},
	}
}

func sortedIndices[T any](m map[int]T) []int {
	idx := make([]int, 0, len(m))
	for k := range m {
		idx = append(idx, k)
	}
	sort.Ints(idx)
	return idx
}

// parseStreamErrorFrame decodes a chunk's `error` member, accepting an object (with a
// string or numeric code) or a bare string. It returns nil when the member is absent or null.
func parseStreamErrorFrame(raw json.RawMessage) *UpstreamStreamError {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}
	var msg string
	if json.Unmarshal(raw, &msg) == nil {
		return &UpstreamStreamError{Message: msg}
	}
	var obj struct {
		Type    string          `json:"type"`
		Code    json.RawMessage `json:"code"`
		Message json.RawMessage `json:"message"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return &UpstreamStreamError{Message: truncate(raw, 400)}
	}
	return &UpstreamStreamError{Type: obj.Type, Code: jsonScalarString(obj.Code), Message: jsonScalarString(obj.Message)}
}

func jsonScalarString(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return truncate(raw, 400)
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "..."
	}
	return string(b)
}

package gateway

// messages_stream_warmcontinue.go — warm-continue a live turn across a worker death by
// replaying the already-delivered output as context (#3353, part of #3352). When the true-
// streaming Anthropic passthrough (messages_stream_passthrough.go) loses its upstream
// MID-TURN — after client SSE bytes have already flowed — the default recovery ends the
// caller's stream with a terminal error frame and cold-restarts the whole session, throwing
// away the partial assistant output. This file instead continues the SAME logical turn on a
// fresh worker.
//
// Clean-room of dynamo's RetryManager (ai-dynamo/dynamo lib/llm/src/migration.rs@be36b52396):
// its track_response() pushes already-delivered tokens back onto the request and sets
// max_tokens = max_tokens.saturating_sub(delivered); its next() recreates the stream on a
// migratable error and hides the failure from the consumer. fak has no per-token ids on the
// SSE wire, so it replays the delivered assistant TEXT as an assistant PREFILL turn (the
// model continues from it without re-emitting it) and decrements the budget by an estimate.
//
// Gated OFF by default (FAK_WARM_CONTINUE) — a gen/next foundation, dogfood-first. Scope is
// deliberately narrow and honest (canWarmContinue): text-only turns with no held tool_use
// block and no thinking block, since a tool batch would double-emit on replay and Anthropic
// rejects an assistant prefill when extended thinking is enabled.

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// warmContinueMaxAttempts bounds the resume loop so a worker pool that keeps dying mid-turn
// degrades to the terminal-error path rather than looping unboundedly. Each attempt replays
// the now-longer prefix and decrements the budget again, exactly as dynamo's next() loop does.
const warmContinueMaxAttempts = 3

var (
	errWarmContinueBuild     = errors.New("gateway: warm-continue could not build the continuation request")
	errWarmContinueExhausted = errors.New("gateway: warm-continue exhausted its attempts")
	errWarmContinueStoodDown = errors.New("gateway: warm-continue stood down — the turn is no longer replayable")
)

// warmContinueEnabled reports whether the #3353 warm-continue path is armed. OFF by default;
// FAK_WARM_CONTINUE=1|true|yes|on opts a deploy or dogfood session in.
func warmContinueEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_WARM_CONTINUE"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// canWarmContinue reports whether this mid-stream death is safely resumable by replaying the
// delivered text as a prefill turn: the gate armed, a migratable (non-status) error, and a turn
// whose shape is still replayable.
func (p *anthropicPassthrough) canWarmContinue(err error) bool {
	return warmContinueEnabled() && warmContinuableErr(err) && p.warmContinueShapeOK()
}

// warmContinueShapeOK reports whether the turn AS IT STANDS NOW is replayable as an assistant
// prefill: some relayed text to replay, no held tool_use block (which would re-emit on replay)
// and no thinking, checked BOTH ways — no thinking block relayed, and extended thinking not
// enabled on the request in the first place.
//
// This is deliberately re-read before EVERY attempt, not just the first (warmContinue's loop),
// because a continuation can change the answer: a resumed stream that itself dies after opening
// a tool_use block leaves that call sitting in toolOrder, and replaying past it would let a
// re-emitted call join the still-held one so flushHeldTools emits the SAME tool_use twice — the
// client would execute the side effect TWICE. Standing down to the terminal-error path costs the
// turn; a duplicated tool call costs whatever the tool did.
func (p *anthropicPassthrough) warmContinueShapeOK() bool {
	return strings.TrimSpace(p.asstText.String()) != "" &&
		len(p.toolOrder) == 0 &&
		!p.sawThinking &&
		!requestEnablesThinking(p.req.Raw)
}

// requestEnablesThinking reports whether the INBOUND body turns extended thinking on.
// Anthropic refuses an assistant prefill outright on such a request, so the replay would buy a
// guaranteed 400 — which warmContinuableErr then (correctly) declines to retry, leaving the
// client with the same terminal error one wasted upstream call later. This is the structural
// half of the thinking guard: p.sawThinking can only react once a thinking block has already
// been relayed, and a thinking-enabled turn whose first block happens to be text would slip
// past it.
func requestEnablesThinking(raw []byte) bool {
	var m struct {
		Thinking struct {
			Type string `json:"type"`
		} `json:"thinking"`
	}
	if json.Unmarshal(raw, &m) != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(m.Thinking.Type), "enabled")
}

// warmContinuableErr reports whether err is a MIGRATABLE mid-stream failure — a transport
// reset or an idle stall — rather than a definite HTTP status. A status error (e.g. a 400 the
// prefill itself provoked, or an exhausted 429) will not be fixed by replaying on another
// worker, so it is surfaced instead of retried.
func warmContinuableErr(err error) bool {
	if err == nil {
		return false
	}
	var se *agent.UpstreamStatusError
	return !errors.As(err, &se)
}

// warmContinue re-issues the turn on a fresh worker, replaying the delivered assistant text as
// a prefill turn with the budget decremented, and resumes the SAME open client stream. It
// returns nil once a continuation runs to a clean terminal (message_stop), or the surviving
// error when the death is not migratable or the attempt budget is spent — in which case the
// caller emits the terminal error frame exactly as before.
func (p *anthropicPassthrough) warmContinue(hp *agent.HTTPPlanner, upstreamKey, upstreamBeta string) error {
	p.continuing = true // suppress the continuation's own message_start echo/notes
	for attempt := 0; attempt < warmContinueMaxAttempts; attempt++ {
		// Re-check the shape every time round: the previous attempt may have opened a tool_use
		// or thinking block before it died, which makes a further replay unsafe (see
		// warmContinueShapeOK). Standing down hands the caller back to the terminal-error path.
		if !p.warmContinueShapeOK() {
			return errWarmContinueStoodDown
		}
		prefix := p.asstText.String()
		body, ok := warmContinueBody(p.req.Raw, prefix, estimateAnthropicTokens(prefix))
		if !ok {
			return errWarmContinueBuild
		}
		// Close a dangling open client block (the death interrupted it) so the continuation
		// opens a fresh, well-formed block on the resumed stream.
		p.closeOpenClientBlock()
		err := hp.StreamAnthropicRaw(p.r.Context(), body, upstreamKey, upstreamBeta, p.onEvent)
		if err == nil {
			return nil // resumed to a clean terminal — one unbroken turn on the client wire
		}
		if !warmContinuableErr(err) {
			return err // a definite status won't be fixed by another replay
		}
		p.s.logf("gateway: warm-continue attempt %d died again mid-stream: %v", attempt+1, err)
	}
	return errWarmContinueExhausted
}

// closeOpenClientBlock emits a content_block_stop for a relayed block left open by a mid-stream
// death, so the resumed stream is well-formed. A no-op when nothing is open.
func (p *anthropicPassthrough) closeOpenClientBlock() {
	if p.openClientBlock < 0 {
		return
	}
	p.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": p.openClientBlock})
	p.openClientBlock = -1
}

// warmContinueBody builds the continuation request from the original inbound body: it lands
// the delivered assistant text as the trailing assistant PREFILL message (the model continues
// it without re-emitting it) and decrements max_tokens by the delivered estimate (saturating at
// 1, mirroring dynamo's max_tokens.saturating_sub). The body is re-marshalled — the dead
// worker's cache prefix is already lost, and the continuation goes to a fresh worker, so
// byte-identity of the cached prefix is not a goal here. Returns ok=false if the body is not a
// JSON object with a messages array, or if the delivered prefix is not a legal prefill.
func warmContinueBody(raw []byte, prefix string, delivered int) ([]byte, bool) {
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return nil, false
	}
	msgsRaw, ok := m["messages"]
	if !ok {
		return nil, false
	}
	var msgs []json.RawMessage
	if json.Unmarshal(msgsRaw, &msgs) != nil {
		return nil, false
	}
	msgs, ok = warmContinuePrefill(msgs, prefix)
	if !ok {
		return nil, false
	}
	nb, err := json.Marshal(msgs)
	if err != nil {
		return nil, false
	}
	m["messages"] = nb
	if mtRaw, ok := m["max_tokens"]; ok {
		var mt int
		if json.Unmarshal(mtRaw, &mt) == nil && mt > 0 {
			rem := mt - delivered
			if rem < 1 {
				rem = 1
			}
			m["max_tokens"] = json.RawMessage(strconv.Itoa(rem))
		}
	}
	out, err := json.Marshal(m)
	if err != nil {
		return nil, false
	}
	return out, true
}

// warmContinuePrefill lands the delivered text as the message array's trailing assistant turn.
// Anthropic holds a final assistant (prefill) turn to two hard rules, and a mid-stream death
// walks into both of them constantly:
//
//   - its content may NOT end in whitespace ("final assistant content cannot end with trailing
//     whitespace") — yet a stream cut mid-markdown or mid-code ends on a newline more often
//     than not, so the naive replay of exactly-what-was-delivered is a coin-flip 400;
//   - it may not FOLLOW another assistant turn — and a caller is free to send its own prefill
//     as the last inbound message, which is precisely the turn the model was continuing when
//     the worker died, so appending a second assistant message stacks two same-role turns.
//
// So the text is right-trimmed for the wire and MERGED onto an existing trailing assistant turn
// instead of stacked after it. The trim is a deliberate small asymmetry: the client already saw
// the trailing newline, and the model may re-emit it, so the resumed text can gain a duplicate
// space — cosmetic, and strictly better than the turn dying outright. Returns ok=false when
// nothing survives the trim (an all-whitespace prefix is not a resumable prefill) or the
// trailing assistant turn has a content shape we will not extend without guessing.
func warmContinuePrefill(msgs []json.RawMessage, prefix string) ([]json.RawMessage, bool) {
	prefix = strings.TrimRight(prefix, " \t\r\n\v\f")
	if prefix == "" {
		return nil, false
	}
	if n := len(msgs); n > 0 {
		var probe struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(msgs[n-1], &probe) == nil && probe.Role == "assistant" {
			merged, ok := growAssistantText(probe.Content, prefix)
			if !ok {
				return nil, false
			}
			var last map[string]json.RawMessage
			if json.Unmarshal(msgs[n-1], &last) != nil {
				return nil, false
			}
			last["content"] = merged
			nb, err := json.Marshal(last)
			if err != nil {
				return nil, false
			}
			out := make([]json.RawMessage, n)
			copy(out, msgs)
			out[n-1] = nb
			return out, true
		}
	}
	am, err := json.Marshal(map[string]any{"role": "assistant", "content": prefix})
	if err != nil {
		return nil, false
	}
	return append(msgs, json.RawMessage(am)), true
}

// growAssistantText concatenates text onto an assistant message's content, preserving whichever
// wire shape it arrived in: a bare string grows in place, a block array grows its LAST text
// block, and an array ending in a non-text block gains a fresh text block. Returns ok=false for
// a content value that is neither a string nor an array.
func growAssistantText(content json.RawMessage, text string) (json.RawMessage, bool) {
	var s string
	if json.Unmarshal(content, &s) == nil {
		nb, err := json.Marshal(s + text)
		return nb, err == nil
	}
	var blocks []json.RawMessage
	if json.Unmarshal(content, &blocks) != nil {
		return nil, false
	}
	if n := len(blocks); n > 0 {
		var last map[string]json.RawMessage
		if json.Unmarshal(blocks[n-1], &last) == nil {
			var typ, prior string
			_ = json.Unmarshal(last["type"], &typ)
			if typ == "text" && json.Unmarshal(last["text"], &prior) == nil {
				tb, err := json.Marshal(prior + text)
				if err != nil {
					return nil, false
				}
				last["text"] = tb
				nb, err := json.Marshal(last)
				if err != nil {
					return nil, false
				}
				blocks[n-1] = nb
				out, err := json.Marshal(blocks)
				return out, err == nil
			}
		}
	}
	tb, err := json.Marshal(map[string]any{"type": "text", "text": text})
	if err != nil {
		return nil, false
	}
	out, err := json.Marshal(append(blocks, json.RawMessage(tb)))
	return out, err == nil
}

// estimateAnthropicTokens approximates how much of the token budget the delivered text already
// spent. fak has no per-token ids on the SSE wire (dynamo counts exact ids), so it estimates at
// ~4 chars/token — an over-decrement only shortens the continuation's cap, never over-runs the
// original budget, so the safe direction to err.
func estimateAnthropicTokens(s string) int {
	n := len([]rune(s))
	if n == 0 {
		return 0
	}
	return (n + 3) / 4
}

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
// delivered text as a prefill turn. It requires the gate armed, some relayed assistant text to
// replay, a migratable (non-status) error, and a turn that is text-only so far: no held
// tool_use block (which would re-emit on replay) and no thinking block (not prefill-replayable).
func (p *anthropicPassthrough) canWarmContinue(err error) bool {
	return warmContinueEnabled() &&
		warmContinuableErr(err) &&
		strings.TrimSpace(p.asstText.String()) != "" &&
		len(p.toolOrder) == 0 &&
		!p.sawThinking
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

// warmContinueBody builds the continuation request from the original inbound body: it appends
// the delivered assistant text as a trailing assistant PREFILL message (the model continues it
// without re-emitting it) and decrements max_tokens by the delivered estimate (saturating at 1,
// mirroring dynamo's max_tokens.saturating_sub). The body is re-marshalled — the dead worker's
// cache prefix is already lost, and the continuation goes to a fresh worker, so byte-identity of
// the cached prefix is not a goal here. Returns ok=false if the body is not a JSON object with a
// messages array.
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
	am, err := json.Marshal(map[string]any{"role": "assistant", "content": prefix})
	if err != nil {
		return nil, false
	}
	msgs = append(msgs, json.RawMessage(am))
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

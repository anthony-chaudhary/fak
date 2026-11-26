package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Local model servers have small context windows, so a single unbounded tool
// result can exhaust one; the ctxmmu paging thresholds are sized for hosted
// models, so a local endpoint gets its own per-result bound here.

// localToolResultBudgetBytes bounds one tool result sent to a local endpoint.
const localToolResultBudgetBytes = 8 << 10

// localCompactedToolResultBytes is the tighter per-result bound used on the one
// compacted retry after the server reports the prompt does not fit.
const localCompactedToolResultBytes = 2 << 10

// localGrepMatchCap bounds the match rows a Grep result carries to a local endpoint.
const localGrepMatchCap = 50

// localCompactedKeepRecent is how many of the newest tool results survive the
// compacted retry untouched; older ones become stubs.
const localCompactedKeepRecent = 2

// localReplyReserveTokens is headroom kept free for the model's reply and the
// chat template on top of the planner's configured output ceiling.
const localReplyReserveTokens = 256

// localBytesPerToken is the token estimate used when the served context size is
// known: prompt tokens ~= prompt bytes / 3.5. It is a heuristic, not a tokenizer,
// so the fit is advisory and the server's own refusal remains authoritative.
const localBytesPerToken = 3.5

// EnvAgentContextTokens names the served context size in tokens. Unset (or
// non-positive) means unknown: only the per-result bounds apply.
const EnvAgentContextTokens = "FAK_AGENT_CONTEXT_TOKENS"

// agentContextTokens returns the operator-declared served context size, or 0.
func agentContextTokens() int {
	n, err := strconv.Atoi(strings.TrimSpace(os.Getenv(EnvAgentContextTokens)))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

type localCompactKey struct{}

// withLocalCompact marks ctx for the single compacted retry.
func withLocalCompact(ctx context.Context) context.Context {
	return context.WithValue(ctx, localCompactKey{}, true)
}

func localCompactRequested(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	v, _ := ctx.Value(localCompactKey{}).(bool)
	return v
}

// localContextFit is the request-time fit policy for a local endpoint.
type localContextFit struct {
	perResultBytes int // per-tool-result byte bound
	contextTokens  int // served context size; 0 = unknown (no whole-prompt fit)
	reserveTokens  int // headroom for the reply and template
	keepRecent     int // newest tool results never stubbed by the whole-prompt fit
}

// localFitReport counts what the fit changed, for the caller's logs and tests.
type localFitReport struct {
	Bounded    int // tool results truncated to the per-result bound
	GrepCapped int // Grep results whose match rows were capped
	Stubbed    int // tool results replaced by a stub to fit the window
	EstTokens  int // estimated prompt tokens after the fit
}

// fitLocalTranscript returns a copy of messages with every tool result bounded
// and, when the served context size is known, the oldest tool results stubbed
// until the estimated prompt fits. Messages are never reordered or dropped.
func fitLocalTranscript(messages []Message, fit localContextFit) ([]Message, localFitReport) {
	out := append([]Message(nil), messages...)
	var rep localFitReport
	for i := range out {
		if out[i].Role != RoleTool {
			continue
		}
		content, kind := boundLocalToolResult(out[i].Name, out[i].Content, fit.perResultBytes)
		switch kind {
		case "grep":
			rep.GrepCapped++
		case "bytes":
			rep.Bounded++
		}
		out[i].Content = content
	}
	est := estimateTranscriptTokens(out)
	if fit.contextTokens > 0 {
		budget := fit.contextTokens - fit.reserveTokens
		lastTool := -1
		for i := range out {
			if out[i].Role == RoleTool {
				lastTool = i
			}
		}
		keep := fit.keepRecent
		if keep < 1 {
			keep = 1
		}
		// Stub the oldest results first, never the newest keep-many.
		newestKept := map[int]bool{}
		seen := 0
		for i := len(out) - 1; i >= 0 && seen < keep; i-- {
			if out[i].Role == RoleTool {
				newestKept[i] = true
				seen++
			}
		}
		for i := 0; i < len(out) && est > budget; i++ {
			if out[i].Role != RoleTool || newestKept[i] || i == lastTool || isLocalStub(out[i].Content) {
				continue
			}
			before := tokensOf(out[i])
			out[i].Content = localStub(out[i].Name, len(out[i].Content))
			est += tokensOf(out[i]) - before
			rep.Stubbed++
		}
		if rep.Stubbed > 0 {
			est = estimateTranscriptTokens(out)
		}
	}
	rep.EstTokens = est
	return out, rep
}

// boundLocalToolResult returns content no longer than limit bytes. Grep results
// are first cut to localGrepMatchCap rows; any remaining excess is cut from the
// tail with a notice that says how much was dropped and how to narrow the query.
// kind is "grep", "bytes", or "" when the content was left alone.
func boundLocalToolResult(name, content string, limit int) (string, string) {
	kind := ""
	if isGrepTool(name) {
		if capped, dropped := capGrepMatches(content); dropped > 0 {
			content, kind = capped, "grep"
		}
	}
	if len(content) <= limit {
		return content, kind
	}
	// The notice carries the dropped count, whose digits are at most those of the
	// full length, so reserving for the full-length notice keeps the result bounded.
	keep := limit - len(localTruncNotice(len(content)))
	if keep < 0 {
		keep = 0
	}
	for keep > 0 && !utf8.RuneStart(content[keep]) {
		keep--
	}
	return content[:keep] + localTruncNotice(len(content)-keep), "bytes"
}

func localTruncNotice(dropped int) string {
	return fmt.Sprintf("\n[fak: tool result truncated for the local context window; %d bytes omitted. Narrow the request (path, glob, pattern, offset, or max_matches) and retry.]", dropped)
}

// capGrepMatches keeps the first localGrepMatchCap rows of a Grep result's
// "matches" array and records how many were dropped. A result that is not a JSON
// object with a longer matches array is returned unchanged with dropped 0.
func capGrepMatches(content string) (string, int) {
	var doc map[string]json.RawMessage
	if json.Unmarshal([]byte(content), &doc) != nil {
		return content, 0
	}
	raw, ok := doc["matches"]
	if !ok {
		return content, 0
	}
	var rows []json.RawMessage
	if json.Unmarshal(raw, &rows) != nil || len(rows) <= localGrepMatchCap {
		return content, 0
	}
	kept, err := json.Marshal(rows[:localGrepMatchCap])
	if err != nil {
		return content, 0
	}
	dropped := len(rows) - localGrepMatchCap
	doc["matches"] = kept
	doc["matches_dropped"] = json.RawMessage(strconv.Itoa(dropped))
	out, err := json.Marshal(doc)
	if err != nil {
		return content, 0
	}
	return string(out), dropped
}

func isGrepTool(name string) bool {
	return strings.EqualFold(name, "grep") || strings.HasSuffix(strings.ToLower(name), "_grep")
}

const localStubPrefix = "[fak: earlier tool result elided to fit the local context window"

func localStub(name string, size int) string {
	if name == "" {
		name = "tool"
	}
	return fmt.Sprintf("%s; %s returned %d bytes. Rerun it with narrower arguments if you still need it.]", localStubPrefix, name, size)
}

func isLocalStub(content string) bool {
	return strings.HasPrefix(content, localStubPrefix)
}

// tokensOf estimates one message's prompt tokens from its text and tool-call arguments.
func tokensOf(m Message) int {
	n := len(m.Content)
	for _, tc := range m.ToolCalls {
		n += len(tc.Function.Name) + len(tc.Function.Arguments)
	}
	return int(float64(n) / localBytesPerToken)
}

func estimateTranscriptTokens(messages []Message) int {
	total := 0
	for _, m := range messages {
		total += tokensOf(m)
	}
	return total
}

// estimateToolDefTokens estimates the prompt tokens the marshalled tool schemas
// add to a request. An unmarshallable definition counts as zero: the request
// itself would fail to serialize, and that error is reported elsewhere.
func estimateToolDefTokens(tools []ToolDef) int {
	if len(tools) == 0 {
		return 0
	}
	raw, err := json.Marshal(tools)
	if err != nil {
		return 0
	}
	return int(float64(len(raw)) / localBytesPerToken)
}

// fitLocalRequest applies the local-endpoint fit to the transcript about to be
// serialized. The compacted retry tightens the per-result bound and keeps only
// the newest tool results whole. Tool definitions travel in the same prompt, so
// their estimated size is reserved out of the window before the transcript fits.
func (p *HTTPPlanner) fitLocalRequest(ctx context.Context, messages []Message, tools []ToolDef) []Message {
	fit := localContextFit{
		perResultBytes: localToolResultBudgetBytes,
		contextTokens:  agentContextTokens(),
		reserveTokens:  p.MaxTokens + localReplyReserveTokens + estimateToolDefTokens(tools),
		keepRecent:     localCompactedKeepRecent,
	}
	if localCompactRequested(ctx) {
		fit.perResultBytes = localCompactedToolResultBytes
	}
	out, _ := fitLocalTranscript(messages, fit)
	return out
}

// localFitActive reports whether the local-endpoint transcript fit and its
// compacted retry apply to this planner's requests.
func (p *HTTPPlanner) localFitActive() bool {
	return p != nil && !p.LocalFitDisabled && IsLocalEndpoint(p.BaseURL)
}

// localContextRetry reports whether a failed call should be compacted and resent
// once: a local endpoint refused the prompt as too large, and this call has not
// already been the compacted retry.
func (p *HTTPPlanner) localContextRetry(ctx context.Context, err error) bool {
	if err == nil || p == nil || ctx == nil || ctx.Err() != nil || localCompactRequested(ctx) {
		return false
	}
	return errors.Is(err, ErrContextOverflow) && p.localFitActive()
}

// localContextRefusal recognizes the 502 a local serve returns when the prompt
// exceeds its context. The serve logs the cause but the client only sees the
// generic body, so the local-only match is what makes the refusal compactable.
func localContextRefusal(p *HTTPPlanner, status int, raw []byte) bool {
	if status != http.StatusBadGateway || !p.localFitActive() {
		return false
	}
	return bodyContainsAny(raw, "upstream model error", "leaves no room in")
}

package eveimport

// parse.go — the two evidence-wire parsers. Both are pure: raw saved bytes in, a Run
// out. Neither opens files, talks to a network, or reads a clock; the caller owns all
// I/O and hands in the bytes plus the path recorded as evidence. Malformed input
// degrades to diagnostics (an honest partial observation), never a panic and never a
// fabricated success.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// ndjsonEvent is the fixture-modeled Eve session-stream event: one JSON object per
// line, discriminated by Type, carrying only framework-owned workflow tags. Extra
// fields on a line (timestamps, request ids) are ignored — the importer reads tags,
// not prose.
type ndjsonEvent struct {
	Type             string `json:"type"`
	SessionID        string `json:"session_id"`
	ParentSessionID  string `json:"parent_session_id"`
	ModelID          string `json:"model_id"`
	Turn             int    `json:"turn"`
	Role             string `json:"role"`
	Body             string `json:"body"`
	Tool             string `json:"tool"`
	Reason           string `json:"reason"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	CacheReadTokens  int    `json:"cache_read_tokens"`
}

// ImportNDJSON reconstructs an execution Run from a saved Eve newline-delimited JSON stream.
// sourcePath is recorded verbatim as the evidence path (never opened). The modeled event
// vocabulary: session.start, turn.start, message, reasoning, tool.call, usage, failure,
// session.end. Unknown event types and unparseable lines become diagnostics; turn-scoped
// events for a session no session.start declared are ORPHAN_EVENT and are dropped rather
// than guessed into a tree.
//
// Invariant: parsing never fabricates success; malformed records degrade to partial observations or INDETERMINATE without raising panics.
// Postcondition: returns a deterministic Run containing session lineage and usage reconstructed strictly from framework tags.
func ImportNDJSON(sourcePath string, data []byte, opt Options) Run {
	b := newBuilder(Source{Kind: "eve-ndjson", Path: sourcePath}, opt)
	for n, raw := range bytes.Split(data, []byte("\n")) {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}
		var ev ndjsonEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			b.diag(DiagBadLine, "", 0, fmt.Sprintf("line %d: not valid JSON", n+1))
			continue
		}
		switch ev.Type {
		case "session.start":
			b.startSession(ev.SessionID, ev.ParentSessionID, ev.ModelID)
		case "session.end":
			// Accepted; the reconstruction does not need an explicit close.
		case "turn.start", "message", "reasoning", "tool.call", "usage", "failure":
			if !b.hasSession(ev.SessionID) {
				b.diag(DiagOrphanEvent, ev.SessionID, ev.Turn,
					fmt.Sprintf("line %d: %s event for undeclared session", n+1, ev.Type))
				continue
			}
			switch ev.Type {
			case "turn.start":
				b.turn(ev.SessionID, ev.Turn)
			case "message":
				b.addBody(ev.SessionID, ev.Turn, "message", ev.Role, ev.Body)
			case "reasoning":
				b.addBody(ev.SessionID, ev.Turn, "reasoning", "", ev.Body)
			case "tool.call":
				b.addToolCall(ev.SessionID, ev.Turn, ev.Tool)
			case "usage":
				b.addUsage(ev.SessionID, ev.Turn, Usage{
					PromptTokens:     ev.PromptTokens,
					CompletionTokens: ev.CompletionTokens,
					CacheReadTokens:  ev.CacheReadTokens,
				})
			case "failure":
				b.addFailure(ev.SessionID, ev.Turn, ev.Reason)
			}
		default:
			b.diag(DiagUnknownEvent, ev.SessionID, ev.Turn,
				fmt.Sprintf("line %d: unmodeled event type %q", n+1, ev.Type))
		}
	}
	return b.finalize()
}

// otelDoc is the fixture-modeled saved span export: a flat span list with an
// already-flattened attribute map per span (the shape agent/instrumentation.ts's
// exporter writes to disk in the fixtures this package is tested against).
type otelDoc struct {
	Spans []otelSpan `json:"spans"`
}

type otelSpan struct {
	Name         string         `json:"name"`
	SpanID       string         `json:"span_id"`
	ParentSpanID string         `json:"parent_span_id"`
	Status       otelStatus     `json:"status"`
	Attributes   map[string]any `json:"attributes"`
}

type otelStatus struct {
	Code    string `json:"code"` // "OK" | "ERROR" | ""
	Message string `json:"message"`
}

// attrs wraps a span's attribute map with `eve.*` / `$eve.*` prefix folding: both
// spellings of a key are the same tag ("$eve.session.id" == "eve.session.id").
type attrs map[string]any

func (a attrs) lookup(key string) (any, bool) {
	if v, ok := a[key]; ok {
		return v, true
	}
	if v, ok := a["$"+key]; ok {
		return v, true
	}
	return nil, false
}

func (a attrs) str(key string) string {
	if v, ok := a.lookup(key); ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func (a attrs) num(key string) (int, bool) {
	v, ok := a.lookup(key)
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64: // encoding/json's number type
		return int(n), true
	case int:
		return n, true
	}
	return 0, false
}

// ImportOTelSpans reconstructs an execution Run from OpenTelemetry span exports with eve attributes.
// Spans carry `eve.*` / `$eve.*` attributes. Span classification is by attribute, not span name:
// eve.tool.name marks a tool-call span, eve.turn.index (without a tool name) marks a turn span
// carrying usage counters and optional bodies, and a span with neither is session-scoped (model id,
// subagent lineage). A span with status ERROR contributes a failure event. Spans with no
// eve.session.id cannot join and degrade to ORPHAN_SPAN diagnostics.
//
// Precondition: data contains JSON span export bytes; unreadable inputs degrade to DiagBadInput with an INDETERMINATE run outcome.
// Postcondition: returns a deterministic Run tree where sessions and turns reflect framework-owned tags while redacting payload bodies.
func ImportOTelSpans(sourcePath string, data []byte, opt Options) Run {
	b := newBuilder(Source{Kind: "eve-otel-spans", Path: sourcePath}, opt)
	var doc otelDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		b.diag(DiagBadInput, "", 0, "span export is not valid JSON")
		return b.finalize()
	}
	for _, sp := range doc.Spans {
		a := attrs(sp.Attributes)
		sid := a.str("eve.session.id")
		if sid == "" {
			b.diag(DiagOrphanSpan, "", 0,
				fmt.Sprintf("span %q (%s) carries no eve.session.id; it cannot join a session", sp.Name, sp.SpanID))
			continue
		}
		if !b.hasSession(sid) {
			b.startSession(sid, a.str("eve.session.parent_id"), a.str("eve.model.id"))
		} else {
			b.fillSession(sid, a.str("eve.session.parent_id"), a.str("eve.model.id"))
		}

		idx, hasIdx := a.num("eve.turn.index")
		if tool := a.str("eve.tool.name"); tool != "" {
			b.addToolCall(sid, idx, tool)
		} else if hasIdx {
			b.turn(sid, idx) // materialize the turn even if it carries no counters
			var u Usage
			seen := false
			if n, ok := a.num("eve.usage.prompt_tokens"); ok {
				u.PromptTokens, seen = n, true
			}
			if n, ok := a.num("eve.usage.completion_tokens"); ok {
				u.CompletionTokens, seen = n, true
			}
			if n, ok := a.num("eve.usage.cache_read_tokens"); ok {
				u.CacheReadTokens, seen = n, true
			}
			if seen {
				b.addUsage(sid, idx, u)
			}
			if body := a.str("eve.message.body"); body != "" {
				role := a.str("eve.message.role")
				if role == "" {
					role = "assistant"
				}
				b.addBody(sid, idx, "message", role, body)
			}
			if body := a.str("eve.reasoning.body"); body != "" {
				b.addBody(sid, idx, "reasoning", "", body)
			}
		}
		if strings.EqualFold(sp.Status.Code, "ERROR") {
			b.addFailure(sid, idx, sp.Status.Message)
		}
	}
	return b.finalize()
}

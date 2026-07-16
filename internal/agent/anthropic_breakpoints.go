package agent

// anthropic_breakpoints.go — the INBOUND cache_control breakpoint layout recorder (#2786).
//
// WHY THIS EXISTS. The compaction valuation prices a shed token at either the full input
// rate (1.0x) or the provider's cache-read marginal (0.1x), and which one is honest depends
// on a fact the byte-level compaction path never recorded: did the dropped middle sit INSIDE
// a live breakpoint's cached prefix? A cache_control breakpoint at messages[i] caches the
// prefix messages[0..i]; a span dropped from inside that prefix was already being billed as a
// cache_read, so valuing it at 1.0x over-credits compaction 10x (the #2794/#2798 class of
// error). Nothing here prices anything — this file only RECORDS the layout that makes the
// question answerable.
//
// WHY IT IS A DISTINCT DATA PATH (not a widened rangeHasCacheControl). rangeHasCacheControl
// answers one narrow question for the fire gate: "does the range I am about to DROP contain a
// breakpoint?" — a bail predicate over the drop range only. Silently repurposing it to also
// describe the KEPT window would overload a gate predicate with a reporting duty and couple
// the fire decision to the valuation record. This recorder is read-only, allocates its own
// answer, and shares no control flow with the fire path: compaction's byte-for-byte behavior
// is unchanged whether or not anyone calls it.
//
// SCOPE FENCE. This supplies the raw LAYOUT, never the overlap JUDGMENT. Deciding whether a
// given fire's shed span was warm — and what that is worth — belongs to the netting in the
// cachevaluereport lane, which joins this layout against the same fire's drop range. Concretely,
// a breakpoint at index i caches messages[0..i], so a reader holding this record and the same
// fire's drop range [dropStart, keepStart) determines the shed span sat inside a live cached
// prefix iff the HIGHEST recorded Index >= keepStart-1. That comparison is deliberately left to
// the reader: this file states the marks, it does not rule on them.

import "encoding/json"

// BreakpointPosition is one inbound cache_control breakpoint: WHERE it sits in the inbound
// messages[] array (Index, 0-based) and the role of the message carrying it (Role, "user" /
// "assistant", or "" when the element has no parseable role). Index is the position in the
// INBOUND body as received — before any compaction rewrite — so it is directly comparable to
// the drop range the same fire reports.
type BreakpointPosition struct {
	Index int    `json:"index"`
	Role  string `json:"role,omitempty"`
}

// InboundBreakpoints is the recorded cache_control breakpoint layout of ONE inbound
// /v1/messages body — the per-session record #2786 persists, carrying the `breakpoint_positions`
// field name the ledger reads.
//
// Real Claude Code traffic marks a static head AND recent turns, which is exactly why a bare
// count is not enough: a body with two breakpoints tells a reader nothing about whether a
// given middle span sat inside one's prefix. Positions is ascending by Index.
//
// SystemMarked / ToolsMarked record the STABLE provider head (a top-level system/tools
// breakpoint). They are tracked separately from Positions because the head caches the prompt
// hierarchy ahead of messages[] rather than any message index, so it can never be expressed as
// a messages[] position — and a body whose ONLY breakpoint is the head has a live cached prefix
// that covers no message at all.
type InboundBreakpoints struct {
	// Count is len(Positions) — the number of messages[] breakpoints. It excludes the
	// system/tools head marks, which are not message positions.
	Count int `json:"breakpoint_count"`
	// Positions is every messages[] cache_control breakpoint, ascending by Index.
	Positions []BreakpointPosition `json:"breakpoint_positions,omitempty"`
	// Messages is the inbound messages[] length, so a reader can tell a breakpoint near the
	// END (the recent-turn mark that anchor-starves compaction, #1407) from one near the head
	// without needing the body.
	Messages int `json:"messages"`
	// SystemMarked / ToolsMarked report a top-level system / tools cache_control breakpoint.
	SystemMarked bool `json:"system_marked,omitempty"`
	ToolsMarked  bool `json:"tools_marked,omitempty"`
}

// RecordInboundBreakpoints reads an inbound Anthropic /v1/messages body and records its
// cache_control breakpoint layout. It is PURE and read-only: it decodes, never rewrites, and
// never touches the bytes forwarded upstream, so recording cannot perturb the cached prefix
// the passthrough exists to preserve.
//
// ok is false when the body is not a JSON object, carries no `messages` key, or its `messages`
// value is not a decodable array — the same fail-safe posture as the compaction path, where an
// unreadable body yields no claim rather than a fabricated zero. A well-formed body with no
// breakpoints at all returns ok=true with Count 0: "we looked and there were none" is a
// finding, and must not be confused with "we could not look".
func RecordInboundBreakpoints(raw []byte) (InboundBreakpoints, bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return InboundBreakpoints{}, false
	}
	msgsRaw, hasMsgs := obj["messages"]
	if !hasMsgs {
		return InboundBreakpoints{}, false
	}
	// Decode elements only — this recorder needs indices and roles, never byte spans, so it
	// avoids the span/base machinery (and the bytes.Index aliasing caution, #3773) the splice
	// path needs.
	var elems []json.RawMessage
	if err := json.Unmarshal(msgsRaw, &elems); err != nil {
		return InboundBreakpoints{}, false
	}
	out := InboundBreakpoints{
		Messages:     len(elems),
		SystemMarked: rawHasCacheControl(obj["system"]),
		ToolsMarked:  rawHasCacheControl(obj["tools"]),
	}
	for i, el := range elems {
		if messageHasCacheControl(el) {
			out.Positions = append(out.Positions, BreakpointPosition{Index: i, Role: messageRole(el)})
		}
	}
	out.Count = len(out.Positions)
	return out, true
}

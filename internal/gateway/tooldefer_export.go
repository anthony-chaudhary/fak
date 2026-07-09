package gateway

// tooldefer_export.go — a read-only A/B seam for the #3532 cold-tool-deferral
// scorecard (epic #3229), sibling of mcp_footprint_export.go.
//
// It exposes the PRODUCTION #3232 transform (deferColdToolsInBody + the production
// defaultHotToolSet + the same agent decode gate maybeDeferColdTools uses) so cmd/fak
// can price the ARMED arm (cold defs deferred) against the ABLATED arm (identity —
// byte-for-byte what FAK_ABLATE_DEFER_TOOLS yields at the live seam) with the ONE house
// estimator, never a re-implementation of the lever.
//
// HONESTY (the load-bearing nuance from #3232): defer_loading does NOT shrink request
// bytes — every def stays in tools[] and the armed body GROWS by the defer_loading keys
// + the injected tool_search_tool. The reduction this seam lets the scorecard report is
// the ESTIMATED PROVIDER-SIDE resident tool slice: the tokens of the defs Anthropic
// actually loads into context (non-deferred defs + the search tool) armed, vs ALL defs
// ablated. The truly OBSERVED reduction lives only in the provider usage relay + the
// fak_gateway_tool_defer_* /metrics counters on a live run (#3233/#3536).

import (
	"bytes"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// DeferABArms holds one canonical body in both arms plus the transform's witnessed
// outcome. Ablated is always the verbatim input; Armed carries defer_loading on the
// cold defs and one injected tool_search_tool when the transform fires, else it equals
// Ablated and Reason names the fail-safe stand-down.
type DeferABArms struct {
	Ablated   []byte
	Armed     []byte
	Changed   bool
	ColdCount int
	Reason    string
}

// DeferColdToolsAB runs the REAL production transform (deferColdToolsInBody with the
// production defaultHotToolSet and the same agent.DecodeAnthropicMessagesRequest gate
// maybeDeferColdTools uses) over raw and returns both arms. It never mutates raw.
func DeferColdToolsAB(raw []byte) DeferABArms {
	res := deferColdToolsInBody(raw, defaultHotToolSet, func(b []byte) error {
		_, err := agent.DecodeAnthropicMessagesRequest(b)
		return err
	})
	arms := DeferABArms{
		Ablated:   raw,
		Changed:   res.Changed,
		ColdCount: res.ColdCount,
		Reason:    res.Reason,
	}
	if res.Changed {
		arms.Armed = res.Body
	} else {
		arms.Armed = raw
	}
	return arms
}

// ResidentToolDefs decodes body and returns only the tool defs the PROVIDER keeps
// resident — every advertised tool whose element does NOT carry defer_loading:true.
// On the ablated body that is every tool; on the armed body it is the hot core plus
// the injected tool_search_tool. Fail-safe: an undecodable body yields nil, which a
// caller prices as a zero-token slice rather than crashing.
func ResidentToolDefs(body []byte) []agent.ToolDef {
	deferred := deferredToolNames(body)
	req, err := agent.DecodeAnthropicMessagesRequest(body)
	if err != nil {
		return nil
	}
	out := make([]agent.ToolDef, 0, len(req.Tools))
	for _, t := range req.Tools {
		if deferred[t.Function.Name] {
			continue
		}
		out = append(out, t)
	}
	return out
}

// deferredToolNames returns the set of advertised tool names marked defer_loading:true
// in body's tools[]. Fail-safe empty set on any parse ambiguity, so a caller treats an
// unparseable body as "nothing deferred" and prices every decoded def.
func deferredToolNames(body []byte) map[string]bool {
	out := map[string]bool{}
	var obj map[string]json.RawMessage
	if json.Unmarshal(body, &obj) != nil {
		return out
	}
	toolsRaw, ok := obj["tools"]
	if !ok {
		return out
	}
	var elems []map[string]json.RawMessage
	if json.Unmarshal(toolsRaw, &elems) != nil {
		return out
	}
	for _, m := range elems {
		if v, ok := m["defer_loading"]; ok && string(v) == "true" {
			if name := rawStringField(m, "name"); name != "" {
				out[name] = true
			}
		}
	}
	return out
}

// NonToolsByteIdentical proves the cache-safety invariant the A/B relies on: every byte
// OUTSIDE the tools[] value is identical across the two bodies. The transform only ever
// splices the tools[] span, so system, messages, and every other field must match to
// the byte — the property that keeps the non-tools cache prefix stable turn-over-turn.
// Returns false (conservative) if either body's tools[] span cannot be located.
func NonToolsByteIdentical(a, b []byte) bool {
	aPre, aSuf, ok := nonToolsSpans(a)
	if !ok {
		return false
	}
	bPre, bSuf, ok := nonToolsSpans(b)
	if !ok {
		return false
	}
	return bytes.Equal(aPre, bPre) && bytes.Equal(aSuf, bSuf)
}

// nonToolsSpans returns the raw bytes before and after the tools[] value in raw, using
// the same value-locating technique the transform proves against.
func nonToolsSpans(raw []byte) (prefix, suffix []byte, ok bool) {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, nil, false
	}
	toolsRaw, has := obj["tools"]
	if !has {
		return nil, nil, false
	}
	start := bytes.Index(raw, toolsRaw)
	if start < 0 {
		return nil, nil, false
	}
	end := start + len(toolsRaw)
	return raw[:start], raw[end:], true
}

// CanonicalDeferABBody builds the deterministic Claude-Code-shaped Anthropic Messages
// body the scorecard prices: a minimal system + one user turn, and a tools[] carrying a
// small hot core (Read/Bash, mirroring the messages_tooldefer_test.go fixture) followed
// by the REAL MCP registry (MCPFloorToolDefs) as the cold tail, with the cache_control
// anchor on the last element. Every def is rendered byte-faithfully from its advertised
// bytes so the priced footprint matches what a live client would send.
func CanonicalDeferABBody() []byte {
	hot := []agent.ToolDef{
		{Function: agent.ToolDefFunction{Name: "Read", Description: "read a file", Parameters: json.RawMessage(`{"type":"object"}`)}},
		{Function: agent.ToolDefFunction{Name: "Bash", Description: "run a command", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	defs := append(hot, MCPFloorToolDefs()...)

	elems := make([]json.RawMessage, 0, len(defs))
	for i, d := range defs {
		params := d.Function.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object"}`)
		}
		m := map[string]json.RawMessage{
			"name":         mustMarshalString(d.Function.Name),
			"input_schema": params,
		}
		if d.Function.Description != "" {
			m["description"] = mustMarshalString(d.Function.Description)
		}
		if i == len(defs)-1 {
			m["cache_control"] = json.RawMessage(`{"type":"ephemeral"}`)
		}
		b, _ := json.Marshal(m)
		elems = append(elems, b)
	}
	toolsArr, _ := json.Marshal(elems)

	var buf bytes.Buffer
	buf.WriteString(`{"model":"claude-x","system":"SYS-PROMPT","messages":[{"role":"user","content":"hello"}],"tools":`)
	buf.Write(toolsArr)
	buf.WriteByte('}')
	return buf.Bytes()
}

func mustMarshalString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

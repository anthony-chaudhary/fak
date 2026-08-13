package gateway

// messages_tooldefer.go — the 10x floor lever (#3232, epic #3229): defer the COLD
// tool tail at the outbound Anthropic Messages seam. On a fresh guarded session
// /context shows ~41k resident before any work; the tool schemas alone are ~35.8k
// (built-in system tools ~25.9k + MCP tools ~9.9k). To fak's gateway those are all
// just req.Tools on the outbound request — the ONE seam where the systemic built-in
// slice is reachable. This marks every ALLOWED-but-COLD custom tool
// `defer_loading:true` and injects a standard `tool_search_tool`, so Anthropic loads
// only the hot core into context and faults a cold schema in on demand when the
// model searches for it.
//
// LOAD-BEARING NUANCE (from the issue): defer_loading does NOT shrink the request
// BYTES — every def stays in tools[] so Anthropic can search them; the body GROWS by
// the defer_loading keys + the search tool. The reduction is PROVIDER-SIDE (Anthropic
// loads only non-deferred defs into context) and shows up in the OBSERVED usage relay,
// never in the ESTIMATED byte footprint.
//
// CACHE SAFETY: the transform is DETERMINISTIC — identical input tools[] yield
// byte-identical deferred tools[] every turn — so the cache_control prefix stays
// stable turn-over-turn and the session cache survives (a turn-0-only rewrite would
// mismatch the client's non-deferred turn-1 body and bust the cache every turn). The
// non-tools body bytes (system, messages) are spliced through VERBATIM, and the result
// is proven byte-identical outside the tools[] span AND re-decoded before it ships.
// Fail-safe identity on ANY ambiguity.
//
// DEFAULT OFF (Config.DeferColdTools / --defer-cold-tools; ablation FAK_ABLATE_DEFER_TOOLS):
// this is the epic's highest-risk lever. The exact Anthropic tool_search_tool wire type
// + beta value (toolSearchToolType / toolSearchBeta below) and the A/B (token-delta ×
// held-accuracy × poison-rate) are the validation gates before the default flips on;
// #3200's pin/quarantine guards the fault-in. The mechanism, its determinism, and its
// fail-safety are witnessed by messages_tooldefer_test.go.

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// toolSearchToolType / toolSearchToolName are the current Anthropic Tool Search
// wire constants. Anthropic retired the 2025-09-17 beta and generic descriptor;
// the current regex variant is accepted without that beta header. Keep these named:
// a stale dated type is rejected upstream before a managed session can start.
const (
	toolSearchToolType = "tool_search_tool_regex_20251119"
	toolSearchToolName = "tool_search_tool_regex"
)

// extendedCacheTTLBeta is the Anthropic beta that admits an extended (1h) cache_control
// TTL. It is a LIVE validation gate: a body carrying
// {"type":"ephemeral","ttl":"1h"} WITHOUT this beta negotiated is 400'd upstream as
// malformed ("parameter ranges"). The wrapped claude CLI defaults to the 5m tier and does
// NOT send it, so the served-request transform must union it in itself the turn the
// managed-cache 1h upgrade (maybeUpgradeAnthropicCacheTTL1H) sets that ttl — when the managed-cache transform sets defer_loading.
const extendedCacheTTLBeta = "extended-cache-ttl-2025-04-11"

// defaultHotToolSet is the eager core kept resident: the guard floor's built-in
// system tools (Read/Edit/Write/Bash/Grep/Glob/Task/TodoWrite + web) plus the search
// tool itself. Everything else custom is cold and deferred. Anthropic-hosted TYPED
// server tools (a non-custom "type") are also kept eager — they cannot be deferred.
var defaultHotToolSet = map[string]bool{
	"Bash": true, "Read": true, "Edit": true, "Write": true, "MultiEdit": true,
	"Glob": true, "Grep": true, "TodoWrite": true, "Task": true,
	"WebFetch": true, "WebSearch": true, "NotebookEdit": true,
	"ToolSearch": true, toolSearchToolName: true,
}

// deferResult is the outcome of a defer transform: Changed with the spliced Body on a
// fired turn, else a named Reason for the fail-safe identity.
//
// ColdNames carries WHICH defs were marked, in tools[] order (#3647). The transform is the
// only place that identity exists — past this point the names are recoverable only by
// re-parsing the spliced body for defer_loading keys — so dropping them here is what left
// the operator surfaces reporting a bare count with no answer to "which tools went cold".
// It is observability-only: nothing in the splice, the byte-equality proof, or the
// determinism the session cache depends on reads this field.
type deferResult struct {
	Body      []byte
	Changed   bool
	Reason    string
	ColdCount int
	ColdNames []string
}

// maybeDeferColdTools is the req.Raw transform slotted next to maybeCompactInboundTools.
// It fires only when the lever is on, the wire is the Anthropic passthrough, and the
// ablation arm is off; it records a witnessed metric and mutates req.Raw only on a
// proven change. Returns the number of cold defs deferred (0 = no-op).
func (s *Server) maybeDeferColdTools(req *agent.AnthropicMessagesRequest, trace string) int {
	if s == nil || !s.deferColdTools || req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return 0
	}
	if envEnabled("FAK_ABLATE_DEFER_TOOLS") {
		return 0
	}
	res := deferColdToolsInBody(req.Raw, defaultHotToolSet, func(b []byte) error {
		_, err := agent.DecodeAnthropicMessagesRequest(b)
		return err
	})
	// Names ride with the count (#3647). observeToolDefer has taken them since the deferred-tool
	// list was added to /debug/vars and the `fak info` Cache tab, but this — the only live caller —
	// passed none, so the distinct-name set stayed empty on every real session and both operator
	// surfaces reported a shed count with no answer to WHICH tools went cold. The unit tests could
	// not catch it: they populate the name set at levels above this seam.
	s.metrics.observeToolDefer(res.ColdCount, res.Changed, res.ColdNames...)
	if !res.Changed {
		// #3621: book the eligible-but-inert turn. Everything above this line is the
		// eligibility gate (lever on, Anthropic passthrough, ablation off), so reaching here
		// means the lever WAS armed and the transform still yielded identity — the one fact a
		// zero-valued fak_gateway_tool_defer_cold_total can otherwise never distinguish from a
		// lever that was simply never turned on. See defer_inert.go for the fold.
		s.metrics.observeToolDeferStandDown(res.Reason)
		return 0
	}
	req.Raw = res.Body
	return res.ColdCount
}

// deferColdToolsInBody is the pure transform. It splices ONLY the tools[] array value,
// keeping every other body byte verbatim, and proves the untouched regions are
// byte-identical before returning Changed. Fail-safe identity (a named Reason, no Body)
// on any ambiguity. decode (optional) re-validates the spliced body still parses.
func deferColdToolsInBody(raw []byte, hot map[string]bool, decode func([]byte) error) deferResult {
	if len(raw) == 0 {
		return deferResult{Reason: "empty"}
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return deferResult{Reason: "not_json_object"}
	}
	toolsRaw, ok := obj["tools"]
	if !ok {
		return deferResult{Reason: "no_tools"}
	}
	// Locate the tools[] value span in raw. json.Unmarshal of an object preserves each
	// value's bytes verbatim, so a sub-search is exact; a wrong guess can only fail the
	// prefix/suffix byte-equality proof below → identity, never breakage.
	start := bytes.Index(raw, toolsRaw)
	if start < 0 {
		return deferResult{Reason: "tools_not_located"}
	}
	end := start + len(toolsRaw)

	var elems []json.RawMessage
	if json.Unmarshal(toolsRaw, &elems) != nil {
		return deferResult{Reason: "undecodable_tools"}
	}
	if len(elems) == 0 {
		return deferResult{Reason: "no_tools"}
	}

	// Idempotent stand-down: if the body is ALREADY deferred (any def carries
	// defer_loading, or a tool_search_tool is present — the Claude Code ENABLE_TOOL_SEARCH
	// case), do not double-apply.
	parsed := make([]map[string]json.RawMessage, len(elems))
	for i, el := range elems {
		var m map[string]json.RawMessage
		if json.Unmarshal(el, &m) != nil {
			return deferResult{Reason: "undecodable_elem"}
		}
		parsed[i] = m
		if _, has := m["defer_loading"]; has {
			return deferResult{Reason: "already_deferred"}
		}
		if strings.HasPrefix(rawStringField(m, "type"), "tool_search_tool") || rawStringField(m, "name") == toolSearchToolName {
			return deferResult{Reason: "already_deferred"}
		}
	}

	// Build the new element list: hot / typed-server tools verbatim; cold custom tools
	// gain defer_loading:true. Track whether the block was cached to carry the anchor.
	newElems := make([]json.RawMessage, 0, len(elems)+1)
	cold := 0
	var coldNames []string
	lastHadCacheControl := false
	for i, m := range parsed {
		if i == len(parsed)-1 {
			_, lastHadCacheControl = m["cache_control"]
		}
		name := rawStringField(m, "name")
		typ := rawStringField(m, "type")
		isCustom := typ == "" || typ == "custom" || typ == "function"
		if hot[name] || !isCustom {
			newElems = append(newElems, elems[i]) // verbatim
			continue
		}
		m["defer_loading"] = json.RawMessage("true")
		nb, err := json.Marshal(m)
		if err != nil {
			return deferResult{Reason: "remarshal_failed"}
		}
		newElems = append(newElems, nb)
		cold++
		coldNames = append(coldNames, name)
	}
	if cold == 0 {
		return deferResult{Reason: "no_cold_tools"}
	}

	// Inject one tool_search_tool as the new tail. Carry the cache_control anchor onto it
	// (only if the client was caching tools) so the augmented block becomes the stable
	// cached head — the re-anchor, done deterministically each turn.
	newElems = append(newElems, toolSearchToolElement(lastHadCacheControl))

	newArr, err := json.Marshal(newElems)
	if err != nil {
		return deferResult{Reason: "marshal_array_failed"}
	}

	out := make([]byte, 0, start+len(newArr)+(len(raw)-end))
	out = append(out, raw[:start]...)
	out = append(out, newArr...)
	out = append(out, raw[end:]...)

	// Prove it: everything OUTSIDE the tools[] value must be byte-identical to the input.
	suffix := raw[end:]
	if !bytes.Equal(raw[:start], out[:start]) || !bytes.Equal(suffix, out[len(out)-len(suffix):]) {
		return deferResult{Reason: "splice_unproven"}
	}
	if decode != nil {
		if err := decode(out); err != nil {
			return deferResult{Reason: "decode_failed"}
		}
	}
	return deferResult{Body: out, Changed: true, ColdCount: cold, ColdNames: coldNames}
}

// toolSearchToolElement is the injected tool_search_tool descriptor, optionally
// carrying the cache_control anchor.
func toolSearchToolElement(cacheControl bool) json.RawMessage {
	if cacheControl {
		return json.RawMessage(`{"type":"` + toolSearchToolType + `","name":"` + toolSearchToolName + `","cache_control":{"type":"ephemeral"}}`)
	}
	return json.RawMessage(`{"type":"` + toolSearchToolType + `","name":"` + toolSearchToolName + `"}`)
}

// rawStringField reads a string field from a decoded object, "" if absent/non-string.
func rawStringField(m map[string]json.RawMessage, key string) string {
	v, ok := m[key]
	if !ok {
		return ""
	}
	var s string
	if json.Unmarshal(v, &s) != nil {
		return ""
	}
	return s
}

// unionBeta adds add to the comma-separated anthropic-beta list if not already present.
func unionBeta(existing, add string) string {
	if strings.TrimSpace(add) == "" {
		return existing
	}
	for _, p := range strings.Split(existing, ",") {
		if strings.TrimSpace(p) == add {
			return existing
		}
	}
	if strings.TrimSpace(existing) == "" {
		return add
	}
	return existing + "," + add
}

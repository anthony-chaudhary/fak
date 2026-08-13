package gateway

// messages_inbound_prune.go — the INBOUND prune seam of the native Anthropic Messages front
// door, split out of messages.go (#5849) so that file stays under its god-file pin. Where
// messages.go compacts what the model is about to SEE of its own history, everything here
// prunes what the CLIENT sent us before the planner sees it: tool definitions the capability
// floor can never admit, and system blocks the prompt MMU says are dead weight.

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// maybeCompactInboundTools is the INBOUND twin of maybeCompactAnthropicRaw: on the
// Anthropic passthrough it prunes tool DEFINITIONS the capability floor can never admit
// (s.toolFloorDenies(name) — a DEFAULT_DENY for every arg, never an arg-conditional tool)
// from the outbound tools[], splicing on the original bytes so the cache_control prefix
// stays byte-identical and the upstream prompt-cache hit survives (promptmmu.CompactInboundTools).
//
// It is behavior-preserving by construction: if the model somehow names a pruned tool, the
// kernel still DEFAULT_DENYs the call — so removing the advertisement only shrinks the
// uncached tool-def tokens, never the reachable action set. A no-op (identity) unless the
// gateway is fronting the real Anthropic API (only there is req.Raw forwarded verbatim) and
// the host supplied the floor predicate. promptmmu is fail-safe: any ambiguity returns
// req.Raw unchanged with a named SkipReason, so this never breaks a turn.
func (s *Server) maybeCompactInboundTools(req *agent.AnthropicMessagesRequest) (pruned []string) {
	// #2440: admit each advertised tool schema into the ctxmmu-owned catalog BEFORE any
	// prune decision. The page table — not this turn's transcript — is the tool catalog's
	// home, so a schema pruned/evicted from the outbound body is still a re-faultable page,
	// never a lost one. Registration is keyed by content hash, so identical schemas re-sent
	// turn after turn dedupe (the tool_page_dedup_hits_total witness) instead of re-inflating.
	s.registerToolSchemaPages(req)
	return s.compactInboundToolsWithDecision(req).Removed
}

// registerToolSchemaPages admits every advertised tool's schema into the ctxmmu tool-page
// catalog as a content-hashed read-only page. Dedup is by content hash (never by name), so a
// schema re-advertised each turn registers once and every repeat is a counted dedup hit; two
// versions of one tool name are two distinct pages that never collide. Pure side effect on the
// catalog — it never mutates req — and nil-safe for a bare Server or an empty tools[].
func (s *Server) registerToolSchemaPages(req *agent.AnthropicMessagesRequest) {
	if s == nil || s.toolPages == nil || req == nil {
		return
	}
	for _, t := range req.Tools {
		if schema := canonicalToolSchemaBytes(t); len(schema) > 0 {
			s.toolPages.Register(t.Function.Name, schema)
		}
	}
}

// canonicalToolSchemaBytes renders one tool definition to a stable byte blob for content
// hashing: name, description, and the parameter JSON Schema joined by NUL separators. The same
// advertised tool yields the same bytes turn after turn (so it dedupes), while any change to
// name, description, or schema yields distinct bytes (a distinct page). An empty name AND empty
// parameters yields no bytes (nothing to page). It is deliberately NOT a JSON re-marshal: the
// raw Parameters bytes are used verbatim so re-marshal whitespace churn can never move the hash.
func canonicalToolSchemaBytes(t agent.ToolDef) []byte {
	if t.Function.Name == "" && len(t.Function.Parameters) == 0 {
		return nil
	}
	var b []byte
	b = append(b, t.Function.Name...)
	b = append(b, 0)
	b = append(b, t.Function.Description...)
	b = append(b, 0)
	b = append(b, t.Function.Parameters...)
	return b
}

func (s *Server) compactInboundToolsWithDecision(req *agent.AnthropicMessagesRequest) promptmmu.ToolSchemaDecision {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) || s.toolFloorDenies == nil {
		return promptmmu.ToolSchemaDecision{Strategy: promptmmu.ToolSchemaUnchanged, SkipReason: inboundPruneDisabled}
	}
	if len(req.Tools) == 0 {
		return promptmmu.ToolSchemaDecision{Strategy: promptmmu.ToolSchemaUnchanged, SkipReason: promptmmu.SkipNoTools}
	}
	drop := make(map[string]bool, len(req.Tools))
	for _, t := range req.Tools {
		if name := t.Function.Name; name != "" && s.toolFloorDenies(name) {
			drop[name] = true
		}
	}
	// tool_choice soundness: a request that PINS a tool ({"tool_choice":{"type":"tool",
	// "name":X}}) references that definition by name — dropping it would forward a body
	// whose tool_choice names a tool absent from tools[], which the upstream rejects as
	// a 400 the client cannot attribute. Keeping the def is the behavior-preserving
	// direction: the model may propose the call and the kernel still DEFAULT_DENYs it
	// with a legible verdict. Observed shape: the host harness's structured-output
	// sidechannel calls (prompt-hook evaluators, schema'd subagents) pin their
	// StructuredOutput return-channel tool.
	if pinned := toolChoicePinnedName(req.Raw); pinned != "" {
		delete(drop, pinned)
	}
	if len(drop) == 0 {
		return promptmmu.ToolSchemaDecision{Strategy: promptmmu.ToolSchemaUnchanged, SkipReason: promptmmu.SkipEmptyPlan}
	}
	res := promptmmu.CompactInboundTools(req.Raw, promptmmu.ToolPlan{Drop: drop}, func(b []byte) error {
		_, err := agent.DecodeAnthropicMessagesRequest(b)
		return err
	})
	decision := promptmmu.ExplainToolSchemaStrategy(res)
	if !res.Changed {
		s.logInboundToolSchemaDecision(decision)
		return decision
	}
	req.Raw = res.Body
	// Record the WITNESSED prune so the lever is no longer invisible: before this, the
	// pruned list was discarded with no metric, so an operator could not tell a turn that
	// shed unreachable tool defs (a pure uncached-token saving) from one that fired zero
	// times. The pruner already proved the cached prefix stayed byte-identical, so a counted
	// prune never bursts the upstream cache (epic #1089 — is-our-thing-ENABLED-and-USED).
	s.metrics.observeInboundToolPrune(len(res.Pruned))
	s.logInboundToolSchemaDecision(decision)
	return decision
}

// toolChoicePinnedName returns the tool name the request's tool_choice pins, or ""
// when tool_choice is absent or not a specific-tool choice ({"type":"auto"|"any"}
// carries no name; a malformed or string-typed field conservatively pins nothing).
func toolChoicePinnedName(raw []byte) string {
	var body struct {
		ToolChoice struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"tool_choice"`
	}
	if json.Unmarshal(raw, &body) != nil {
		return ""
	}
	if body.ToolChoice.Type != "tool" {
		return ""
	}
	return body.ToolChoice.Name
}

func (s *Server) logInboundToolSchemaDecision(decision promptmmu.ToolSchemaDecision) {
	if s == nil || s.logf == nil || decision.Strategy == promptmmu.ToolSchemaUnchanged {
		return
	}
	s.logf("gateway: inbound tool-schema strategy=%s removed=%d skip=%s cache_tradeoff=%q token_tradeoff=%q",
		decision.Strategy, len(decision.Removed), decision.SkipReason, decision.CacheTradeoff, decision.TokenTradeoff)
}

// inboundSystemPrune is the legible outcome of one turn's inbound system[] prune — the
// system-side twin of the promptmmu.ToolSchemaDecision the tools path already produces.
// Before #5446 maybeCompactInboundSystem returned a bare name list its only caller threw
// away and dropped every promptmmu SkipReason on the floor, so a turn that pruned nothing
// because the request carried no system[] array and a turn that pruned nothing because the
// blocks could not be read were indistinguishable from outside the process.
type inboundSystemPrune struct {
	// Pruned names the removed blocks as "<block>:<name>", in system[] order. These are
	// fak-internal block identifiers, never prompt bytes, so they are safe to emit.
	Pruned []string
	// SkipReason is the closed-set promptmmu reason nothing was pruned; empty when
	// something was.
	SkipReason string
	// Structural reports whether SkipReason names a fak fault (a malformed body, an
	// unreadable system[], an unproven splice) rather than an ordinary idle turn. It is
	// promptmmu.SkipReasonIsStructural's verdict, never re-derived here.
	Structural bool
}

// maybeCompactInboundSystem prunes the floor-denied system[] blocks from the outbound body
// and REPORTS what happened. The report is the point: an operator must be able to tell
// "pruned nothing because there was no system[] array" from "pruned nothing because the
// blocks could not be read", and before #5446 both surfaced as an empty list.
func (s *Server) maybeCompactInboundSystem(req *agent.AnthropicMessagesRequest) inboundSystemPrune {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) || s.systemBlockDrop == nil {
		return inboundSystemPrune{SkipReason: inboundPruneDisabled}
	}
	plans, reason := inboundSystemBlockPlans(req.Raw, s.systemBlockDrop)
	if len(plans) == 0 {
		return inboundSystemPrune{SkipReason: reason, Structural: promptmmu.SkipReasonIsStructural(reason)}
	}
	var out inboundSystemPrune
	for _, plan := range plans {
		res := promptmmu.CompactInboundSystem(req.Raw, plan, func(b []byte) error {
			_, err := agent.DecodeAnthropicMessagesRequest(b)
			return err
		})
		if res.Changed {
			req.Raw = res.Body
			for _, name := range res.Pruned {
				out.Pruned = append(out.Pruned, plan.Block+":"+name)
			}
			continue
		}
		// A structural identity outranks everything else this turn, including a sibling
		// block's successful prune: it is never routine, so it must not vanish behind a
		// busy turn. A benign identity is kept only when nothing better was recorded.
		if promptmmu.SkipReasonIsStructural(res.SkipReason) {
			out.SkipReason, out.Structural = res.SkipReason, true
			continue
		}
		if out.SkipReason == "" {
			out.SkipReason = res.SkipReason
		}
	}
	if len(out.Pruned) > 0 && !out.Structural {
		out.SkipReason = ""
	}
	return out
}

// inboundPruneDisabled is the reason both inbound prune paths report when the seam is not
// armed at all — a non-Anthropic wire, or no floor predicate supplied. It is deliberately
// NOT a promptmmu constant: the spine never ran, so no spine reason applies.
const inboundPruneDisabled = "gateway-disabled"

// logInboundSystemPrune emits the out-of-band witness for one inbound system[] prune,
// mirroring logInboundToolSchemaDecision on the tools side. It stays silent on the dominant
// idle turn (no system[] array, or nothing droppable) so the log never fills with vacuous
// lines, and speaks on exactly the two events worth reading: a real prune, and a STRUCTURAL
// skip — a fak fault that must never be readable as "there was nothing to do".
func (s *Server) logInboundSystemPrune(rec inboundSystemPrune) {
	if s == nil || s.logf == nil {
		return
	}
	if len(rec.Pruned) == 0 && !rec.Structural {
		return
	}
	s.logf("gateway: inbound system-block prune pruned=%d blocks=%v skip=%s structural=%t",
		len(rec.Pruned), rec.Pruned, rec.SkipReason, rec.Structural)
}

func auditPromptSerialization(raw []byte) promptmmu.SerializationAudit {
	return promptmmu.AuditJSONRemarshal(raw)
}

// inboundSystemBlockPlans reads raw's droppable system[] blocks, grouped by block, AND
// names the closed-set promptmmu reason it produced none. The reason is what makes the
// prune legible: this reader — not the promptmmu spine below it — is where a system[]
// failure is actually REACHABLE on wire bytes, because it is the step that insists the
// elements are typed block objects. Returning a bare empty list from all four exits made an
// unreadable system[] look exactly like an absent one (#5446). reason is "" iff at least
// one plan came back.
func inboundSystemBlockPlans(raw []byte, drop func(block, name string) bool) ([]promptmmu.BlockPlan, string) {
	if drop == nil {
		return nil, inboundPruneDisabled
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, promptmmu.SkipNotJSONObject
	}
	systemRaw := obj["system"]
	// Shape BEFORE any typed read, so an absent `system`, a bare-string one, a JSON null
	// and an object are each reported as the benign non-array they are. Every one of those
	// is a legitimate wire shape and must never be counted as a read failure.
	if len(systemRaw) == 0 || systemRaw[0] != '[' {
		return nil, promptmmu.SkipNoSystem
	}
	var elems []struct {
		Block string `json:"block"`
		Name  string `json:"name"`
	}
	if err := json.Unmarshal(systemRaw, &elems); err != nil {
		// system[] IS array-shaped and still would not read — a non-object element, or a
		// block/name of the wrong JSON type. STRUCTURAL: the ordinary idle turn does not
		// look like this, so it must not share the idle turn's reason.
		return nil, promptmmu.SkipUndecodableSystem
	}
	byBlock := map[string]map[string]bool{}
	var order []string
	for _, elem := range elems {
		block := strings.TrimSpace(elem.Block)
		name := strings.TrimSpace(elem.Name)
		if block == "" || name == "" || !drop(block, name) {
			continue
		}
		if byBlock[block] == nil {
			byBlock[block] = map[string]bool{}
			order = append(order, block)
		}
		byBlock[block][name] = true
	}
	if len(order) == 0 {
		// The array read fine and simply held nothing the floor denies — the dominant,
		// expected-large shape. Benign, and named as such.
		return nil, promptmmu.SkipEmptyPlan
	}
	plans := make([]promptmmu.BlockPlan, 0, len(order))
	for _, block := range order {
		plans = append(plans, promptmmu.BlockPlan{Block: block, Drop: byBlock[block]})
	}
	return plans, ""
}

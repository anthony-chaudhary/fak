package gateway

// ctxfootprint.go — the LIVE gateway footprint (#3233, epic #3229): the ESTIMATED
// structural split of the always-sent request floor, taken on a real inbound
// Anthropic request at fak's own seam. #3230 priced fak's own ~24-tool MCP
// registry OFFLINE; this measures the FULL as-sent floor the harness actually
// ships — the harness system prompt + the built-in system tools (Read/Write/Edit/
// Bash/Grep/Task…) + every MCP tool (mcp__fak__*, mcp__dos__*, …) — the number
// that matches /context's ~41k resident before any work.
//
// It is audit-only: it reads the decoded request and mutates nothing the model
// sees. It runs at the SAME pre-transform anchor as the harness-coherence prefix
// digest (messages.go), BEFORE maybeCompactInboundTools prunes any tool def, so
// the surfaced floor is the harness's AS-SENT floor, not fak's post-prune floor.
//
// Provenance is ESTIMATED (~4 chars/token, agent.RequestFootprint's house
// divisor), carried on the surfaced block and NEVER conflated with the OBSERVED
// resident-token counters in CtxValueReport.Tokens (Law A2): the OBSERVED number
// says how FULL the window is; this ESTIMATED split says WHERE the bytes went.

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// estBytesPerToken mirrors agent's unexported bytesPerTokenEstimate (~4 chars/
// token). Kept in lockstep so the surfaced FloorTokens can never drift from the
// agent.Footprint bucket tokens the same estimator produces.
const estBytesPerToken = 4

// ctxFootprintBytes is the per-trace accumulator's raw byte record of the latest
// inbound footprint. Bytes are exact and additive (unlike floored tokens), so the
// built-in/MCP partition and the Floor==System+Tools invariant hold precisely.
type ctxFootprintBytes struct {
	systemBytes  int
	builtinBytes int // tool schemas whose name has NO mcp__ prefix (built-in system tools)
	mcpBytes     int // tool schemas named mcp__<server>__<tool>
	builtinCount int
	mcpCount     int
	historyBytes int
	tailBytes    int
}

// CtxValueFootprint is the ESTIMATED where-did-they-go split surfaced beside the
// OBSERVED resident tokens in a CtxValueReport. FloorTokens (= system + tools) is
// the headline number comparable to /context; the byte fields are the exact
// partition (BuiltinToolBytes + MCPToolBytes == ToolsBytes; System + Tools ==
// Floor). Absent (nil) until the first Anthropic-passthrough turn is observed.
type CtxValueFootprint struct {
	Provenance       string `json:"provenance"` // ESTIMATED
	FloorTokens      int    `json:"floor_tokens"`
	FloorBytes       int    `json:"floor_bytes"`
	SystemBytes      int    `json:"system_bytes"`
	ToolsBytes       int    `json:"tools_bytes"`
	BuiltinToolBytes int    `json:"builtin_tool_bytes"`
	MCPToolBytes     int    `json:"mcp_tool_bytes"`
	BuiltinToolCount int    `json:"builtin_tool_count"`
	MCPToolCount     int    `json:"mcp_tool_count"`
	HistoryBytes     int    `json:"history_bytes"`
	TailBytes        int    `json:"tail_bytes"`
	TotalBytes       int    `json:"total_bytes"`
}

// report renders the accumulator into the surfaced block. The Floor and Total are
// recomputed from the exact byte fields so the invariants are structural.
func (cf *ctxFootprintBytes) report() *CtxValueFootprint {
	if cf == nil {
		return nil
	}
	toolsBytes := cf.builtinBytes + cf.mcpBytes
	floorBytes := cf.systemBytes + toolsBytes
	return &CtxValueFootprint{
		Provenance:       agent.FootprintProvenance,
		FloorTokens:      floorBytes / estBytesPerToken,
		FloorBytes:       floorBytes,
		SystemBytes:      cf.systemBytes,
		ToolsBytes:       toolsBytes,
		BuiltinToolBytes: cf.builtinBytes,
		MCPToolBytes:     cf.mcpBytes,
		BuiltinToolCount: cf.builtinCount,
		MCPToolCount:     cf.mcpCount,
		HistoryBytes:     cf.historyBytes,
		TailBytes:        cf.tailBytes,
		TotalBytes:       floorBytes + cf.historyBytes + cf.tailBytes,
	}
}

// deFoldSystemReq corrects the folded-system double-count specific to a decoded
// live request. DecodeAnthropicMessagesRequest prepends the system prompt as a
// leading RoleSystem message into req.Messages AND keeps req.System, so a naive
// RequestFootprint counts the system twice (the System bucket AND a History/Tail
// message). This returns a shallow copy whose leading folded-system duplicate is
// dropped, so System is counted once and Floor == System + Tools matches
// /context. The original req is never mutated (the hot path forwards it verbatim).
func deFoldSystemReq(req *agent.AnthropicMessagesRequest) *agent.AnthropicMessagesRequest {
	if req == nil {
		return nil
	}
	if req.System != "" && len(req.Messages) > 0 &&
		req.Messages[0].Role == agent.RoleSystem && req.Messages[0].Content == req.System {
		cp := *req
		cp.Messages = req.Messages[1:]
		return &cp
	}
	return req
}

// observeCtxFootprint prices the inbound request's ESTIMATED footprint and folds
// it per-trace, mirroring observeCtxValue. The expensive char-walk runs OUTSIDE
// the lock; only the pointer swap is guarded. Built-in vs MCP is keyed on the
// wire-witnessed mcp__ prefix Claude Code names every MCP tool with. A nil server,
// empty trace, or nil request is a safe no-op.
func (s *Server) observeCtxFootprint(trace string, req *agent.AnthropicMessagesRequest) {
	if s == nil || strings.TrimSpace(trace) == "" || req == nil {
		return
	}
	fp := agent.RequestFootprint(deFoldSystemReq(req))
	cf := &ctxFootprintBytes{
		systemBytes:  fp.System.Bytes,
		historyBytes: fp.History.Bytes,
		tailBytes:    fp.Tail.Bytes,
	}
	for _, pt := range fp.PerTool {
		if strings.HasPrefix(pt.Name, "mcp__") {
			cf.mcpBytes += pt.Bytes
			cf.mcpCount++
		} else {
			cf.builtinBytes += pt.Bytes
			cf.builtinCount++
		}
	}

	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	v := s.ctxValueForLocked(trace)
	if v == nil {
		return
	}
	v.footprint = cf
}

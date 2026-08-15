package gateway

import "sort"

// debugTokenSavingsVars is the bounded, privacy-safe answer to four operator
// questions: is each default armed, did it actually fire, why did it bail, and
// what one-line rollback restores the baseline. It contains no prompt, path,
// tool argument, trace, or result content.
type debugTokenSavingsVars struct {
	NativeMCPFilter debugTokenSavingLever `json:"native_mcp_filter"`
	ColdToolDefer   debugTokenSavingLever `json:"cold_tool_defer"`
	StaleReadElide  debugTokenSavingLever `json:"stale_read_elide"`
	HistoryCompact  debugTokenSavingLever `json:"history_compact"`
}

type debugTokenSavingLever struct {
	Configured  bool              `json:"configured"`
	State       string            `json:"state"` // active | ready | bypassed | off
	Reason      string            `json:"reason"`
	Fired       uint64            `json:"fired"`
	Units       uint64            `json:"units"`
	SavedBytes  uint64            `json:"saved_bytes,omitempty"`
	SavedTokens uint64            `json:"saved_tokens,omitempty"`
	BailReasons map[string]uint64 `json:"bail_reasons,omitempty"`
	Rollback    string            `json:"rollback"`
}

func (s *Server) tokenSavingsVars(sum AdjudicationSummary) debugTokenSavingsVars {
	mcp := s.MCPToolFilterStatusSnapshot()
	native := debugTokenSavingLever{
		Configured: true, State: mcp.Mode, Reason: mcp.Reason,
		Units:      uint64(mcp.ToolsBefore - mcp.ToolsAfter),
		SavedBytes: uint64(max(mcp.SavedBytes, 0)),
		Rollback:   "FAK_ABLATE_MCP_TOOL_FILTER=1",
	}
	if mcp.Mode == "active" {
		native.Fired = 1
	} else {
		native.State = "bypassed"
	}

	deferLever := debugTokenSavingLever{
		Configured: s.deferColdTools, Fired: sum.DeferColdTurns, Units: sum.DeferColdCount,
		BailReasons: sum.DeferStandDownReasons,
		Rollback:    "--defer-cold-tools=false",
	}
	switch {
	case !s.deferColdTools:
		deferLever.State, deferLever.Reason = "off", "configured_off"
	case envEnabled("FAK_ABLATE_DEFER_TOOLS"):
		deferLever.State, deferLever.Reason = "bypassed", "ablation"
	case sum.DeferColdTurns > 0:
		deferLever.State, deferLever.Reason = "active", "deferred"
	case sum.DeferStandDownTurns > 0:
		deferLever.State, deferLever.Reason = "bypassed", topReason(sum.DeferStandDownReasons)
	default:
		deferLever.State, deferLever.Reason = "ready", "not_observed"
	}

	staleTurns, staleReads, staleBytes, staleTokens, staleBails := s.metrics.staleElideSnapshot()
	stale := debugTokenSavingLever{
		Configured: s.elideStaleReads, Fired: staleTurns, Units: staleReads,
		SavedBytes: staleBytes, SavedTokens: staleTokens, BailReasons: staleBails,
		Rollback: "--elide-stale-reads=false",
	}
	switch {
	case !s.elideStaleReads:
		stale.State, stale.Reason = "off", "configured_off"
	case staleTurns > 0:
		stale.State, stale.Reason = "active", "elided"
	case len(staleBails) > 0:
		stale.State, stale.Reason = "bypassed", topReason(staleBails)
	default:
		stale.State, stale.Reason = "ready", "not_observed"
	}

	compact := debugTokenSavingLever{
		Configured: s.compactHistoryBudget > 0, Fired: sum.CompactionFired,
		Units: sum.CompactionDroppedTurns, SavedTokens: sum.CompactionShedTokens,
		BailReasons: sum.CompactionBailReasons,
		Rollback:    "--compact-history-budget=0",
	}
	switch {
	case s.compactHistoryBudget <= 0:
		compact.State, compact.Reason = "off", "configured_off"
	case sum.CompactionFired > 0:
		compact.State, compact.Reason = "active", "compacted"
	case sum.CompactionBailed > 0:
		compact.State, compact.Reason = "bypassed", topReason(sum.CompactionBailReasons)
	default:
		compact.State, compact.Reason = "ready", "not_observed"
	}
	return debugTokenSavingsVars{native, deferLever, stale, compact}
}

func topReason(reasons map[string]uint64) string {
	if len(reasons) == 0 {
		return "identity"
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best := keys[0]
	for _, k := range keys[1:] {
		if reasons[k] > reasons[best] {
			best = k
		}
	}
	return best
}

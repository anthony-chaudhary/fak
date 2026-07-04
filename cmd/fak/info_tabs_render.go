package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// Rendering for the interactive tabbed `fak info` overlay (input + state live in info_tabs.go).
// renderGuardInfoInteractiveBlock is the interactive twin of renderGuardInfoVisualBlock: it draws
// the tab bar, then either the glossary overlay or the active view's focused body, fitted to the
// pane. Every view is a pure projection of the same payload-free /debug/vars snapshot the overview
// renders — no new gateway read, no prompt/result text.

// renderGuardInfoInteractiveBlock renders the whole interactive block for the current UI state:
// row 1 is the tab bar; the rest is the glossary overlay (when open) or the active view's body.
// It reuses the overview's cell-width helpers so a sparkline/gauge/label can never wrap the pane,
// and caps the total rows to the pane height so the in-place redraw math stays exact.
func renderGuardInfoInteractiveBlock(state infoViewState, v guardInfoVars, tr *guardInfoTrend, width, height int) string {
	if width <= 0 {
		width = 80
	}
	ctx := guardInfoPanelCtx{
		v:      v,
		tr:     tr,
		width:  width,
		sparkW: clampIntTUI(width-26, 8, 28),
		gaugeW: clampIntTUI(width-28, 6, 20),
	}
	rows := []string{buildInfoTabBar(state.active, state.glossaryOpen).text}

	// bodyHeight is the room left under the tab bar; 0/negative height (unknown pane) means
	// "roomy", so the body renders in full and the loop's own cap pins it.
	bodyHeight := 0
	if height > 0 {
		bodyHeight = height - 1
	}
	var body []string
	if state.glossaryOpen {
		body = renderInfoGlossaryBody(state.glossaryTerm, width)
	} else {
		body = renderInfoView(state.active, ctx, bodyHeight)
	}
	rows = append(rows, body...)

	if height > 0 && len(rows) > height {
		rows = rows[:height]
	}
	for i, r := range rows {
		rows[i] = takeCellsTUI(r, width)
	}
	return strings.Join(rows, "\n")
}

// renderInfoView renders one focused view's body (the rows below the tab bar), fitted to
// bodyHeight (0/negative = roomy). Overview reuses the full composed panel stack; the rest are
// single-subsystem projections that show one subsystem without the overview's degradation.
func renderInfoView(view infoView, ctx guardInfoPanelCtx, bodyHeight int) []string {
	switch view {
	case viewAgents:
		return fitInfoBody(renderInfoAgentsView(ctx.v), bodyHeight)
	case viewEndpoints:
		return fitInfoBody(renderInfoEndpointsView(ctx), bodyHeight)
	case viewCache:
		return fitInfoBody(renderInfoCacheView(ctx), bodyHeight)
	case viewSafety:
		return fitInfoBody(renderInfoSafetyView(ctx.v), bodyHeight)
	default: // viewOverview
		// The overview reuses the whole composed panel stack (with its identity row), fitted by
		// the existing composer to the body height.
		return composeGuardInfoPanels(ctx, guardInfoPanels(), bodyHeight)
	}
}

// fitInfoBody caps rows to bodyHeight, folding the remainder into a trailing "+N more" row so a
// long list (many agents, many deny reasons) can never scroll the pane. bodyHeight<=0 (roomy or
// unknown) returns rows unchanged.
func fitInfoBody(rows []string, bodyHeight int) []string {
	if bodyHeight <= 0 || len(rows) <= bodyHeight {
		return rows
	}
	if bodyHeight == 1 {
		return []string{fmt.Sprintf(" +%d more (pane too short)", len(rows))}
	}
	kept := rows[:bodyHeight-1]
	extra := len(rows) - (bodyHeight - 1)
	out := make([]string, 0, bodyHeight)
	out = append(out, kept...)
	out = append(out, fmt.Sprintf(" +%d more", extra))
	return out
}

// renderInfoAgentsView is the expanded Agents view: a fleet-summary header, then one full row per
// live session (main + every sub-agent, with lineage/run-state/wall-clock/budget/activity). It
// shows EVERY session — the overview's 4-row cap is exactly what this view exists to lift.
func renderInfoAgentsView(v guardInfoVars) []string {
	if len(v.Sessions) == 0 {
		return []string{" agents: none running (no session registry wired, or nothing live)"}
	}
	rows := []string{" agents: " + guardInfoAgentsSummary(v.Sessions)}
	for _, s := range v.Sessions {
		rows = append(rows, "  "+guardInfoAgentText(s))
	}
	return rows
}

// renderInfoEndpointsView is the expanded Accounts+Nodes view: the accounts header + seat chips +
// node list (the overview's endpoints panel), then one detail row per seat naming its login
// identity and posture — the roster read an operator wants when a failover walled a seat.
func renderInfoEndpointsView(ctx guardInfoPanelCtx) []string {
	ep := ctx.v.Endpoints
	if ep == nil || (len(ep.Accounts) == 0 && len(ep.Nodes) == 0) {
		return []string{" accounts/nodes: none reported (a fak serve gateway, or no provider configured)"}
	}
	rows := guardInfoEndpointsPanelRows(ctx, guardPanelFull)
	if len(ep.Accounts) > 0 {
		rows = append(rows, " seats:")
		for _, a := range ep.Accounts {
			rows = append(rows, "  "+guardInfoSeatDetail(a))
		}
	}
	return rows
}

// guardInfoSeatDetail renders one seat's detail row: its posture (active/walled/idle), name,
// login identity (email when known), and login readiness — the per-seat expansion the compact
// chip row folds away.
func guardInfoSeatDetail(a gateway.SessionAccount) string {
	posture := "idle"
	switch {
	case a.Active:
		posture = "active"
	case a.Walled:
		posture = "walled"
	case !a.CanServe:
		posture = "not offerable"
	}
	parts := []string{fmt.Sprintf("%-12s %s", a.Name, posture)}
	if email := strings.TrimSpace(a.Email); email != "" {
		parts = append(parts, "login "+email)
	}
	if login := strings.TrimSpace(a.LoginStatus); login != "" {
		parts = append(parts, guardLoginWord(login))
	}
	return strings.Join(parts, " · ")
}

// renderInfoCacheView is the expanded Cache view: the cache gauge + saving verdict + owner split,
// the trend sparklines (hit/save/work + turns saved), and the raw savings figure — the whole cache
// economy in one place instead of split across the overview's trends + tasks panels.
func renderInfoCacheView(ctx guardInfoPanelCtx) []string {
	v := ctx.v
	cacheRow := fmt.Sprintf(" cache  %s %.0f%%  %s", gaugeBarTUI(guardInfoHitFrac(v), ctx.gaugeW), guardInfoHitPct(v), guardInfoSavingWord(v))
	if split := guardInfoCacheAttributionText(v); split != "" {
		cacheRow += " · " + split
	}
	rows := []string{cacheRow, fmt.Sprintf(" saved  %s tokens so far", signedTokens(guardInfoSaved(v)))}
	rows = append(rows, guardInfoTrendsPanelRows(ctx, guardPanelFull)...)
	return rows
}

// renderInfoSafetyView is the expanded Safety view: the plain-words safety word, then the FULL
// deny/quarantine reason breakdown (every reason, uncapped — the overview's why-clause shows only
// the top three), the held-for-witness and deferred tallies, and any upstream incident.
func renderInfoSafetyView(v guardInfoVars) []string {
	rows := []string{" " + guardSafetyWord(v)}
	if a := v.Adjudication; a != nil {
		for _, rc := range sortedReasonCounts(a.ByReason) {
			rows = append(rows, fmt.Sprintf("  blocked: %-18s ×%d", rc.code, rc.count))
		}
		if a.Escalated > 0 {
			rows = append(rows, fmt.Sprintf("  held for witness: %d (paused pending approval)", a.Escalated))
		}
		if a.Deferred > 0 {
			rows = append(rows, fmt.Sprintf("  deferred: %d (result let through, taint raised)", a.Deferred))
		}
	}
	if incident := guardInfoIncidentText(v); incident != "" {
		rows = append(rows, " incident: "+incident)
	}
	return rows
}

// reasonCount is one (reason code, count) pair for the safety view's full breakdown.
type reasonCount struct {
	code  string
	count uint64
}

// sortedReasonCounts returns the ByReason map as a slice ordered by count desc, then code asc —
// the same stable order guardInfoTopReasons uses, so the overview's capped clause and the safety
// view's full list agree on which reasons lead.
func sortedReasonCounts(byReason map[string]uint64) []reasonCount {
	rows := make([]reasonCount, 0, len(byReason))
	for code, n := range byReason {
		if n == 0 {
			continue
		}
		rows = append(rows, reasonCount{code, n})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].count != rows[j].count {
			return rows[i].count > rows[j].count
		}
		return rows[i].code < rows[j].code
	})
	return rows
}

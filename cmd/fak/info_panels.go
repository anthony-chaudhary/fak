package main

// The modular fak-info panel spine. The live `fak info` pane (the 20% strip
// `fak guard --split` opens beside the agent) started as one hard-coded layout switch;
// this file replaces that with a REGISTRY of self-contained panels so the pane can grow
// a section per fak subsystem without every addition re-deriving the height math.
//
// HOW TO GROW THE PANE — adding a panel is three steps, all in this file:
//
//  1. write a pure rows func `func myPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string`
//     that projects guardInfoVars (one /debug/vars snapshot; extend that decode in
//     info.go if the gateway grows a new block) into gutter-labeled rows. Return nil
//     when the panel has nothing to say — a silent panel costs zero rows.
//  2. append `{name, degrade, rows}` to guardInfoPanels(). `name` is the section-rule
//     label at full height; `degrade` is the shrink rank (higher = degraded to its
//     one-row mini form, then hidden, sooner as the pane gets shorter).
//  3. pin the panel's content + fit in info_visual_test.go / info_panels_test.go.
//
// The composer below owns ALL fitting: rules are dropped first, then panels degrade
// full→mini→hidden in degrade order, so every panel automatically works at every pane
// height. Panels stay payload-free by construction — they can only render what
// /debug/vars carries, and that surface never carries prompt/result text.

import (
	"fmt"
	"strings"
)

// guardInfoPanelLevel is how much room a panel has been granted by the composer.
type guardInfoPanelLevel int

const (
	guardPanelFull guardInfoPanelLevel = iota // all body rows
	guardPanelMini                            // a single summary row
)

// guardInfoPanelCtx is everything a panel's rows func may read: the current
// /debug/vars snapshot, the trend ring, and the pre-computed sparkline/gauge widths.
// Panels take values and return strings — no I/O, no shared state.
type guardInfoPanelCtx struct {
	v      guardInfoVars
	tr     *guardInfoTrend
	width  int
	sparkW int
	gaugeW int
}

// guardInfoPanel is one self-contained section of the live pane.
type guardInfoPanel struct {
	name    string // section-rule label when the pane is tall enough for rules
	degrade int    // shrink rank: higher degrades (mini) and hides sooner
	rows    func(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string
}

// guardInfoPanels is the pane's panel registry, in DISPLAY order. Degrade ranks are
// independent of display order: incident (0) survives longest — it only exists when
// something is wrong — then tasks (safety+cache verdicts), then the agents and
// resources live-usage panels, with the trend sparklines (nice-to-have shape) first
// to shrink.
func guardInfoPanels() []guardInfoPanel {
	return []guardInfoPanel{
		{name: "trends", degrade: 4, rows: guardInfoTrendsPanelRows},
		{name: "tasks", degrade: 1, rows: guardInfoTasksPanelRows},
		{name: "incident", degrade: 0, rows: guardInfoIncidentPanelRows},
		{name: "resources", degrade: 3, rows: guardInfoResourcesPanelRows},
		{name: "agents", degrade: 2, rows: guardInfoAgentsPanelRows},
	}
}

// composeGuardInfoPanels fits the registered panels into a pane of the given height
// (0 or negative = unknown, treated as roomy) and returns the final rows, identity
// row first. The fit sequence is deterministic:
//
//  1. every non-silent panel at full, with section rules — the roomy layout;
//  2. drop the rules (the old "compact" look);
//  3. degrade panels full→mini one at a time in degrade order (highest first);
//  4. hide panels one at a time in the same order;
//  5. a pane too short even for that (height 1-2) gets the single compact status line.
func composeGuardInfoPanels(ctx guardInfoPanelCtx, panels []guardInfoPanel, height int) []string {
	if height > 0 && height <= 2 {
		return []string{guardInfoVisualTinyRow(ctx.v)}
	}
	budget := height - 1 // the identity row is always spent
	unbounded := height <= 0

	// Materialize each panel's full/mini forms once; a panel whose full form is empty
	// is silent this tick and costs nothing.
	type paneState struct {
		panel guardInfoPanel
		full  []string
		mini  []string
		level guardInfoPanelLevel
		show  bool
	}
	states := make([]*paneState, 0, len(panels))
	for _, p := range panels {
		full := p.rows(ctx, guardPanelFull)
		if len(full) == 0 {
			continue
		}
		states = append(states, &paneState{panel: p, full: full, mini: p.rows(ctx, guardPanelMini), level: guardPanelFull, show: true})
	}

	count := func(rules bool) int {
		n := 0
		for _, st := range states {
			if !st.show {
				continue
			}
			rows := st.full
			if st.level == guardPanelMini {
				rows = st.mini
			}
			n += len(rows)
			if rules && st.level == guardPanelFull {
				n++ // the section rule row
			}
		}
		return n
	}
	render := func(rules bool) []string {
		out := []string{guardInfoVisualIdentityRow(ctx.v)}
		for _, st := range states {
			if !st.show {
				continue
			}
			rows := st.full
			if st.level == guardPanelMini {
				rows = st.mini
			}
			if rules && st.level == guardPanelFull {
				out = append(out, guardInfoRuleTUI(st.panel.name, ctx.width))
			}
			out = append(out, rows...)
		}
		return out
	}

	if unbounded || count(true) <= budget {
		return render(true)
	}
	if count(false) <= budget {
		return render(false)
	}
	// Degrade, then hide, in degrade order (highest rank first; registry order breaks
	// ties so the walk is deterministic).
	order := make([]*paneState, len(states))
	copy(order, states)
	for i := 0; i < len(order); i++ {
		for j := i + 1; j < len(order); j++ {
			if order[j].panel.degrade > order[i].panel.degrade {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	for _, st := range order {
		if len(st.mini) == 0 {
			st.show = false // no mini form: degrading IS hiding
		} else {
			st.level = guardPanelMini
		}
		if count(false) <= budget {
			return render(false)
		}
	}
	for _, st := range order {
		st.show = false
		if count(false) <= budget {
			return render(false)
		}
	}
	return []string{guardInfoVisualTinyRow(ctx.v)}
}

// ── the ported panels: trends / tasks / incident ──

// guardInfoTrendsPanelRows is the trend sub-pane: sparklines of savings, cache hit,
// and work over the last ~minute of ticks. Mini keeps the headline save sparkline.
func guardInfoTrendsPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	v := ctx.v
	saveRow := " save  " + sparklineTUI(ctx.tr.saved, ctx.sparkW) + "  " + signedTokens(guardInfoSaved(v)) + " tok"
	if level == guardPanelMini {
		return []string{saveRow}
	}
	return []string{
		saveRow,
		fmt.Sprintf(" hit   %s  %.0f%%  ×%.2f", sparklineTUI(ctx.tr.hit, ctx.sparkW), guardInfoHitPct(v), guardInfoMult(v)),
		fmt.Sprintf(" work  %s  %d replies · busy %d", sparklineTUI(ctx.tr.turns, ctx.sparkW), v.Inference.Turns, v.Gateway.InflightRequests),
	}
}

// guardInfoTasksPanelRows is the task-manager sub-pane: the cache gauge (with the
// owner split when reported) and the safety floor summary. Mini keeps the cache gauge.
func guardInfoTasksPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	v := ctx.v
	cacheRow := fmt.Sprintf(" cache  %s %.0f%%  %s", gaugeBarTUI(guardInfoHitFrac(v), ctx.gaugeW), guardInfoHitPct(v), guardInfoSavingWord(v))
	if split := guardInfoCacheAttributionText(v); split != "" {
		cacheRow += " · " + split
	}
	if level == guardPanelMini {
		return []string{cacheRow}
	}
	return []string{cacheRow, " safety " + guardInfoSafetyText(v)}
}

// guardInfoIncidentPanelRows surfaces upstream/provider incidents (errors by kind,
// auth-refresh outcomes, retries). Silent — zero rows — on a clean session.
func guardInfoIncidentPanelRows(ctx guardInfoPanelCtx, _ guardInfoPanelLevel) []string {
	incident := guardInfoIncidentText(ctx.v)
	if incident == "" {
		return nil
	}
	return []string{" incident " + incident}
}

// ── the live-usage panels: resources / agents ──

// guardInfoResourcesPanelRows is the gateway's own live resource usage — heap (with
// trend sparkline), OS-reserved memory, goroutines, GC cycles — plus the generation
// rates (output tok/s, mean TTFT) and the oldest in-flight request age (the hung-
// request tell). Hidden until the gateway reports a runtime block (NumGoroutine > 0:
// a live Go process always has goroutines, so zero means "no data").
func guardInfoResourcesPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	v := ctx.v
	if v.Runtime.NumGoroutine <= 0 {
		return nil
	}
	heap := guardInfoBytesText(v.Runtime.Memory.HeapAllocBytes)
	if level == guardPanelMini {
		return []string{fmt.Sprintf(" res    %s heap · %d gor · %.1f tok/s", heap, v.Runtime.NumGoroutine, v.Inference.OutputTokensPerSecond)}
	}
	memRow := fmt.Sprintf(" mem    %s  %s heap · %s sys · %d gor · gc %d",
		sparklineTUI(ctx.tr.heap, ctx.sparkW), heap,
		guardInfoBytesText(v.Runtime.Memory.SysBytes), v.Runtime.NumGoroutine, v.Runtime.Memory.NumGC)
	rateRow := fmt.Sprintf(" rate   %.1f tok/s out", v.Inference.OutputTokensPerSecond)
	if v.Inference.MeanTTFTSeconds > 0 {
		rateRow += fmt.Sprintf(" · ttft %.2fs", v.Inference.MeanTTFTSeconds)
	}
	if v.Gateway.InflightRequests > 0 && v.Inference.InflightMaxAgeSeconds > 0 {
		rateRow += fmt.Sprintf(" · oldest req %.0fs", v.Inference.InflightMaxAgeSeconds)
	}
	return []string{memRow, rateRow}
}

// guardInfoAgentsMaxRows caps the per-session rows the agents panel spends at full
// level; sessions beyond the cap fold into one "+N more" row so a wide sub-agent
// fan-out can never scroll the pane.
const guardInfoAgentsMaxRows = 4

// guardInfoAgentsPanelRows is the per-session sub-pane: one row per live session —
// the main agent and each SUB-AGENT it spawned (parent trace + generation) — with run
// state, live wall-clock, and the remaining budget axes. Mini is the one-row fleet
// summary. Hidden when no session registry is wired or nothing is running.
func guardInfoAgentsPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	ss := ctx.v.Sessions
	if len(ss) == 0 {
		return nil
	}
	if level == guardPanelMini {
		return []string{" agents " + guardInfoAgentsSummary(ss)}
	}
	limit := len(ss)
	if limit > guardInfoAgentsMaxRows {
		limit = guardInfoAgentsMaxRows
	}
	rows := make([]string, 0, limit+1)
	for _, s := range ss[:limit] {
		rows = append(rows, " agent  "+guardInfoAgentText(s))
	}
	if extra := len(ss) - limit; extra > 0 {
		rows = append(rows, fmt.Sprintf(" agent  +%d more", extra))
	}
	return rows
}

// guardInfoAgentText renders one session row: short trace id, root/sub lineage, run
// state, live wall-clock, and whatever budget axes are actually seeded (a 0 axis is
// "never seeded" and is omitted, never fabricated as exhausted).
func guardInfoAgentText(s guardInfoSession) string {
	id := strings.TrimSpace(s.TraceID)
	if len(id) > 10 {
		id = id[:10]
	}
	if id == "" {
		id = "?"
	}
	role := "root"
	if strings.TrimSpace(s.ParentTrace) != "" {
		gen := s.Generation
		if gen < 1 {
			gen = 1
		}
		role = fmt.Sprintf("sub g%d", gen)
	}
	parts := []string{id, role}
	if run := strings.TrimSpace(s.Run); run != "" {
		parts = append(parts, run)
	}
	if s.ElapsedSeconds > 0 {
		parts = append(parts, humanUptime(float64(s.ElapsedSeconds)))
	}
	if s.TokensLeft > 0 {
		parts = append(parts, guardInfoShortCount(s.TokensLeft)+" tok left")
	}
	if s.TurnsLeft > 0 {
		parts = append(parts, fmt.Sprintf("%d turns left", s.TurnsLeft))
	}
	return strings.Join(parts, " · ")
}

// guardInfoAgentsSummary is the agents panel's one-row mini form (also reused by the
// compact status line): active count, sub-agent count, and the deepest spawn depth.
func guardInfoAgentsSummary(ss []guardInfoSession) string {
	subs, deepest := 0, 0
	for _, s := range ss {
		if strings.TrimSpace(s.ParentTrace) != "" {
			subs++
			gen := s.Generation
			if gen < 1 {
				gen = 1
			}
			if gen > deepest {
				deepest = gen
			}
		}
	}
	out := fmt.Sprintf("%d active", len(ss))
	if subs > 0 {
		out += fmt.Sprintf(" (%d sub, deepest g%d)", subs, deepest)
	}
	return out
}

// guardInfoBytesText renders a byte count at human scale (MB below 1 GiB, GB above),
// sized for a gutter row where "41MB" reads better than eight digits.
func guardInfoBytesText(b uint64) string {
	const mb = 1 << 20
	const gb = 1 << 30
	if b >= gb {
		return fmt.Sprintf("%.1fGB", float64(b)/gb)
	}
	if b >= 10*mb {
		return fmt.Sprintf("%.0fMB", float64(b)/mb)
	}
	return fmt.Sprintf("%.1fMB", float64(b)/mb)
}

// guardInfoShortCount renders a count compactly for a budget cell: 950 -> "950",
// 380000 -> "380k", 1200000 -> "1.2M".
func guardInfoShortCount(n int) string {
	switch {
	case n >= 1_000_000:
		return strings.TrimSuffix(fmt.Sprintf("%.1f", float64(n)/1_000_000), ".0") + "M"
	case n >= 1_000:
		return strings.TrimSuffix(fmt.Sprintf("%.1f", float64(n)/1_000), ".0") + "k"
	default:
		return fmt.Sprintf("%d", n)
	}
}

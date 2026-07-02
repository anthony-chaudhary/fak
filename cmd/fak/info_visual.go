package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// The visual `fak info` overlay — the default rendering for the 20% pane `fak guard --split`
// opens beside the agent. The single compact status line (renderGuardInfoLine) reads the turn
// economy at a glance, but a glance is all it gives: a number with no shape. The visual block
// turns that same payload-free /debug/vars feed into two stacked SUB-PANES that read like a
// task manager — one that shows the TREND of the economy (sparklines of savings, cache hit, and
// work over the last ~minute of ticks) and one that shows the live TASKS (gauge bars for cache
// hit and the safety counters). It is still a read-only poll: it adds zero new gateway reads,
// just a richer projection of the snapshot the line already fetched.
//
// Everything here is PURE (sparklineTUI / gaugeBarTUI / renderGuardInfoVisualBlock take values
// and return strings) except writeGuardInfoFrame, the thin multi-line in-place redraw the watch
// loop drives. The cell-width math reuses dispWidthTUI/trimTUI so a sparkline or gauge can never
// wrap a narrow split pane.

// guardInfoTrendCap bounds the per-series history the sparklines sample. At the 2s default tick
// this is ~96s of trend — enough to see a cache warming up or a burst of refusals without
// holding unbounded memory. The sparkline samples the TAIL, so a wider pane shows more history.
const guardInfoTrendCap = 48

// guardInfoSparkRunes is the 8-level unicode block ramp the sparkline draws with. Each rune is a
// single terminal cell (block-elements range), so a sparkline of N samples is exactly N cells.
var guardInfoSparkRunes = []rune("▁▂▃▄▅▆▇█")

// guardInfoTrend is the bounded ring of recent /debug/vars samples the visual block sparklines.
// Each series is capped to guardInfoTrendCap; push appends one tick and trims the oldest. It is
// the only state the overlay carries across ticks — the gateway stays the single source of truth.
type guardInfoTrend struct {
	cap      int
	saved    []float64 // net saved-token-equiv (the headline economic signal; can be negative)
	hit      []float64 // cache hit rate, 0..1
	turns    []float64 // cumulative replies (model turns) — its slope is the work rate
	inflight []float64 // requests in flight right now
	heap     []float64 // gateway heap-alloc bytes — the resources panel's live memory trend
}

// newGuardInfoTrend returns an empty trend ring with the given per-series cap (clamped to >=1).
func newGuardInfoTrend(capN int) *guardInfoTrend {
	if capN < 1 {
		capN = 1
	}
	return &guardInfoTrend{cap: capN}
}

// push records one tick's values into each series, trimming each to the cap (oldest dropped).
// A nil VCache (no provider cache activity yet) contributes a zero saving and zero hit, so the
// sparkline shows the pre-cache flat baseline rather than a gap.
func (t *guardInfoTrend) push(v guardInfoVars) {
	saved, hit := 0.0, 0.0
	if v.VCache != nil {
		saved = v.VCache.SavedTokenEquiv
		hit = v.VCache.HitRate
	}
	t.saved = appendCappedTUI(t.saved, saved, t.cap)
	t.hit = appendCappedTUI(t.hit, hit, t.cap)
	t.turns = appendCappedTUI(t.turns, float64(v.Inference.Turns), t.cap)
	t.inflight = appendCappedTUI(t.inflight, float64(v.Gateway.InflightRequests), t.cap)
	t.heap = appendCappedTUI(t.heap, float64(v.Runtime.Memory.HeapAllocBytes), t.cap)
}

// appendCappedTUI appends v to s and keeps only the last capN elements (a fixed-size tail ring).
func appendCappedTUI(s []float64, v float64, capN int) []float64 {
	s = append(s, v)
	if len(s) > capN {
		s = s[len(s)-capN:]
	}
	return s
}

// sparklineTUI renders the TAIL of vals as a unicode block sparkline at most width cells wide.
// It normalizes against the window's OWN min..max so the shape of the recent trend is visible
// regardless of absolute scale (a savings series in the thousands and a hit rate in 0..1 both
// fill the 8-level ramp). A flat series renders as a mid-height baseline rather than collapsing
// to the floor, so "steady" is distinguishable from "zero". Empty input or width<=0 -> "".
func sparklineTUI(vals []float64, width int) string {
	if width <= 0 || len(vals) == 0 {
		return ""
	}
	if len(vals) > width {
		vals = vals[len(vals)-width:]
	}
	min, max := vals[0], vals[0]
	for _, v := range vals {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	span := max - min
	last := len(guardInfoSparkRunes) - 1
	var b strings.Builder
	for _, v := range vals {
		idx := last / 2 // flat window: a mid baseline
		if span > 0 {
			idx = int((v-min)/span*float64(last) + 0.5)
		}
		if idx < 0 {
			idx = 0
		}
		if idx > last {
			idx = last
		}
		b.WriteRune(guardInfoSparkRunes[idx])
	}
	return b.String()
}

// gaugeBarTUI renders frac (clamped 0..1) as a width-cell horizontal bar: filled cells (█) for
// the proportion done, light cells (░) for the remainder — the task-manager gauge. Each glyph
// is one cell, so the bar is exactly width cells. width<=0 -> "".
func gaugeBarTUI(frac float64, width int) string {
	if width <= 0 {
		return ""
	}
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	fill := int(frac*float64(width) + 0.5)
	if fill > width {
		fill = width
	}
	return strings.Repeat("█", fill) + strings.Repeat("░", width-fill)
}

// guardInfoRuleTUI draws a sub-pane section header: "── label " padded with a horizontal rule to
// the pane width. It is the visual seam between the trends and tasks sub-panes. Trimmed (never
// over-drawn) on a pane too narrow to hold even the label.
func guardInfoRuleTUI(label string, width int) string {
	head := "── " + label + " "
	if width <= 0 {
		return head
	}
	if dispWidthTUI(head) >= width {
		return trimTUI(head, width)
	}
	return head + strings.Repeat("─", width-dispWidthTUI(head))
}

// renderGuardInfoVisualBlock projects one /debug/vars snapshot + the trend ring into the visual
// sub-pane block. The layout is composed from the guardInfoPanels() registry (info_panels.go):
// every registered panel that has something to say gets rows, and composeGuardInfoPanels fits
// them to the pane height — section rules dropped first, then panels degraded full→mini→hidden
// in degrade order — so the block always fits without scrolling, down to the 1-2 row tiny pane
// (the single compact status line). Every row is trimmed to the pane width so a sparkline or
// gauge can never wrap. The block is the in-place-redrawn frame the watch loop pins to the
// bottom of the pane.
func renderGuardInfoVisualBlock(v guardInfoVars, tr *guardInfoTrend, width, height int) string {
	if width <= 0 {
		width = 80
	}
	// Sparkline / gauge widths scale with the pane but stay bounded so the trailing label+value
	// always has room; on a narrow pane they shrink rather than push the value off-screen.
	ctx := guardInfoPanelCtx{
		v:      v,
		tr:     tr,
		width:  width,
		sparkW: clampIntTUI(width-26, 8, 28),
		gaugeW: clampIntTUI(width-28, 6, 20),
	}
	rows := composeGuardInfoPanels(ctx, guardInfoPanels(), height)
	// Defensive cap: never emit more rows than the pane can hold (keeps the redraw cursor math
	// exact even if a panel mis-sizes for an odd pane height).
	if height > 0 && len(rows) > height {
		rows = rows[:height]
	}
	// Cap each row to the pane width with takeCellsTUI (NOT trimTUI): the gutter labels, gauges,
	// and sparklines align on intentional internal spacing, which trimTUI's whitespace-collapse
	// would destroy. takeCellsTUI truncates to the cell budget without touching interior spacing.
	for i, r := range rows {
		rows[i] = takeCellsTUI(r, width)
	}
	return strings.Join(rows, "\n")
}

// guardInfoVisualIdentityRow is the block's header: which fak this pane watches, how long it has
// run, and the live liveness (replies / in-flight) — the persistent identity the scrolled-off
// startup banner can no longer give.
func guardInfoVisualIdentityRow(v guardInfoVars) string {
	return fmt.Sprintf("%s · ↑%s · replies %d · busy %d",
		guardInfoVersionTag(), humanUptime(v.Gateway.UptimeSeconds), v.Inference.Turns, v.Gateway.InflightRequests)
}

// guardInfoVisualTinyRow is the 1-row fallback for a pane too short for any sub-pane: the compact
// status line, so even a sliver pane still shows the economy + safety in plain words.
func guardInfoVisualTinyRow(v guardInfoVars) string {
	return renderGuardInfoLine(v)
}

// guardInfoSafetyText is the safety sub-pane's value (the "safety" label is the row gutter): the
// plain-words floor summary without renderGuardInfoLine's "safety: " prefix.
func guardInfoSafetyText(v guardInfoVars) string {
	return strings.TrimPrefix(
		guardFloorSafetyWord(v.Kernel.Denies, v.Kernel.Transforms, v.Kernel.Quarantines, v.Kernel.ResultDenies),
		"safety: ")
}

func guardInfoIncidentText(v guardInfoVars) string {
	var parts []string
	if s := guardInfoUpstreamErrorsText(v.Upstream.ErrorsByKind); s != "" {
		parts = append(parts, "upstream "+s)
	}
	if s := guardInfoAuthRefreshText(v.Upstream.AuthRefreshByOutcome); s != "" {
		parts = append(parts, "auth-refresh "+s)
	}
	if v.Upstream.Retries > 0 {
		parts = append(parts, fmt.Sprintf("retries x%d", v.Upstream.Retries))
	}
	return strings.Join(parts, "; ")
}

func guardInfoUpstreamErrorsText(counts map[string]uint64) string {
	if len(counts) == 0 {
		return ""
	}
	order := []string{"auth", "rate_limited", "forbidden", "stalled", "status_5xx", "overloaded", "unreachable", "oom", "status_4xx", "other"}
	seen := map[string]bool{}
	var parts []string
	for _, kind := range order {
		seen[kind] = true
		if n := counts[kind]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s x%d", guardInfoUpstreamKindLabel(kind), n))
		}
	}
	var extra []string
	for kind, n := range counts {
		if n > 0 && !seen[kind] {
			extra = append(extra, kind)
		}
	}
	sort.Strings(extra)
	for _, kind := range extra {
		parts = append(parts, fmt.Sprintf("%s x%d", guardInfoUpstreamKindLabel(kind), counts[kind]))
	}
	return strings.Join(parts, ", ")
}

func guardInfoAuthRefreshText(counts map[string]uint64) string {
	if len(counts) == 0 {
		return ""
	}
	var parts []string
	for _, outcome := range []string{"exhausted", "recovered"} {
		if n := counts[outcome]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s x%d", outcome, n))
		}
	}
	return strings.Join(parts, ", ")
}

func guardInfoUpstreamKindLabel(kind string) string {
	switch strings.TrimSpace(kind) {
	case "auth":
		return "auth/401"
	case "rate_limited":
		return "rate_limited/429"
	case "forbidden":
		return "forbidden/403"
	case "stalled":
		return "stalled/504"
	case "status_5xx":
		return "status_5xx/5xx"
	case "overloaded":
		return "overloaded/529"
	default:
		return kind
	}
}

// guardInfoSaved / guardInfoHitPct / guardInfoHitFrac / guardInfoMult / guardInfoSavingWord pull
// the cache fields with a nil-VCache (no provider cache activity yet) reading as the honest zero.
func guardInfoSaved(v guardInfoVars) float64 {
	if v.VCache != nil {
		return v.VCache.SavedTokenEquiv
	}
	return 0
}

func guardInfoHitFrac(v guardInfoVars) float64 {
	if v.VCache != nil {
		return v.VCache.HitRate
	}
	return 0
}

func guardInfoHitPct(v guardInfoVars) float64 { return guardInfoHitFrac(v) * 100 }

func guardInfoMult(v guardInfoVars) float64 {
	if v.VCache != nil {
		return v.VCache.Multiplier
	}
	return 0
}

// guardInfoSavingWord is the cache gauge's plain-words verdict: whether re-using text has paid off
// yet (the same three states the status line uses), so the gauge bar carries a meaning, not just a
// number.
func guardInfoSavingWord(v guardInfoVars) string {
	if v.VCache == nil {
		return "no cache yet"
	}
	if strings.EqualFold(strings.TrimSpace(v.VCache.Status), "PROVEN") {
		return "saving money"
	}
	return "not saving yet"
}

// guardInfoVisualIntro is the one-time line printed above the live visual block: what the pane is
// and how to stop it. It scrolls into history while the block redraws in place below it, so the
// "Ctrl-C to stop" hint stays discoverable without the block having to spend a row on it forever.
func guardInfoVisualIntro(base string, interval time.Duration, width int) string {
	line := fmt.Sprintf("fak info · live sub-panes · %s · every %s · Ctrl-C to stop", base, interval)
	if width > 0 {
		line = trimTUI(line, width)
	}
	return line + "\n"
}

// writeGuardInfoFrame draws a multi-line block in place on a TTY and returns its row count for the
// next call. prevRows is the previous frame's row count (0 = first paint, no cursor move). It moves
// the cursor up to the top of the previous block, clears from there to the end of the pane, and
// reprints — so a block of stable height stays pinned to the bottom of the pane and redraws cleanly
// each tick (the multi-line analogue of the single-line \r\033[K redraw). It writes NO trailing
// newline, leaving the cursor parked at the end of the last row (the "dirty" invariant the loop's
// note/exit paths break with a newline).
func writeGuardInfoFrame(w io.Writer, block string, prevRows int) int {
	lines := strings.Split(block, "\n")
	if prevRows > 0 {
		if prevRows > 1 {
			fmt.Fprintf(w, "\033[%dA", prevRows-1) // up to the first row of the previous block
		}
		fmt.Fprint(w, "\r\033[J") // column 0, then clear from here to the end of the pane
	}
	fmt.Fprint(w, strings.Join(lines, "\n"))
	return len(lines)
}

// clampIntTUI clamps v to [lo, hi] (lo wins if lo>hi).
func clampIntTUI(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

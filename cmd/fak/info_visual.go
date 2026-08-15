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
	cap             int
	baseline        guardInfoWorkDoneBaseline
	baselineChanges uint64
	saved           []float64 // net saved-token-equiv (the headline economic signal; can be negative)
	hit             []float64 // cache hit rate, 0..1
	turns           []float64 // cumulative replies (model turns) — its slope is the work rate
	inflight        []float64 // requests in flight right now
	heap            []float64 // gateway heap-alloc bytes — the resources panel's live memory trend
	savedCalls      []float64 // cumulative engine calls fak avoided (turns saved) — its slope is the saving rate
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
	baseline := guardInfoWorkDoneFromVars(v).Baseline
	if t.baseline.ID == "" {
		t.baseline = baseline
	} else if !guardInfoWorkDoneBaselineCompatible(t.baseline, baseline) {
		t.baseline = baseline
		t.baselineChanges++
		t.saved, t.hit, t.turns, t.inflight, t.heap, t.savedCalls = nil, nil, nil, nil, nil, nil
	}
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
	t.savedCalls = appendCappedTUI(t.savedCalls, float64(guardInfoTurnsSaved(v)), t.cap)
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
	ctx := newGuardInfoPanelCtx(v, tr, width)
	rows := composeGuardInfoPanels(ctx, guardInfoPanels(), height)
	// Height-cap, width-cap and join: see joinPaneRowsTUI for why the width cap is
	// takeCellsTUI rather than trimTUI (this pane's gauges and sparklines align on interior
	// spacing that a whitespace-collapsing trim would destroy).
	return joinPaneRowsTUI(rows, width, height)
}

// guardInfoVisualIdentityRow is the block's header: which fak this pane watches, how long it has
// run, and the live liveness (replies / in-flight) — the persistent identity the scrolled-off
// startup banner can no longer give.
func guardInfoVisualIdentityRow(v guardInfoVars) string {
	local := fmt.Sprintf("%s · ↑%s · replies %d · busy %d",
		guardInfoVersionTag(), humanUptime(v.Gateway.UptimeSeconds), v.Inference.Turns, v.Gateway.InflightRequests)
	// Fleet posture leads the pinned row when guard publishes it. On narrow terminals the
	// width cap therefore preserves the cross-machine status instead of truncating it behind
	// the local gateway identity; the expanded Agents tab carries the complete machine sample.
	if fleet := guardInfoFleetSummary(v.Fleet); fleet != "" {
		return fleet + " · local " + local
	}
	return local
}

// guardInfoVisualTinyRow is the 1-row fallback for a pane too short for any sub-pane: the compact
// status line, so even a sliver pane still shows the economy + safety in plain words.
func guardInfoVisualTinyRow(v guardInfoVars) string {
	w := guardInfoWorkDoneFromVars(v)
	if w.Metrics.InputTokensAvoided.Available || w.Metrics.ModelCallsAvoided.Available {
		tokens := "tok unavailable"
		if w.Metrics.InputTokensAvoided.Available {
			tokens = guardInfoSignedShortCount(w.Metrics.InputTokensAvoided.Value) + " input tok"
		}
		calls := "calls unavailable"
		if w.Metrics.ModelCallsAvoided.Available {
			calls = guardInfoShortCount(int(w.Metrics.ModelCallsAvoided.Value)) + " calls"
		}
		line := "work vs direct provider · " + tokens + " avoided · " + calls + " avoided · " + guardSafetyWord(v)
		if len(v.Sessions) > 0 {
			line += " · agents " + guardInfoAgentsSummary(v.Sessions)
		}
		return line
	}
	return renderGuardInfoLine(v)
}

// guardInfoSafetyText is the safety sub-pane's value (the "safety" label is the row gutter): the
// plain-words floor summary without renderGuardInfoLine's "safety: " prefix.
func guardInfoSafetyText(v guardInfoVars) string {
	return strings.TrimPrefix(guardSafetyWord(v), "safety: ")
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

// guardInfoTurnsSaved is the session's "turns saved": the count of engine calls fak
// spared the agent this session — vDSO memo hits plus inline-served turns — the SAME
// FakVDSOAvoidedCalls the guard exit summary prints as "vDSO N avoided call(s)". It is
// WITNESSED/fak-authored (its witness is skipped engine calls, not provider-relayed
// tokens), and reads the honest zero when the gateway has not reported the cache-
// attribution block yet (nil pointer), so the trends panel stays silent on a session
// that avoided nothing rather than fabricating a saving.
func guardInfoTurnsSaved(v guardInfoVars) uint64 {
	if v.CacheAttribution != nil && v.CacheAttribution.FakVDSOAvoidedCalls > 0 {
		return v.CacheAttribution.FakVDSOAvoidedCalls
	}
	if v.Adjudication == nil {
		return 0
	}
	return v.Adjudication.Transformed
}

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

// padBlockToHeight appends empty rows so the rendered block is exactly height rows tall — a no-op
// when height <= 0 (unknown pane) or the block already meets/exceeds it. The interactive overlay
// pads every PAINTED frame (info.go's writeFrame) to the pane height so the in-place block always
// bottom-parks: it fills the bottom `height` rows of the pane on every tick, whatever view is
// active. That constant geometry is what keeps blockRelativeRow's absolute→block-row translation
// exact. Without it, a frame shorter than the tallest one drawn so far (e.g. after switching from
// the full Overview to a stubby Safety view) anchors HIGHER than height-prevRows, and every tab /
// chip click silently mis-hits or goes inert — the "clicks work at first, then stop" bug. The pad
// rows are the same blank cells the redraw's \033[J clear already leaves below the content, so a
// padded frame is visually identical to an unpadded one; only the parked-block height changes.
func padBlockToHeight(block string, height int) string {
	if height <= 0 {
		return block
	}
	rows := strings.Count(block, "\n") + 1
	if rows >= height {
		return block
	}
	return block + strings.Repeat("\n", height-rows)
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

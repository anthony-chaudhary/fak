package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

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
	ctx := newGuardInfoPanelCtx(v, tr, width)
	ctx.cacheMech = state.cacheMech // expand the clicked ablation mechanism's detail sub-panel
	topRow := buildInfoTabBar(state.active, state.glossaryOpen).text
	if state.copyMode {
		topRow = buildInfoCopyBanner(width) // frozen for copy: swap the tab bar for the how-to banner
	}
	rows := []string{topRow}
	if state.launchNotice != "" {
		rows = append(rows, state.launchNotice)
	}
	// Shared source/provenance context belongs to the overview. Focused tabs lead
	// with their own subsystem so switching pages changes the frame immediately;
	// renderers that need source truth (notably Cache) carry it in their body.
	if state.active == viewOverview && !state.glossaryOpen {
		rows = append(rows, guardInfoObservationRows(v.Observation)...)
	}

	// bodyHeight is the room left under the tab bar; 0/negative height (unknown pane) means
	// "roomy", so the body renders in full and the loop's own cap pins it.
	bodyHeight := 0
	if height > 0 {
		bodyHeight = height - len(rows)
	}
	var body []string
	if state.glossaryOpen {
		body = renderInfoGlossaryBody(state.glossaryTerm, width)
	} else {
		// The active view scrolls: build its full un-degraded rows, then window them to the
		// body height at the view's stored offset. The overview pins its identity row (#3778);
		// the focused views pin nothing.
		full, pinned := infoViewFullRows(state.active, ctx)
		body, _ = scrollInfoWindow(full, pinned, state.scroll[state.active], bodyHeight)
	}
	rows = append(rows, body...)

	return joinPaneRowsTUI(rows, width, height)
}

// buildInfoCopyBanner is the one-line banner that replaces the tab bar while copy/freeze mode is
// active. It states plainly that the pane is frozen and how to select/copy and resume, so a
// watcher who pressed 'c' is never left guessing why the frame stopped updating. A trailing rule
// fills the row so the banner reads as a distinct mode line; the caller width-caps every row, so
// a narrow pane trims the tail rather than wrapping.
func buildInfoCopyBanner(width int) string {
	if width <= 0 {
		width = 80
	}
	msg := "COPY MODE — drag to select, copy with your terminal · c or Ctrl-C to resume "
	if pad := width - dispWidthTUI(msg); pad > 0 {
		msg += strings.Repeat("─", pad)
	}
	return takeCellsTUI(msg, width)
}

// newGuardInfoPanelCtx builds the panel render context for a pane of the given width, scaling
// the sparkline/gauge widths to the pane but keeping them bounded so the trailing label+value
// always has room. Shared by the visual block, the interactive block, and the scroll clamp so
// all three size their content identically.
func newGuardInfoPanelCtx(v guardInfoVars, tr *guardInfoTrend, width int) guardInfoPanelCtx {
	if width <= 0 {
		width = 80
	}
	return guardInfoPanelCtx{
		v:      v,
		tr:     tr,
		width:  width,
		sparkW: clampIntTUI(width-26, 8, 28),
		gaugeW: clampIntTUI(width-28, 6, 20),
	}
}

// infoViewFullRows builds one view's complete, un-windowed body rows plus the count of leading
// rows pinned to the top (never scrolled). The overview pins its identity row and scrolls the
// full panel stack at full detail (#3778 — the scroll model replaces the overview's degradation
// in the interactive pane); the focused views pin nothing and show one subsystem uncapped, the
// list the overview's per-panel caps exist to lift.
func infoViewFullRows(view infoView, ctx guardInfoPanelCtx) (full []string, pinned int) {
	switch view {
	case viewAgents:
		return renderInfoAgentsView(ctx.v), 0
	case viewFleet:
		return fleetWorkspaceRows(ctx.v), 0
	case viewEndpoints:
		return renderInfoEndpointsView(ctx), 0
	case viewCache:
		return renderInfoCacheView(ctx), 0
	case viewSafety:
		return renderInfoSafetyView(ctx.v), 0
	case viewStartup:
		return startupViewRows(ctx.v), 0
	default: // viewOverview
		return guardInfoRoomyPanelRows(ctx), 1
	}
}

// guardInfoRoomyPanelRows is the overview's full, un-degraded content: the identity row, then
// every non-silent panel at full detail with its section rule — exactly the layout
// composeGuardInfoPanels emits when the pane is tall enough for everything. The interactive
// overview scrolls THIS (identity row pinned) instead of degrading panels to fit, so an operator
// can page through every subsystem at full detail. The non-interactive visual block still
// degrades via composeGuardInfoPanels — a piped, non-interactive frame cannot scroll.
func guardInfoRoomyPanelRows(ctx guardInfoPanelCtx) []string {
	out := []string{guardInfoVisualIdentityRow(ctx.v)}
	for _, p := range guardInfoPanels() {
		full := p.rows(ctx, guardPanelFull)
		if len(full) == 0 {
			continue // a silent panel this tick costs nothing
		}
		out = append(out, guardInfoRuleTUI(p.name, ctx.width))
		out = append(out, full...)
	}
	return out
}

// scrollInfoWindow renders full into a window of height lines, scrolled so that offset scrollable
// rows are hidden above the visible slice, and returns the visible rows plus the clamped offset
// actually used (so the loop can persist the clamp and never drift past the ends). The first
// pinned rows are an anchor: always shown at the top, never scrolled. When the content fits,
// every row shows and the clamped offset is 0 (so a roomy pane is byte-identical to the pre-scroll
// render). When it does not, an "↑ N more above" / "↓ N more below" indicator marks the hidden
// rows on each side that has them. Pure: no I/O, no TTY.
func scrollInfoWindow(full []string, pinned, offset, height int) (rows []string, clamped int) {
	if height <= 0 || len(full) <= height { // roomy/unknown, or it already fits — no window
		return full, 0
	}
	if pinned < 0 {
		pinned = 0
	}
	if pinned > len(full) {
		pinned = len(full)
	}
	if height <= pinned { // too short even for the pinned prefix: show what fits, top-down
		return append([]string(nil), full[:height]...), 0
	}
	pinnedRows := full[:pinned]
	scrollable := full[pinned:]
	s := len(scrollable)
	avail := height - pinned          // lines for indicators + scrollable content; avail >= 1, s > avail
	maxOffset := maxInt(0, s-avail+1) // at maxOffset the window ends at s with an above indicator
	offset = clampIntTUI(offset, 0, maxOffset)

	above := offset > 0
	slots := avail
	if above {
		slots-- // an "↑ more above" line
	}
	below := offset+slots < s
	if below {
		slots-- // a "↓ more below" line
	}
	if slots < 0 {
		slots = 0
	}
	end := offset + slots
	if end > s {
		end = s
	}
	out := make([]string, 0, height)
	out = append(out, pinnedRows...)
	if above {
		out = append(out, fmt.Sprintf(" ↑ %d more above", offset))
	}
	out = append(out, scrollable[offset:end]...)
	if below {
		out = append(out, fmt.Sprintf(" ↓ %d more below", s-end))
	}
	return out, offset
}

// clampInfoScrollToSample pins the active view's stored scroll offset to the content that view
// currently renders at (width,height), so an offset set past the end (the End key, a page step,
// or content that shrank) is pulled back to the last page instead of drifting. The loop calls it
// after applyInfoInput so the stored offset the NEXT keystroke reads is already honest. The
// glossary overlay does not scroll, so it is returned unchanged. Pure.
func clampInfoScrollToSample(s infoViewState, v guardInfoVars, tr *guardInfoTrend, width, height int) infoViewState {
	if s.glossaryOpen {
		return s
	}
	ctx := newGuardInfoPanelCtx(v, tr, width)
	ctx.cacheMech = s.cacheMech // count the expanded mechanism's detail rows when clamping the scroll
	bodyHeight := 0
	if height > 0 {
		bodyHeight = height - 1
	}
	full, pinned := infoViewFullRows(s.active, ctx)
	_, clamped := scrollInfoWindow(full, pinned, s.scroll[s.active], bodyHeight)
	s.scroll[s.active] = clamped
	return s
}

// renderInfoAgentsView is the expanded Agents view: a fleet-summary header, then one full row per
// live session (main + every sub-agent, with lineage/run-state/wall-clock/budget/activity). It
// shows EVERY session — the overview's 4-row cap is exactly what this view exists to lift.
func renderInfoAgentsView(v guardInfoVars) []string {
	rows := renderInfoFleetRows(v.Fleet)
	if len(v.Sessions) == 0 {
		if v.Observation != nil {
			return append(rows, " agents: "+guardInfoObservationMetricText("sessions", v.Observation.Sessions))
		}
		return append(rows, " agents: none running (no session registry wired, or nothing live)")
	}
	rows = append(rows, " agents: "+guardInfoAgentsSummary(v.Sessions))
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
// the raw savings figure, the live per-mechanism cache ABLATION (what each caching mechanism is
// saving this session — the live twin of the offline `fak ablate` concept bars, sourced from this
// session's own witnessed counters), and the trend sparklines — the whole cache economy in one
// place instead of split across the overview's trends + tasks panels.
func renderInfoCacheView(ctx guardInfoPanelCtx) []string {
	v := ctx.v
	if zeroObservationGap(v.Observation) {
		// This diagnosis is Cache content, not global chrome. Keep the typed cause
		// and next check here without making every focused tab repeat it.
		return guardInfoObservationRows(v.Observation)
	}
	cacheRow := " cache  " + guardInfoSavingWord(v)
	if guardInfoCacheSourceObserved(v) {
		cacheRow = fmt.Sprintf(" cache  %s %.0f%%  %s", gaugeBarTUI(guardInfoHitFrac(v), ctx.gaugeW), guardInfoHitPct(v), guardInfoSavingWord(v))
	}
	if split := guardInfoCacheAttributionText(v); split != "" {
		cacheRow += " · " + split
	}
	w := guardInfoWorkDoneFromVars(v)
	savedRow := fmt.Sprintf(" saved  %s tokens so far · vs %s r%d", signedTokens(guardInfoSaved(v)), w.Baseline.Label, w.Baseline.Revision)
	if !guardInfoCacheSourceObserved(v) {
		savedRow = " saved  " + guardInfoSavingWord(v)
	}
	rows := []string{cacheRow, savedRow}
	rows = append(rows, guardInfoWorkDoneBaselineDetailRows(w)...)
	rows = append(rows, guardInfoWorkDoneSourceRows(w)...)
	rows = append(rows, renderInfoCacheAblationRows(ctx)...)
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

// startupViewRows is the post-ready home for one-shot startup detail. It
// renders only the structured /debug/vars.startup block, so a launch can stay quiet
// without making the load mode, phase profile, warnings, or guard report ephemeral.
func startupViewRows(v guardInfoVars) []string {
	s := v.Startup
	if s == nil {
		return []string{" gateway startup: not reported (older gateway)"}
	}
	status := strings.ToUpper(strings.TrimSpace(s.Status))
	if status == "" {
		status = "UNKNOWN"
	}
	header := " gateway startup: " + status
	if s.TimeToReadySeconds > 0 {
		header += " · ready in " + formatStartupDuration(s.TimeToReadySeconds)
	}
	if s.UnaccountedSeconds > 0 {
		header += " · " + formatStartupDuration(s.UnaccountedSeconds) + " unaccounted"
	}
	rows := []string{header}
	if s.StartedAt != "" || s.ReadyAt != "" {
		rows = append(rows, fmt.Sprintf(" timeline: started %s · ready %s", formatStartupInstant(s.StartedAt), formatStartupInstant(s.ReadyAt)))
	}
	if len(s.Phases) > 0 {
		rows = append(rows, " phases:")
		for _, ph := range s.Phases {
			rows = append(rows, fmt.Sprintf("  %-28s %9s · %s · %s", ph.Name, formatStartupDuration(ph.Seconds), ph.Provenance, ph.Stage))
		}
	}
	if m := s.ModelLoad; m != nil {
		summary := fmt.Sprintf(" model load: %s · %s", emptyAs(m.Mode, "unknown mode"), formatStartupDuration(m.TotalSeconds))
		if m.Tensors > 0 {
			summary += fmt.Sprintf(" · %d tensors", m.Tensors)
		}
		if m.Bytes > 0 {
			summary += " · " + formatStartupBytes(m.Bytes)
		}
		if m.Bottleneck != "" {
			summary += " · bottleneck " + m.Bottleneck
		}
		rows = append(rows, summary)
		if m.Source != "" {
			rows = append(rows, "  source: "+m.Source)
		}
		for _, ph := range m.Phases {
			detail := fmt.Sprintf("  load phase %-20s %9s", ph.Phase, formatStartupDuration(ph.Seconds))
			if ph.Tensors > 0 {
				detail += fmt.Sprintf(" · %d tensors", ph.Tensors)
			}
			if ph.Bytes > 0 {
				detail += " · " + formatStartupBytes(ph.Bytes)
			}
			rows = append(rows, detail)
		}
		for _, path := range m.LoadPaths {
			rows = append(rows, fmt.Sprintf("  load path %-8s %-6s resident %d/%s · dequant %d/%s",
				path.QuantType, path.Class, path.ResidentTensors, formatStartupBytes(path.ResidentBytes), path.DequantTensors, formatStartupBytes(path.DequantBytes)))
		}
	}
	if len(s.Messages) > 0 {
		rows = append(rows, " startup messages:")
		for _, message := range s.Messages {
			label := strings.Trim(strings.Join([]string{message.Level, message.Source, message.Kind}, "/"), "/")
			lines := strings.Split(strings.TrimSpace(message.Text), "\n")
			for i, line := range lines {
				if strings.TrimSpace(line) == "" {
					continue
				}
				if i == 0 {
					rows = append(rows, "  "+label+": "+line)
				} else {
					rows = append(rows, "    "+line)
				}
			}
		}
	}
	return rows
}

func formatStartupDuration(seconds float64) string {
	if seconds <= 0 {
		return "0s"
	}
	d := time.Duration(seconds * float64(time.Second))
	if d < time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	return d.Round(time.Millisecond).String()
}

func formatStartupInstant(raw string) string {
	if raw == "" {
		return "pending"
	}
	if parsed, err := time.Parse(time.RFC3339Nano, raw); err == nil {
		return parsed.Local().Format("15:04:05.000")
	}
	return raw
}

func formatStartupBytes(bytes int64) string {
	if bytes <= 0 {
		return "0 B"
	}
	return guardInfoBytesText(uint64(bytes))
}

func emptyAs(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
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

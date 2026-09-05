package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// richVisualVars is provenVisualVars plus the live-usage blocks the new panels render:
// a runtime resource block and a two-session registry (an agent + its continuation).
func richVisualVars() guardInfoVars {
	v := provenVisualVars()
	v.Runtime.NumGoroutine = 24
	v.Runtime.Memory.HeapAllocBytes = 41 << 20
	v.Runtime.Memory.SysBytes = 68 << 20
	v.Runtime.Memory.NumGC = 12
	v.Inference.OutputTokensPerSecond = 12.3
	v.Inference.MeanTTFTSeconds = 1.2
	v.Inference.InflightMaxAgeSeconds = 3.4
	v.Sessions = []guardInfoSession{
		{TraceID: "main-trace-long", Run: "running", TokensLeft: 380_000, TurnsLeft: 7, ElapsedSeconds: 95},
		// A CONTINUATION of the row above, which is what a ParentTrace means: the same
		// agent re-continued under a fresh trace after a budget reset. Not a sub-agent.
		{TraceID: "cont-trace", Run: "running", ParentTrace: "main-trace-long", Generation: 1},
	}
	return v
}

// TestRenderGuardInfoVisualBlockResourcesAndAgents proves the pane carries the LIVE
// info the entry/exit summaries used to monopolize: the gateway's own resource usage
// (heap + sparkline, goroutines, gc, generation rate) and the per-session agent rows
// (root + continuation lineage, wall-clock, remaining budget) — each under its own
// section rule at roomy height.
func TestRenderGuardInfoVisualBlockResourcesAndAgents(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < 6; i++ {
		v := richVisualVars()
		v.Runtime.Memory.HeapAllocBytes = uint64(30+i) << 20 // a rising heap trend
		tr.push(v)
	}
	block := renderGuardInfoVisualBlock(richVisualVars(), tr, 140, 0 /*roomy*/)

	for _, want := range []string{
		"── resources ", "── agents ", // the new section rules
		" mem    ",                        // resources gutter label
		"41MB heap", "68MB sys", "24 gor", // live resource axes
		"gc 12",                      //
		" rate   ", "12.3 tok/s out", // generation rate
		"ttft 1.20s", "oldest req 3s", // latency + hung-request tell
		" agent  ",          // agents gutter label
		"main-trace · root", // the root session (trace id capped at 10)
		"running", "1m35s",  // run state + live wall-clock
		"380k tok left", "7 turns left", // remaining budget axes
		"cont-trace · cont g1", // the continuation lineage row
	} {
		if !strings.Contains(block, want) {
			t.Errorf("visual block missing %q:\n%s", want, block)
		}
	}
	// The pre-existing sub-panes must still be there — growth is additive.
	for _, want := range []string{"── trends ", "── tasks ", " save  ", " cache  ", " safety "} {
		if !strings.Contains(block, want) {
			t.Errorf("visual block lost pre-existing section %q:\n%s", want, block)
		}
	}
}

// TestRenderGuardInfoVisualBlockCarriesTurnsSaved proves the turns-saved row survives
// the full composer path into the roomy visual block, under the trends section rule.
func TestRenderGuardInfoVisualBlockCarriesTurnsSaved(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := provenVisualVars()
	v.CacheAttribution = &guardInfoCacheAttribution{FakVDSOAvoidedCalls: 4}
	for i := 0; i < 6; i++ {
		tr.push(v)
	}
	block := renderGuardInfoVisualBlock(v, tr, 120, 0 /*roomy*/)
	for _, want := range []string{"── trends ", " saved ", "4 calls avoided"} {
		if !strings.Contains(block, want) {
			t.Errorf("visual block missing turns-saved element %q:\n%s", want, block)
		}
	}
}

// TestGuardInfoPanelsSilentWithoutData pins the zero-cost contract for absent data: a
// snapshot with no runtime block and no sessions renders NO resources/agents rows, so
// old gateways and bare fixtures keep the original two-sub-pane layout byte-for-byte.
func TestGuardInfoPanelsSilentWithoutData(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < 3; i++ {
		tr.push(provenVisualVars())
	}
	block := renderGuardInfoVisualBlock(provenVisualVars(), tr, 120, 0)
	for _, banned := range []string{"── resources ", "── agents ", " mem    ", " agent  "} {
		if strings.Contains(block, banned) {
			t.Errorf("panel must stay silent without data, found %q:\n%s", banned, block)
		}
	}
}

// TestComposeGuardInfoPanelsDegrades proves the composer's shrink ladder: at a height
// too short for every panel at full size, panels fold to their one-row mini forms in
// degrade order — and every produced layout still fits the height budget exactly.
func TestComposeGuardInfoPanelsDegrades(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := richVisualVars()
	for i := 0; i < 3; i++ {
		tr.push(v)
	}
	ctx := guardInfoPanelCtx{v: v, tr: tr, width: 120, sparkW: 12, gaugeW: 10}

	// Roomy: everything full, with rules.
	roomy := composeGuardInfoPanels(ctx, guardInfoPanels(), 0)
	joined := strings.Join(roomy, "\n")
	for _, want := range []string{"── trends ", "── agents ", " res", " agent  "} {
		if !strings.Contains(joined, want) {
			t.Errorf("roomy compose missing %q:\n%s", want, joined)
		}
	}

	// Tight: still every panel present, but folded — the agents panel must appear as
	// its one-row summary rather than vanishing while lower-value rows survive.
	for _, height := range []int{5, 6, 7, 8} {
		rows := composeGuardInfoPanels(ctx, guardInfoPanels(), height)
		if len(rows) > height {
			t.Fatalf("h=%d: composed %d rows, exceeds budget:\n%s", height, len(rows), strings.Join(rows, "\n"))
		}
	}
	tight := strings.Join(composeGuardInfoPanels(ctx, guardInfoPanels(), 6), "\n")
	if !strings.Contains(tight, "agents 2 active (1 continued, deepest g1)") {
		t.Errorf("tight compose must keep the agents mini summary:\n%s", tight)
	}

	// Tiny: the single compact status line, which also carries the agents summary.
	tiny := composeGuardInfoPanels(ctx, guardInfoPanels(), 2)
	if len(tiny) != 1 {
		t.Fatalf("h=2 must compose exactly the tiny row, got %d rows", len(tiny))
	}
	if !strings.Contains(tiny[0], "agents 2 active") {
		t.Errorf("tiny row must carry the agents summary: %q", tiny[0])
	}
}

// TestGuardInfoAgentText pins the per-session row grammar: trace ids cap at 10 chars,
// a parent trace makes a CONTINUATION row (generation floored at 1), zero budget axes
// are omitted (never rendered as exhausted), and empty ids degrade to "?".
func TestGuardInfoAgentText(t *testing.T) {
	root := guardInfoAgentText(guardInfoSession{TraceID: "abcdefghijKLMNOP", Run: "running", TokensLeft: 1_200_000, ElapsedSeconds: 61})
	for _, want := range []string{"abcdefghij", "root", "running", "1m1s", "1.2M tok left"} {
		if !strings.Contains(root, want) {
			t.Errorf("root row missing %q: %q", want, root)
		}
	}
	if strings.Contains(root, "turns left") {
		t.Errorf("unseeded turns axis must be omitted: %q", root)
	}
	cont := guardInfoAgentText(guardInfoSession{TraceID: "cont", Run: "paused", ParentTrace: "abc"})
	if !strings.Contains(cont, "cont g1") || !strings.Contains(cont, "paused") {
		t.Errorf("continuation row must carry lineage + run state: %q", cont)
	}
	if got := guardInfoAgentText(guardInfoSession{}); !strings.HasPrefix(got, "?") {
		t.Errorf("empty session must degrade to ?, got %q", got)
	}
}

// TestGuardInfoAgentsPanelCapsRows proves a wide sub-agent fan-out folds into "+N
// more" instead of scrolling the pane.
func TestGuardInfoAgentsPanelCapsRows(t *testing.T) {
	var v guardInfoVars
	for i := 0; i < 7; i++ {
		v.Sessions = append(v.Sessions, guardInfoSession{TraceID: "t", Run: "running"})
	}
	rows := guardInfoAgentsPanelRows(guardInfoPanelCtx{v: v}, guardPanelFull)
	if len(rows) != guardInfoAgentsMaxRows+1 {
		t.Fatalf("7 sessions must render %d rows + overflow, got %d", guardInfoAgentsMaxRows, len(rows))
	}
	if !strings.Contains(rows[len(rows)-1], "+3 more") {
		t.Errorf("overflow row must fold the remainder: %q", rows[len(rows)-1])
	}
}

// TestGuardInfoShortCount and TestGuardInfoBytesText pin the compact number grammar
// the new rows depend on.
func TestGuardInfoShortCount(t *testing.T) {
	for _, tc := range []struct {
		n    int
		want string
	}{{950, "950"}, {1_000, "1k"}, {380_000, "380k"}, {1_200_000, "1.2M"}, {2_000_000, "2M"}} {
		if got := guardInfoShortCount(tc.n); got != tc.want {
			t.Errorf("shortCount(%d) = %q, want %q", tc.n, got, tc.want)
		}
	}
}

func TestGuardInfoBytesText(t *testing.T) {
	for _, tc := range []struct {
		b    uint64
		want string
	}{{5 << 20, "5.0MB"}, {41 << 20, "41MB"}, {3 << 30, "3.0GB"}} {
		if got := guardInfoBytesText(tc.b); got != tc.want {
			t.Errorf("bytesText(%d) = %q, want %q", tc.b, got, tc.want)
		}
	}
}

// TestGuardInfoTurnsSaved pins the "turns saved" projection: nil attribution reads the
// honest zero, and a reported FakVDSOAvoidedCalls is surfaced verbatim.
func TestGuardInfoTurnsSaved(t *testing.T) {
	var v guardInfoVars
	if got := guardInfoTurnsSaved(v); got != 0 {
		t.Errorf("nil attribution must read 0 turns saved, got %d", got)
	}
	v.CacheAttribution = &guardInfoCacheAttribution{FakVDSOAvoidedCalls: 7}
	if got := guardInfoTurnsSaved(v); got != 7 {
		t.Errorf("turns saved = %d, want 7", got)
	}
}

// TestGuardInfoTrendsSavedCalls proves the trends panel grows a "saved" row surfacing
// the fak-authored turns saved (avoided engine calls) when the gateway reports them,
// stays silent — the panels' zero-cost contract — when nothing was avoided, and keeps
// the mini form the single headline save row either way.
func TestGuardInfoTrendsSavedCalls(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := provenVisualVars()
	v.CacheAttribution = &guardInfoCacheAttribution{FakVDSOAvoidedCalls: 3}
	for i := 0; i < 4; i++ {
		tr.push(v)
	}
	ctx := guardInfoPanelCtx{v: v, tr: tr, width: 120, sparkW: 12, gaugeW: 10}

	full := strings.Join(guardInfoTrendsPanelRows(ctx, guardPanelFull), "\n")
	for _, want := range []string{" saved ", "3 calls avoided"} {
		if !strings.Contains(full, want) {
			t.Errorf("trends panel must carry the turns-saved row (%q):\n%s", want, full)
		}
	}
	// The saved currency is "calls", never "tok" — it must not read as a token saving.
	if strings.Contains(full, "calls avoided tok") || strings.Contains(full, "calls tok") {
		t.Errorf("turns-saved row must stay in the calls currency:\n%s", full)
	}

	// Silent without any avoided calls: the saved row stays absent. #9450 deliberately
	// adds one honesty row for an unavailable token-destination recorder, so the base
	// layout is now the three pre-existing rows plus that explicit state.
	bareCtx := guardInfoPanelCtx{v: provenVisualVars(), tr: tr, width: 120, sparkW: 12, gaugeW: 10}
	bare := guardInfoTrendsPanelRows(bareCtx, guardPanelFull)
	bareRows := strings.Join(bare, "\n")
	if strings.Contains(bareRows, "calls avoided") {
		t.Errorf("trends panel must omit the saved row when nothing was avoided:\n%s", bareRows)
	}
	if !strings.Contains(bareRows, "tokens→ unavailable") || len(bare) != 4 {
		t.Errorf("base trends panel must keep three economy rows plus unavailable destination state, got %d:\n%s", len(bare), bareRows)
	}

	proxy := bareCtx
	proxy.v.Adjudication = &gateway.AdjudicationSummary{Transformed: 3, E2ELatencySumSeconds: 12, E2ELatencyCount: 3}
	proxyRows := strings.Join(guardInfoTrendsPanelRows(proxy, guardPanelFull), "\n")
	if !strings.Contains(proxyRows, "3 calls avoided") || !strings.Contains(proxyRows, "faster ≈ ~12s") {
		t.Fatalf("proxy trends missing reconciled observed time saving: %q", proxyRows)
	}
	proxy.v.Adjudication.E2ELatencySumSeconds = 0
	proxy.v.Adjudication.E2ELatencyCount = 0
	if rows := strings.Join(guardInfoTrendsPanelRows(proxy, guardPanelFull), "\n"); strings.Contains(rows, "faster ≈") {
		t.Fatalf("untimed trends fabricated wall-clock saving: %q", rows)
	}

	// Mini stays the single save row whether or not calls were avoided.
	mini := guardInfoTrendsPanelRows(ctx, guardPanelMini)
	if len(mini) != 1 || strings.Contains(mini[0], "calls avoided") {
		t.Errorf("mini trends form must stay the single save row: %q", mini)
	}
}

// TestRenderGuardInfoLineCarriesTurnsSaved proves the compact status line (line mode +
// the tiny-pane fallback) surfaces turns saved when present and omits it when zero, so
// the metric is watchable even when the visual layout has no room for the trends row.
func TestRenderGuardInfoLineCarriesTurnsSaved(t *testing.T) {
	v := provenVisualVars()
	v.CacheAttribution = &guardInfoCacheAttribution{FakVDSOAvoidedCalls: 5}
	if line := renderGuardInfoLine(v); !strings.Contains(line, "saved 5 calls") {
		t.Errorf("status line must carry turns saved when present: %q", line)
	}
	if line := renderGuardInfoLine(provenVisualVars()); strings.Contains(line, "calls") {
		t.Errorf("status line must omit turns saved when none were avoided: %q", line)
	}
}

// TestRenderGuardInfoLineCarriesAgents proves the compact status line (line mode, and
// the tiny-pane fallback) also surfaces the live agent fleet, so sub-agent visibility
// does not depend on the visual layout having room.
func TestRenderGuardInfoLineCarriesAgents(t *testing.T) {
	v := richVisualVars()
	line := renderGuardInfoLine(v)
	if !strings.Contains(line, "agents 2 active (1 continued, deepest g1)") {
		t.Errorf("status line must carry the agents summary: %q", line)
	}
	var bare guardInfoVars
	if strings.Contains(renderGuardInfoLine(bare), "agents") {
		t.Errorf("status line must omit agents with no sessions: %q", renderGuardInfoLine(bare))
	}
}

func TestGuardInfoTrendsPanelCapturesPerTurnPhaseAndCost(t *testing.T) {
	tr := newGuardInfoTrend(8)
	tr.prefillPerTurn = []float64{100, 120}
	tr.decodePerTurn = []float64{20, 30}
	tr.costPerTurn = []float64{120, 150}
	ctx := guardInfoPanelCtx{v: provenVisualVars(), tr: tr, width: 120, sparkW: 8, gaugeW: 10}
	rows := guardInfoTrendsPanelRows(ctx, guardPanelFull)
	got := strings.Join(rows, "\n")
	for _, exact := range []string{
		" uncached ▁█  120 tok/reply",
		" output   ▁█  30 tok/reply",
		" usage    ▁█  150 tok/reply · avg 135 · trend ↑ +25%",
	} {
		if !strings.Contains(got, exact) {
			t.Errorf("captured panel missing exact row %q:\n%s", exact, got)
		}
	}
	if strings.Contains(got, "$") || strings.Contains(got, "USD") {
		t.Fatalf("token-only endpoint was rendered as currency:\n%s", got)
	}
}

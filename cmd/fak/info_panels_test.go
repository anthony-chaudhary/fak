package main

import (
	"strings"
	"testing"
)

// richVisualVars is provenVisualVars plus the live-usage blocks the new panels render:
// a runtime resource block and a two-session registry (main agent + one sub-agent).
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
		{TraceID: "sub-trace", Run: "running", ParentTrace: "main-trace-long", Generation: 1},
	}
	return v
}

// TestRenderGuardInfoVisualBlockResourcesAndAgents proves the pane carries the LIVE
// info the entry/exit summaries used to monopolize: the gateway's own resource usage
// (heap + sparkline, goroutines, gc, generation rate) and the per-session agent rows
// (root + sub-agent lineage, wall-clock, remaining budget) — each under its own
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
		" mem    ",                          // resources gutter label
		"41MB heap", "68MB sys", "24 gor",   // live resource axes
		"gc 12",                             //
		" rate   ", "12.3 tok/s out",        // generation rate
		"ttft 1.20s", "oldest req 3s",       // latency + hung-request tell
		" agent  ",                          // agents gutter label
		"main-trace · root",                 // the root session (trace id capped at 10)
		"running", "1m35s",                  // run state + live wall-clock
		"380k tok left", "7 turns left",     // remaining budget axes
		"sub-trace · sub g1",                // the sub-agent lineage row
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
	if !strings.Contains(tight, "agents 2 active (1 sub, deepest g1)") {
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
// a parent trace makes a sub-agent row (generation floored at 1), zero budget axes are
// omitted (never rendered as exhausted), and empty ids degrade to "?".
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
	sub := guardInfoAgentText(guardInfoSession{TraceID: "sub", Run: "paused", ParentTrace: "abc"})
	if !strings.Contains(sub, "sub g1") || !strings.Contains(sub, "paused") {
		t.Errorf("sub-agent row must carry lineage + run state: %q", sub)
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

// TestRenderGuardInfoLineCarriesAgents proves the compact status line (line mode, and
// the tiny-pane fallback) also surfaces the live agent fleet, so sub-agent visibility
// does not depend on the visual layout having room.
func TestRenderGuardInfoLineCarriesAgents(t *testing.T) {
	v := richVisualVars()
	line := renderGuardInfoLine(v)
	if !strings.Contains(line, "agents 2 active (1 sub, deepest g1)") {
		t.Errorf("status line must carry the agents summary: %q", line)
	}
	var bare guardInfoVars
	if strings.Contains(renderGuardInfoLine(bare), "agents") {
		t.Errorf("status line must omit agents with no sessions: %q", renderGuardInfoLine(bare))
	}
}

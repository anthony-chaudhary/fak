package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func endpointsFixture() *gateway.SessionEndpoints {
	return &gateway.SessionEndpoints{
		Accounts: []gateway.SessionAccount{
			{Name: "july2", Email: "july2@x", Active: true, CanServe: true, LoginStatus: "ready"},
			{Name: "work", Walled: true, CanServe: true, LoginStatus: "ready"},
			{Name: "backup", CanServe: true, LoginStatus: "ready"},
		},
		Nodes: []gateway.SessionNode{
			{Role: "kernel", ID: "win-box", Kind: "host"},
			{Role: "serving", ID: "api.anthropic.com", Kind: "proxy", Detail: "anthropic"},
			{Role: "serving", ID: "dgx1:8080", Kind: "remote-serve"},
		},
	}
}

func TestGuardInfoEndpointsSummary(t *testing.T) {
	if got := guardInfoEndpointsSummary(nil); got != "" {
		t.Fatalf("nil endpoints summary = %q, want empty", got)
	}
	got := guardInfoEndpointsSummary(endpointsFixture())
	if !strings.Contains(got, "accts 3") || !strings.Contains(got, "active july2") || !strings.Contains(got, "nodes 3") {
		t.Fatalf("summary = %q, want seat count + active seat + node count", got)
	}
}

func TestGuardInfoEndpointsPanelFull(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Endpoints: endpointsFixture()}, width: 120}
	rows := guardInfoEndpointsPanelRows(ctx, guardPanelFull)
	joined := strings.Join(rows, "\n")
	// accounts header names the active seat + walled count; chips mark active/walled.
	if !strings.Contains(joined, "3 seats") || !strings.Contains(joined, "active july2 (ready)") || !strings.Contains(joined, "1 walled") {
		t.Fatalf("accounts head missing from:\n%s", joined)
	}
	if !strings.Contains(joined, guardChipActive+"july2") || !strings.Contains(joined, guardChipWalled+"work") {
		t.Fatalf("account chips missing active/walled markers:\n%s", joined)
	}
	// nodes row shows the kernel host + a serving node; the extra serving node continues.
	if !strings.Contains(joined, "kernel win-box") || !strings.Contains(joined, "serving api.anthropic.com") {
		t.Fatalf("node row missing kernel/serving:\n%s", joined)
	}
	if !strings.Contains(joined, "dgx1:8080") {
		t.Fatalf("second serving node missing:\n%s", joined)
	}
}

func TestGuardInfoEndpointsPanelSilentWhenAbsent(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{}, width: 80}
	if rows := guardInfoEndpointsPanelRows(ctx, guardPanelFull); rows != nil {
		t.Fatalf("endpoints panel with no block = %v, want nil (silent)", rows)
	}
}

// TestGuardSafetyWordPrefersAdjudication is the correctness witness for the vacuous-safety
// fix: on the guard proxy the Kernel counters are 0 while the operation-ledger Adjudication
// tally holds the real refusals, so the safety word must read Adjudication.
func TestGuardSafetyWordPrefersAdjudication(t *testing.T) {
	// Proxy: Kernel all-zero (Decide never increments it), Adjudication has real refusals.
	v := guardInfoVars{Adjudication: &gateway.AdjudicationSummary{Denied: 2, Transformed: 1, Quarantined: 3}}
	got := guardSafetyWord(v)
	if !strings.Contains(got, "blocked 2") || !strings.Contains(got, "fixed 1") || !strings.Contains(got, "set aside 3") {
		t.Fatalf("safety word = %q, want it read from the adjudication tally", got)
	}
	// No adjudication block (a fak serve gateway): fall back to the kernel counters.
	v = guardInfoVars{}
	v.Kernel.Denies = 5
	if got := guardSafetyWord(v); !strings.Contains(got, "blocked 5") {
		t.Fatalf("safety word fallback = %q, want the kernel counters", got)
	}
	// Clean session reads the all-clear.
	if got := guardSafetyWord(guardInfoVars{Adjudication: &gateway.AdjudicationSummary{}}); got != "safety: nothing blocked" {
		t.Fatalf("clean safety word = %q, want nothing blocked", got)
	}
}

// TestGuardInfoAdjudicationDetail pins the promoted exit-detail "why": a nil block and a clean
// adjudication stay silent (the zero-cost contract), and a real one leads with the top reason
// by count, breaks ties by code, caps at three with a "+N more" tail, and carries the held-for-
// witness and deferred tallies.
func TestGuardInfoAdjudicationDetail(t *testing.T) {
	if got := guardInfoAdjudicationDetail(nil); got != "" {
		t.Fatalf("nil adjudication detail = %q, want empty", got)
	}
	if got := guardInfoAdjudicationDetail(&gateway.AdjudicationSummary{Total: 5, Allowed: 5}); got != "" {
		t.Fatalf("clean adjudication detail = %q, want empty (nothing refused, nothing held)", got)
	}
	a := &gateway.AdjudicationSummary{
		Denied:    3,
		Escalated: 1,
		Deferred:  2,
		ByReason: map[string]uint64{
			"dangerous_command": 2,
			"out_of_tree_write": 1,
			"secret_in_arg":     1,
			"unknown_tool":      1,
		},
	}
	got := guardInfoAdjudicationDetail(a)
	if !strings.Contains(got, "why dangerous_command x2") {
		t.Fatalf("detail must lead with the top reason by count: %q", got)
	}
	// Four reasons, cap 3 → the top three (count desc, then code asc) plus one folded.
	if !strings.Contains(got, "out_of_tree_write x1") || !strings.Contains(got, "secret_in_arg x1") {
		t.Fatalf("detail must show the tie-broken next reasons: %q", got)
	}
	if strings.Contains(got, "unknown_tool") || !strings.Contains(got, "+1 more") {
		t.Fatalf("detail must fold the reason past the cap into +N more: %q", got)
	}
	if !strings.Contains(got, "1 held for witness") || !strings.Contains(got, "2 deferred") {
		t.Fatalf("detail must carry the held/deferred tallies: %q", got)
	}
}

// TestGuardInfoTasksPanelCarriesWhy proves the tasks sub-pane grows a "why" row surfacing the
// adjudication reasons at full level, keeps the mini form the single cache gauge, and stays
// silent (its original two-row form) when the gateway reported no adjudication block.
func TestGuardInfoTasksPanelCarriesWhy(t *testing.T) {
	v := provenVisualVars()
	v.Adjudication = &gateway.AdjudicationSummary{Denied: 2, ByReason: map[string]uint64{"dangerous_command": 2}}
	ctx := guardInfoPanelCtx{v: v, width: 120, gaugeW: 10}
	full := strings.Join(guardInfoTasksPanelRows(ctx, guardPanelFull), "\n")
	if !strings.Contains(full, " why    ") || !strings.Contains(full, "dangerous_command x2") {
		t.Fatalf("tasks panel full must carry the why row:\n%s", full)
	}
	if mini := guardInfoTasksPanelRows(ctx, guardPanelMini); len(mini) != 1 || strings.Contains(mini[0], "why") {
		t.Fatalf("tasks mini must stay the single cache row: %v", mini)
	}
	bare := strings.Join(guardInfoTasksPanelRows(guardInfoPanelCtx{v: provenVisualVars(), width: 120, gaugeW: 10}, guardPanelFull), "\n")
	if strings.Contains(bare, " why    ") {
		t.Fatalf("tasks panel must omit the why row without an adjudication block:\n%s", bare)
	}
}

// TestRenderGuardInfoLineCarriesWhy proves the compact status line (line mode + the tiny-pane
// fallback) also surfaces the adjudication why when present and omits it otherwise.
func TestRenderGuardInfoLineCarriesWhy(t *testing.T) {
	v := provenVisualVars()
	v.Adjudication = &gateway.AdjudicationSummary{Denied: 1, ByReason: map[string]uint64{"dangerous_command": 1}}
	if line := renderGuardInfoLine(v); !strings.Contains(line, "why dangerous_command x1") {
		t.Fatalf("status line must carry the adjudication why when present: %q", line)
	}
	if line := renderGuardInfoLine(provenVisualVars()); strings.Contains(line, "why ") {
		t.Fatalf("status line must omit the why clause without an adjudication block: %q", line)
	}
}

// TestGuardInfoLegendDocumentsWhy proves the pane's in-line guide explains the promoted
// why-detail clause, so an operator who sees "why <reason> xN" beside the safety count knows
// it is the reason code(s) behind those blocks (the live twin of the exit summary breakdown).
func TestGuardInfoLegendDocumentsWhy(t *testing.T) {
	legend := guardInfoLegend()
	if !strings.Contains(legend, "why") || !strings.Contains(legend, "reason code") {
		t.Fatalf("legend must document the promoted why-detail line:\n%s", legend)
	}
}

func TestGuardInfoHarnessText(t *testing.T) {
	if got := guardInfoHarnessText(nil); got != "" {
		t.Fatalf("nil harness text = %q, want empty", got)
	}
	h := &gateway.SessionHarness{KernelCPUPercent: 42, KernelRSSBytes: 64 << 20, NetRxBytes: 5 << 20, NetTxBytes: 2 << 20}
	got := guardInfoHarnessText(h)
	if !strings.Contains(got, "cpu 42%") || !strings.Contains(got, "rss ") || !strings.Contains(got, "net ") {
		t.Fatalf("harness text = %q, want cpu/rss/net", got)
	}
}

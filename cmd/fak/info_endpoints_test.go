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

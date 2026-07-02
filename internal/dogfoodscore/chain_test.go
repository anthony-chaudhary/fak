package dogfoodscore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// chainFixture builds a temp root holding one packet report (and optionally a
// bridge receipt) with pinned mtimes, so scanChain is deterministic.
func chainFixture(t *testing.T, reportAge time.Duration, receipt map[string]any, receiptAge time.Duration, now time.Time) string {
	t.Helper()
	root := t.TempDir()
	stamp := filepath.Join(root, ".fak", "recent-feature-dogfood", "20260701T000000Z")
	if err := os.MkdirAll(stamp, 0o755); err != nil {
		t.Fatalf("mkdir fixture stamp: %v", err)
	}
	report := filepath.Join(stamp, "report.json")
	if err := os.WriteFile(report, []byte(`{"schema":"fak.recent-feature-dogfood.v1","probes":[]}`), 0o644); err != nil {
		t.Fatalf("write fixture report: %v", err)
	}
	rt := now.Add(-reportAge)
	if err := os.Chtimes(report, rt, rt); err != nil {
		t.Fatalf("chtimes report: %v", err)
	}
	if receipt != nil {
		b, err := json.Marshal(receipt)
		if err != nil {
			t.Fatalf("marshal fixture receipt: %v", err)
		}
		path := filepath.Join(stamp, chainReceiptName)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatalf("write fixture receipt: %v", err)
		}
		ct := now.Add(-receiptAge)
		if err := os.Chtimes(path, ct, ct); err != nil {
			t.Fatalf("chtimes receipt: %v", err)
		}
	}
	return root
}

func chainKPI(t *testing.T, rows []KPIResult, key string) KPIResult {
	t.Helper()
	for _, r := range rows {
		if r.Key == key {
			return r
		}
	}
	t.Fatalf("chain KPI %q missing from %+v", key, rows)
	return KPIResult{}
}

// A host with no packet evidence is honestly UNSCORED: both rungs report their
// gap (naming the producing command) but stay SOFT — no debt, and Build drops
// the axis from the composite — so a fresh clone (the CI ratchet runner) folds
// bit-identically to the pre-chain card instead of carrying unretirable debt.
func TestChain_NoEvidenceIsUnscoredNotDebt(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	ce := scanChain(t.TempDir(), now)
	rows := chainResults(ce, DefaultChainWindowHours)
	fresh := chainKPI(t, rows, "chain_packet_fresh")
	bridged := chainKPI(t, rows, "chain_actions_bridged")
	if fresh.Passed || bridged.Passed {
		t.Fatalf("no evidence must not read as clean: fresh=%+v bridged=%+v", fresh, bridged)
	}
	if fresh.Hard || bridged.Hard {
		t.Fatalf("no evidence must stay SOFT (unscored, not debt): fresh=%+v bridged=%+v", fresh, bridged)
	}
	if !strings.Contains(fresh.Detail, "make dogfood-recent") {
		t.Fatalf("fresh detail must name the producing command, got %q", fresh.Detail)
	}
	if !strings.Contains(bridged.Detail, "make dogfood-recent") {
		t.Fatalf("bridged detail must name the producing command, got %q", bridged.Detail)
	}
}

// With evidence present the rungs are HARD — the operator-host pressure the axis
// exists for.
func TestChain_EvidenceMakesRungsHard(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	root := chainFixture(t, 24*time.Hour, nil, 0, now)
	for _, r := range chainResults(scanChain(root, now), DefaultChainWindowHours) {
		if !r.Hard {
			t.Fatalf("with evidence the chain rungs must be hard: %+v", r)
		}
	}
}

// The unmeasured axis drops out of Build's composite: on a root with no packet
// evidence the corpus says chain_measured=false and the dogfood debt counts no
// chain defect.
func TestBuild_UnmeasuredChainAddsNoDebt(t *testing.T) {
	root := t.TempDir() // no .fak evidence at all
	p := Build(Options{Root: root, Now: time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC), ClaudeHome: t.TempDir()})
	if measured, _ := p.Corpus["chain_measured"].(bool); measured {
		t.Fatalf("empty root must report chain_measured=false")
	}
	for _, r := range p.Chain {
		if r.Hard {
			t.Fatalf("unmeasured chain rung must be soft: %+v", r)
		}
	}
}

// A fresh report with no receipt greens freshness but reds the bridge rung with
// the exact bridging command — unverified is not verified-clean.
func TestChain_FreshReportWithoutReceiptRedsBridge(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	root := chainFixture(t, 24*time.Hour, nil, 0, now)
	ce := scanChain(root, now)
	if ce.Reports != 1 || ce.ReceiptPresent {
		t.Fatalf("fixture fold wrong: %+v", ce)
	}
	rows := chainResults(ce, DefaultChainWindowHours)
	if !chainKPI(t, rows, "chain_packet_fresh").Passed {
		t.Fatalf("a 24h-old report inside a %dh window must be fresh", DefaultChainWindowHours)
	}
	bridged := chainKPI(t, rows, "chain_actions_bridged")
	if bridged.Passed {
		t.Fatalf("a report without a receipt must red the bridge rung")
	}
	if !strings.Contains(bridged.Detail, "fak dogfood-issues --live") {
		t.Fatalf("bridge detail must name the bridging command, got %q", bridged.Detail)
	}
}

// A stale report reds freshness even when the bridge receipt is clean.
func TestChain_StaleReportRedsFreshness(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	rec := map[string]any{"schema": "fak.dogfood-issues-receipt.v1", "mode": "live", "planned_creates": 0, "synced_ok": 2, "synced_failed": 0}
	root := chainFixture(t, time.Duration(DefaultChainWindowHours+48)*time.Hour, rec, time.Duration(DefaultChainWindowHours+47)*time.Hour, now)
	rows := chainResults(scanChain(root, now), DefaultChainWindowHours)
	if chainKPI(t, rows, "chain_packet_fresh").Passed {
		t.Fatalf("a report older than the window must red chain_packet_fresh")
	}
	if !chainKPI(t, rows, "chain_actions_bridged").Passed {
		t.Fatalf("a clean live receipt newer than its report must green the bridge rung")
	}
}

// The receipt semantics table: live-with-failures reds, live-clean greens,
// fetch-existing greens only at zero pending creates, an unknown mode reds, and a
// receipt OLDER than its report reds (it witnessed a previous report).
func TestChain_ReceiptSemantics(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name    string
		receipt map[string]any
		stale   bool
		want    bool
		detail  string
	}{
		{name: "live clean", receipt: map[string]any{"mode": "live", "synced_ok": 3, "synced_failed": 0}, want: true, detail: "3 action(s)"},
		{name: "live failures", receipt: map[string]any{"mode": "live", "synced_ok": 1, "synced_failed": 2}, want: false, detail: "2 gh sync failure(s)"},
		{name: "fetch clean", receipt: map[string]any{"mode": "fetch-existing", "planned_creates": 0}, want: true, detail: "already tracked"},
		{name: "fetch pending", receipt: map[string]any{"mode": "fetch-existing", "planned_creates": 4}, want: false, detail: "4 unfiled action(s)"},
		{name: "unknown mode", receipt: map[string]any{"mode": "dry-run"}, want: false, detail: "unrecognized"},
		{name: "receipt predates report", receipt: map[string]any{"mode": "live", "synced_ok": 3, "synced_failed": 0}, stale: true, want: false, detail: "predates"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receiptAge := 1 * time.Hour
			if tc.stale {
				receiptAge = 72 * time.Hour // report is 24h old below, so this receipt is older
			}
			root := chainFixture(t, 24*time.Hour, tc.receipt, receiptAge, now)
			rows := chainResults(scanChain(root, now), DefaultChainWindowHours)
			got := chainKPI(t, rows, "chain_actions_bridged")
			if got.Passed != tc.want {
				t.Fatalf("passed = %v, want %v (%+v)", got.Passed, tc.want, got)
			}
			if !strings.Contains(got.Detail, tc.detail) {
				t.Fatalf("detail %q must contain %q", got.Detail, tc.detail)
			}
		})
	}
}

// Build carries the chain axis end-to-end: the payload exposes the chain rows and
// the corpus carries a chain value, so the control pane (and through it the
// improve-loops / tend super loops) see chain debt without any new wiring.
func TestBuild_CarriesChainAxis(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	rec := map[string]any{"mode": "live", "synced_ok": 1, "synced_failed": 0}
	root := chainFixture(t, 2*time.Hour, rec, 1*time.Hour, now)
	p := Build(Options{Root: root, Now: now, ClaudeHome: t.TempDir()})
	if len(p.Chain) != 2 {
		t.Fatalf("payload must carry both chain rows, got %d", len(p.Chain))
	}
	if p.Evidence.Chain.Reports != 1 {
		t.Fatalf("evidence must fold the packet report, got %+v", p.Evidence.Chain)
	}
	if _, ok := p.Corpus["chain_value"]; !ok {
		t.Fatalf("corpus must carry chain_value")
	}
	for _, r := range p.Chain {
		if !r.Passed {
			t.Fatalf("fresh report + clean live receipt must green the chain, got %+v", r)
		}
	}
	rendered := Render(p)
	if !strings.Contains(rendered, "CHAIN (") {
		t.Fatalf("Render must show the chain section:\n%s", rendered)
	}
	md := Markdown(p)
	if !strings.Contains(md, "## Chain") {
		t.Fatalf("Markdown must show the chain section:\n%s", md)
	}
}

// The receipt contract this package reads is owned by internal/dogfoodissues.
// This pins the shared shape (name, mode values, count fields) so the two
// packages cannot drift apart without a test going red on one side.
func TestChain_ReceiptContractPinned(t *testing.T) {
	if chainReceiptName != "issues-sync.json" {
		t.Fatalf("receipt name drifted: %q", chainReceiptName)
	}
	// Field names read by scanChain, as dogfoodissues.Receipt marshals them.
	for _, field := range []string{"mode", "planned_creates", "synced_ok", "synced_failed"} {
		raw := `{"mode":"live","planned_creates":1,"synced_ok":2,"synced_failed":3}`
		var rec map[string]any
		if err := json.Unmarshal([]byte(raw), &rec); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, ok := rec[field]; !ok {
			t.Fatalf("pinned receipt field %q missing", field)
		}
	}
}

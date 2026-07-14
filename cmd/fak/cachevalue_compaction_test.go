package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// TestReadGatewayLedgersDedup witnesses the multi-ledger merge: a session exit that
// appears in BOTH a live ledger and its published docs mirror (the overlapping window)
// must fold exactly once, so a regime comparison spanning both ledger windows is a single
// invocation that never double-counts the overlap. A row unique to one ledger survives.
func TestReadGatewayLedgersDedup(t *testing.T) {
	dir := t.TempDir()

	// shared is byte-identical in both files (same pid/unix_millis/generated_at) — the
	// overlap that must collapse to one row.
	shared := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"guard","pid":100,"unix_millis":1000,"generated_at":"2026-07-11T10:00:00Z","provenance":{"compact_history_budget":96000},"counters":{"observed_turns":50,"compaction_fired":10,"compaction_shed_tokens":700,"cached_prompt_tokens":300}}`
	// liveOnly and docsOnly are unique to their respective files.
	liveOnly := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"guard","pid":101,"unix_millis":2000,"generated_at":"2026-07-12T10:00:00Z","provenance":{"compact_history_budget":96000},"counters":{"observed_turns":60,"compaction_fired":20,"compaction_shed_tokens":800,"cached_prompt_tokens":200}}`
	docsOnly := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"guard","pid":102,"unix_millis":3000,"generated_at":"2026-07-10T10:00:00Z","provenance":{"compact_history_budget":48000},"counters":{"observed_turns":70,"compaction_fired":30,"compaction_shed_tokens":900,"cached_prompt_tokens":100}}`

	live := filepath.Join(dir, "live.jsonl")
	docs := filepath.Join(dir, "docs.jsonl")
	if err := os.WriteFile(live, []byte(shared+"\n"+liveOnly+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(docs, []byte(shared+"\n"+docsOnly+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	rows := readGatewayLedgersDedup([]string{live, docs})
	// 2 files × 2 rows = 4 raw, minus the 1 duplicated shared row = 3 distinct.
	if len(rows) != 3 {
		t.Fatalf("deduped row count = %d, want 3 (shared row folded once)", len(rows))
	}

	// The fold over the merged rows must see all three distinct sessions across both regimes.
	rep := gatewayusageledger.FoldCompaction(rows, "")
	if rep.ExitRows != 3 {
		t.Fatalf("FoldCompaction ExitRows = %d, want 3", rep.ExitRows)
	}
	var sessions uint64
	sawBudgets := map[int]bool{}
	for _, seg := range rep.Segments {
		sessions += uint64(seg.Sessions)
		sawBudgets[seg.Budget] = true
	}
	if sessions != 3 {
		t.Fatalf("segmented sessions total = %d, want 3", sessions)
	}
	if !sawBudgets[48000] || !sawBudgets[96000] {
		t.Fatalf("merged fold missing a regime: budgets seen = %v, want both 48000 and 96000", sawBudgets)
	}
}

// TestReadGatewayLedgersDedupSinglePathUnchanged pins that a single-ledger call is a plain
// read — no row is dropped when there is nothing to dedup against.
func TestReadGatewayLedgersDedupSinglePathUnchanged(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "one.jsonl")
	body := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"guard","pid":1,"unix_millis":10,"generated_at":"2026-07-11T00:00:00Z","counters":{"observed_turns":5,"compaction_fired":1,"compaction_shed_tokens":10,"cached_prompt_tokens":90}}` + "\n" +
		`{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"guard","pid":2,"unix_millis":20,"generated_at":"2026-07-11T00:01:00Z","counters":{"observed_turns":6,"compaction_fired":2,"compaction_shed_tokens":20,"cached_prompt_tokens":80}}` + "\n"
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := readGatewayLedgersDedup([]string{p})
	if len(rows) != 2 {
		t.Fatalf("single-path row count = %d, want 2 (no dedup applied)", len(rows))
	}
}

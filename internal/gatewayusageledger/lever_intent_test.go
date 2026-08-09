package gatewayusageledger

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestLeverIntentCountersRoundTrip pins the durable half of #4349: the two managed-cache
// INTENT flags and the cold-defer EFFECT count survive Append -> ReadLedgerFile, so a
// session that armed a lever and did nothing is still distinguishable from one that never
// armed it after the process is gone.
func TestLeverIntentCountersRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "gateway-usage.jsonl")

	row := NewRow("exit", "guard", "claude", "", time.Minute, nil, Counters{
		ManagedCacheActive:  true,
		DeferColdToolsArmed: true,
		DeferColdCount:      7,
	}, time.Unix(1000, 0))
	if err := Append(ledgerPath, row); err != nil {
		t.Fatalf("Append: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	got := rows[0].Counters
	if !got.ManagedCacheActive || !got.DeferColdToolsArmed || got.DeferColdCount != 7 {
		t.Fatalf("lever intent/effect did not round-trip: active=%v armed=%v count=%d",
			got.ManagedCacheActive, got.DeferColdToolsArmed, got.DeferColdCount)
	}
}

// TestLeverIntentAbsentOnPreFieldRows is the wire-compatibility fence these fields were
// given omitempty for: a row that arms nothing serializes with NO lever keys at all, so
// every row written before #4349 stays byte-identical (and its RowKey — a hash over this
// struct's JSON — unchanged).
//
// The cost of that choice is stated in Counters' own doc and pinned here so a reader
// cannot miss it: a MEASURED lever-OFF row is byte-identical to an UNMEASURED one, so
// these fields witness the ARMED case only. Absent must read "not instrumented", never
// "the lever was off".
func TestLeverIntentAbsentOnPreFieldRows(t *testing.T) {
	zero := NewRow("exit", "serve", "http", "", 0, nil, Counters{}, time.Unix(2000, 0))
	zb, err := json.Marshal(zero)
	if err != nil {
		t.Fatalf("marshal zero row: %v", err)
	}
	for _, key := range []string{"managed_cache_active", "defer_cold_tools_armed", "defer_cold_count"} {
		if strings.Contains(string(zb), key) {
			t.Fatalf("an unarmed row must omit %s (pre-#4349 rows stay byte-identical):\n%s", key, zb)
		}
	}

	// An armed row DOES carry them — otherwise omitempty would have made the fields inert.
	armed := NewRow("exit", "serve", "http", "", 0, nil, Counters{
		ManagedCacheActive: true, DeferColdToolsArmed: true, DeferColdCount: 1,
	}, time.Unix(2000, 0))
	ab, err := json.Marshal(armed)
	if err != nil {
		t.Fatalf("marshal armed row: %v", err)
	}
	for _, key := range []string{`"managed_cache_active":true`, `"defer_cold_tools_armed":true`, `"defer_cold_count":1`} {
		if !strings.Contains(string(ab), key) {
			t.Fatalf("armed row missing %s:\n%s", key, ab)
		}
	}
}

// TestSumCountersOrsLeverIntent pins the carryforward algebra a flag has: sumCounters ORs
// the intent flags (a folded window had the lever armed iff ANY row in it did) and sums
// the cold-defer count like every other quantity. OR is the direction that cannot lose
// the evidence a lever was armed — AND would erase an armed session the moment it folded
// beside a never-armed one.
func TestSumCountersOrsLeverIntent(t *testing.T) {
	var dst Counters
	if err := sumCounters(&dst, Counters{DeferColdCount: 2}); err != nil {
		t.Fatal(err)
	}
	if dst.ManagedCacheActive || dst.DeferColdToolsArmed {
		t.Fatalf("folding an unarmed row must not arm a lever: %+v", dst)
	}
	if err := sumCounters(&dst, Counters{
		ManagedCacheActive: true, DeferColdToolsArmed: true, DeferColdCount: 3,
	}); err != nil {
		t.Fatal(err)
	}
	if !dst.ManagedCacheActive || !dst.DeferColdToolsArmed {
		t.Fatalf("one armed row must arm the folded window: %+v", dst)
	}
	if dst.DeferColdCount != 5 {
		t.Fatalf("DeferColdCount = %d, want 5 (2+3 — the effect half still sums)", dst.DeferColdCount)
	}
	// OR is idempotent: folding another unarmed row cannot DISARM the window.
	if err := sumCounters(&dst, Counters{DeferColdCount: 1}); err != nil {
		t.Fatal(err)
	}
	if !dst.ManagedCacheActive || !dst.DeferColdToolsArmed || dst.DeferColdCount != 6 {
		t.Fatalf("a later unarmed row must not disarm the window: %+v", dst)
	}
}

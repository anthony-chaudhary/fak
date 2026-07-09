package gatewayusageledger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestAppendTwoSessionsProducesTwoRows is the #1610 fail-before/pass-after proof: it
// simulates two independent `fak serve` sessions (as if the process restarted between
// them — a fresh PID/UnixMillis/counters, same ledger file) and asserts both land as
// separate JSONL rows in a temp ledger file. Before this package existed there was no
// way to durably observe a served-turn counter snapshot across a restart at all; this
// test is the evidence the ledger now provides that durability.
func TestAppendTwoSessionsProducesTwoRows(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "gateway-usage.jsonl")

	session1 := NewRow("exit", "serve", "http", "gw-1", 12*time.Second, nil, Counters{
		Submits:            10,
		VDSOHits:           4,
		EngineCalls:        6,
		Denies:             1,
		Admitted:           9,
		InputTokens:        1000,
		OutputTokens:       200,
		CachedPromptTokens: 400,
	}, time.Unix(1000, 0))
	if err := Append(ledgerPath, session1); err != nil {
		t.Fatalf("Append session1: %v", err)
	}

	// Simulate a restart: a second, independent session appends to the SAME file.
	session2 := NewRow("exit", "serve", "http", "gw-2", 30*time.Second, nil, Counters{
		Submits:            25,
		VDSOHits:           12,
		EngineCalls:        13,
		Denies:             2,
		Admitted:           23,
		InputTokens:        3000,
		OutputTokens:       800,
		CachedPromptTokens: 1200,
	}, time.Unix(2000, 0))
	if err := Append(ledgerPath, session2); err != nil {
		t.Fatalf("Append session2: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows after two sessions, got %d: %+v", len(rows), rows)
	}
	if rows[0].SessionID != "gw-1" || rows[1].SessionID != "gw-2" {
		t.Fatalf("rows out of expected order/content: %+v", rows)
	}
	if rows[0].Schema != Schema || rows[1].Schema != Schema {
		t.Fatalf("expected schema %q on both rows, got %q and %q", Schema, rows[0].Schema, rows[1].Schema)
	}
	if rows[0].Counters.Submits != 10 || rows[1].Counters.Submits != 25 {
		t.Fatalf("counters did not round-trip: %+v", rows)
	}

	// The reader function: fold >=2 rows into a trend (acceptance criteria).
	trend, ok := FoldTrend(rows)
	if !ok {
		t.Fatalf("FoldTrend: expected ok=true for 2 rows")
	}
	if trend.Sessions != 2 {
		t.Fatalf("trend.Sessions = %d, want 2", trend.Sessions)
	}
	if trend.DeltaSubmits != 15 {
		t.Fatalf("trend.DeltaSubmits = %d, want 15 (25-10)", trend.DeltaSubmits)
	}
	if trend.DeltaInputTokens != 2000 {
		t.Fatalf("trend.DeltaInputTokens = %d, want 2000 (3000-1000)", trend.DeltaInputTokens)
	}
	if trend.DeltaVDSOHits != 8 {
		t.Fatalf("trend.DeltaVDSOHits = %d, want 8 (12-4)", trend.DeltaVDSOHits)
	}
}

// TestFoldTrendInsufficientOnFewerThanTwoRows asserts the fall-open posture: zero or
// one row is not a failure, just "not enough data yet" (ok=false), mirroring
// cachevalueledger's thin-corpus posture.
func TestFoldTrendInsufficientOnFewerThanTwoRows(t *testing.T) {
	if _, ok := FoldTrend(nil); ok {
		t.Fatalf("FoldTrend(nil): expected ok=false")
	}
	one := []Row{NewRow("exit", "serve", "", "gw-1", 0, nil, Counters{}, time.Now())}
	if _, ok := FoldTrend(one); ok {
		t.Fatalf("FoldTrend(1 row): expected ok=false")
	}
}

// TestReadLedgerFileMissingIsEmptyNotError matches ReadLedgerFile's documented
// fall-open posture — a ledger that has never been written to is a clean first-run
// state (nil rows), not an error a caller must special-case.
func TestReadLedgerFileMissingIsEmptyNotError(t *testing.T) {
	dir := t.TempDir()
	rows := ReadLedgerFile(filepath.Join(dir, "does-not-exist.jsonl"))
	if len(rows) != 0 {
		t.Fatalf("expected 0 rows for a missing ledger file, got %d", len(rows))
	}
}

// TestCacheTTLUpgradeCountersRoundTrip is the #1844-C6 durable-witness proof: the
// managed-cache TTL-upgrade outcome family survives Append -> ReadLedgerFile, and the
// serialized row carries the ttl-bearing counter keys the managed-cache proving ground
// (tools/managed_cache_proving_ground.py) auto-detects to climb ttl_upgrade_1h past
// UNWIRED. The unconditional key matters: even an upgraded=0 row from a lever-on
// session proves the durable channel exists.
func TestCacheTTLUpgradeCountersRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "gateway-usage.jsonl")

	row := NewRow("exit", "guard", "claude", "", time.Minute, nil, Counters{
		CacheTTLUpgradesUpgraded: 3,
		CacheTTLUpgradeReasons:   map[string]uint64{"volatile_head": 2, "no_stable_breakpoint": 1},
	}, time.Unix(1000, 0))
	if err := Append(ledgerPath, row); err != nil {
		t.Fatalf("Append: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	got := rows[0].Counters
	if got.CacheTTLUpgradesUpgraded != 3 {
		t.Fatalf("CacheTTLUpgradesUpgraded = %d, want 3", got.CacheTTLUpgradesUpgraded)
	}
	if got.CacheTTLUpgradeReasons["volatile_head"] != 2 || got.CacheTTLUpgradeReasons["no_stable_breakpoint"] != 1 {
		t.Fatalf("CacheTTLUpgradeReasons did not round-trip: %v", got.CacheTTLUpgradeReasons)
	}

	// The serialized counter keys are the proving ground's auto-detect contract: any
	// "ttl"-bearing key in a usage row climbs the C6 rung with no tool change. The
	// upgraded key must serialize even at zero (no omitempty), so a lever-on session
	// with no eligible head still leaves durable evidence of the channel.
	raw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	for _, key := range []string{`"cache_ttl_upgrades_upgraded":3`, `"cache_ttl_upgrade_reasons"`} {
		if !strings.Contains(string(raw), key) {
			t.Fatalf("serialized row missing %s:\n%s", key, raw)
		}
	}
	zero := NewRow("exit", "serve", "http", "", 0, nil, Counters{}, time.Unix(2000, 0))
	zb, err := json.Marshal(zero)
	if err != nil {
		t.Fatalf("marshal zero row: %v", err)
	}
	if !strings.Contains(string(zb), `"cache_ttl_upgrades_upgraded":0`) {
		t.Fatalf("zero row must still carry the ttl counter key:\n%s", zb)
	}
	if strings.Contains(string(zb), "cache_ttl_upgrade_reasons") {
		t.Fatalf("zero row must omit the empty reasons map:\n%s", zb)
	}
}

// TestProvenanceAndObservedTurnsRoundTrip proves the two self-describing additions survive
// Append -> ReadLedgerFile, and — the load-bearing schema guarantee — that a nil-provenance
// row stays byte-compatible with the pre-provenance shape (no "provenance" key at all), while
// a supplied provenance carrying a MEANINGFUL zero (AssumeSessionTurns:0 = prior disabled) is
// preserved as present-and-zero, not collapsed to absent. That distinction is exactly why Row
// holds Provenance by pointer.
func TestProvenanceAndObservedTurnsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "gateway-usage.jsonl")

	prov := &Provenance{AssumeSessionTurns: 50, CompactHistoryBudget: 120000, BuildRevision: "abc123-dirty"}
	row := NewRow("exit", "guard", "claude", "g-1", time.Minute, prov, Counters{
		ObservedTurns: 42,
		CachedTurns:   7,
	}, time.Unix(1000, 0))
	if err := Append(ledgerPath, row); err != nil {
		t.Fatalf("Append: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	if rows[0].Counters.ObservedTurns != 42 {
		t.Fatalf("ObservedTurns did not round-trip: got %d, want 42", rows[0].Counters.ObservedTurns)
	}
	if rows[0].Provenance == nil {
		t.Fatalf("provenance dropped on read")
	}
	if p := rows[0].Provenance; p.AssumeSessionTurns != 50 || p.CompactHistoryBudget != 120000 || p.BuildRevision != "abc123-dirty" {
		t.Fatalf("provenance did not round-trip: %+v", p)
	}

	// observed_turns is unconditional (no omitempty) so a length-distribution reader can
	// always percentile it; a cold session records observed_turns:0 rather than a missing key.
	raw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if !strings.Contains(string(raw), `"observed_turns":42`) || !strings.Contains(string(raw), `"provenance"`) {
		t.Fatalf("serialized row missing observed_turns/provenance:\n%s", raw)
	}

	// nil provenance stays on the pre-provenance byte shape (no "provenance" key), but a
	// SUPPLIED zero provenance is present — the pointer's whole reason for being.
	nilProv := NewRow("exit", "serve", "http", "", 0, nil, Counters{}, time.Unix(2000, 0))
	nb, err := json.Marshal(nilProv)
	if err != nil {
		t.Fatalf("marshal nil-provenance row: %v", err)
	}
	if strings.Contains(string(nb), "provenance") {
		t.Fatalf("nil-provenance row must omit the provenance key entirely:\n%s", nb)
	}
	zeroProv := NewRow("exit", "serve", "http", "", 0, &Provenance{}, Counters{}, time.Unix(2000, 0))
	zb, err := json.Marshal(zeroProv)
	if err != nil {
		t.Fatalf("marshal zero-provenance row: %v", err)
	}
	if !strings.Contains(string(zb), `"provenance":{}`) {
		t.Fatalf("supplied zero provenance must serialize as present-but-empty:\n%s", zb)
	}
}

// TestParseLedgerSkipsCorruptLines asserts a malformed or foreign line never aborts
// the whole read — only that line is dropped.
func TestParseLedgerSkipsCorruptLines(t *testing.T) {
	content := `{"schema":"fak-gateway-usage-ledger/1","kind":"exit","session_type":"serve","pid":1,"unix_millis":1000,"counters":{},"generated_at":"2026-01-01T00:00:00Z"}
not json at all
{"schema":"","kind":"exit"}

{"schema":"fak-gateway-usage-ledger/1","kind":"periodic","session_type":"serve","pid":2,"unix_millis":2000,"counters":{},"generated_at":"2026-01-01T00:01:00Z"}
`
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("expected 2 valid rows out of 5 lines, got %d: %+v", len(rows), rows)
	}
}

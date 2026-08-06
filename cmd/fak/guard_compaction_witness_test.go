package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// compactionWitnessFixture is one session's live summary, shaped so every conditional
// row in the formatter has something to say: fires and bails both non-zero, an
// anchor-starved count, a solvency-forced subset of the fires, and a two-reason bail
// breakdown (the lump that is uninterpretable without it).
func compactionWitnessFixture() gateway.AdjudicationSummary {
	return gateway.AdjudicationSummary{
		CompactionFired:           3,
		CompactionBailed:          2,
		CompactionOff:             1,
		CompactionAnchorStarved:   4,
		CompactionSolvencyForced:  2,
		CompactionShedTokens:      12345,
		CompactionBudget:          60000,
		CompactionCacheReadTokens: 98765,
		CompactionBailReasons:     map[string]uint64{"under_budget": 1, "burst_unprofitable": 1},
	}
}

// TestNewGuardCompactionWitnessFold pins the pure fold: every counter lands on the row
// under the caller's clock and session, with no I/O.
func TestNewGuardCompactionWitnessFold(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 30, 0, 0, time.UTC)
	row := newGuardCompactionWitness("  sess-1  ", guardCompactionAnchorHead, compactionWitnessFixture(), now)

	if row.Schema != guardCompactionWitnessSchema {
		t.Errorf("schema = %d, want %d", row.Schema, guardCompactionWitnessSchema)
	}
	// The session id is trimmed: it is the join key a later audit looks up by, so
	// stray whitespace must never fork one session into two rows.
	if row.Session != "sess-1" {
		t.Errorf("session = %q, want %q (trimmed)", row.Session, "sess-1")
	}
	if row.RecordedAt != "2026-08-05T12:30:00Z" {
		t.Errorf("recorded_at = %q, want the caller's clock in RFC3339 UTC", row.RecordedAt)
	}
	if row.AnchorMode != guardCompactionAnchorHead {
		t.Errorf("anchor_mode = %q, want %q", row.AnchorMode, guardCompactionAnchorHead)
	}
	for _, c := range []struct {
		field string
		got   uint64
		want  uint64
	}{
		{"fired", row.Fired, 3},
		{"bailed", row.Bailed, 2},
		{"off", row.Off, 1},
		{"anchor_starved", row.AnchorStarved, 4},
		{"solvency_forced", row.SolvencyForced, 2},
		{"shed_tokens", row.ShedTokens, 12345},
		{"cache_read_at_fire", row.CacheReadAtFire, 98765},
	} {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.field, c.got, c.want)
		}
	}
	if row.Budget != 60000 {
		t.Errorf("budget = %d, want 60000", row.Budget)
	}
}

// TestNewGuardCompactionWitnessCopiesBailReasons pins the invariant the fold's comment
// calls out: the bail-reason map is COPIED, not aliased. A durable row must not share
// mutable state with the live gateway it is meant to outlive — an aliased map would let
// the process mutate a row that has already been "recorded".
func TestNewGuardCompactionWitnessCopiesBailReasons(t *testing.T) {
	sum := compactionWitnessFixture()
	row := newGuardCompactionWitness("sess-1", guardCompactionAnchorHead, sum, time.Now())
	if row.BailReasons["under_budget"] != 1 {
		t.Fatalf("bail_reasons not folded: %+v", row.BailReasons)
	}
	// Mutate the SOURCE after the fold. An alias would show through.
	sum.CompactionBailReasons["under_budget"] = 999
	sum.CompactionBailReasons["late_addition"] = 7
	if got := row.BailReasons["under_budget"]; got != 1 {
		t.Errorf("row aliased the gateway's map: under_budget = %d after mutation, want 1", got)
	}
	if _, ok := row.BailReasons["late_addition"]; ok {
		t.Error("row aliased the gateway's map: a key added after the fold appeared on the row")
	}
}

// TestGuardCompactionWitnessLedgerRoundTrip witnesses the append-only contract end to
// end: two sessions append, both read back in append order, and the newest row for a
// session is the one the join returns.
func TestGuardCompactionWitnessLedgerRoundTrip(t *testing.T) {
	// A nested dir that does not exist yet — the writer must create it.
	path := filepath.Join(t.TempDir(), "nested", "compaction-health.jsonl")
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)

	first := newGuardCompactionWitness("sess-A", guardCompactionAnchorHead, compactionWitnessFixture(), now)
	if err := appendGuardCompactionWitnessTo(path, first); err != nil {
		t.Fatalf("append first: %v", err)
	}
	other := newGuardCompactionWitness("sess-B", guardCompactionAnchorFirst, gateway.AdjudicationSummary{CompactionOff: 5}, now.Add(time.Minute))
	if err := appendGuardCompactionWitnessTo(path, other); err != nil {
		t.Fatalf("append second: %v", err)
	}
	// A SECOND row for sess-A — a re-run of the same session id. The join must return
	// this one, not the earlier one.
	newer := newGuardCompactionWitness("sess-A", guardCompactionAnchorHead, gateway.AdjudicationSummary{CompactionFired: 42}, now.Add(2*time.Minute))
	if err := appendGuardCompactionWitnessTo(path, newer); err != nil {
		t.Fatalf("append third: %v", err)
	}

	rows, err := readGuardCompactionWitnesses(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("read %d rows, want 3 (append-only, nothing rewritten)", len(rows))
	}
	if rows[0].Session != "sess-A" || rows[1].Session != "sess-B" || rows[2].Session != "sess-A" {
		t.Errorf("rows out of append order: %q %q %q", rows[0].Session, rows[1].Session, rows[2].Session)
	}
	got, ok := latestGuardCompactionWitness(path, "sess-A")
	if !ok {
		t.Fatal("latest(sess-A) not found")
	}
	if got.Fired != 42 {
		t.Errorf("latest(sess-A).fired = %d, want 42 (the NEWEST row, not the first)", got.Fired)
	}
	if _, ok := latestGuardCompactionWitness(path, "sess-missing"); ok {
		t.Error("latest() found a session that was never recorded")
	}
	// An empty session id is not a wildcard that matches the first row.
	if _, ok := latestGuardCompactionWitness(path, "   "); ok {
		t.Error("latest() treated a blank session id as a match")
	}
}

// TestReadGuardCompactionWitnessesSkipsTornRows pins the concurrency scar the reader is
// built for: every guarded session on the host appends to this one ledger, so a torn or
// blank final line must cost the reader nothing but that line.
func TestReadGuardCompactionWitnessesSkipsTornRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	good := `{"schema":1,"recorded_at":"2026-08-05T12:00:00Z","session":"sess-A","anchor_mode":"head","fired":3}`
	torn := `{"schema":1,"session":"sess-B","fir` // a half-written append
	body := good + "\n\n" + "not json at all\n" + good + "\n" + torn + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, err := readGuardCompactionWitnesses(path)
	if err != nil {
		t.Fatalf("read returned an error for a scarred ledger: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("read %d rows, want 2 intact rows (blank/malformed/torn skipped)", len(rows))
	}
	for _, r := range rows {
		if r.Session != "sess-A" {
			t.Errorf("unexpected row survived: %+v", r)
		}
	}
}

// TestReadGuardCompactionWitnessesMissingLedger pins that an absent ledger is an error
// to the low-level reader (the caller decides whether that is fatal) rather than a panic.
func TestReadGuardCompactionWitnessesMissingLedger(t *testing.T) {
	if _, err := readGuardCompactionWitnesses(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Error("want an error for a missing ledger, got nil")
	}
}

// TestFormatGuardCompactionWitnessBlock pins the rendered exit-summary block — the bytes
// the guard banner emits. The conditional rows appear only when their count is non-zero.
func TestFormatGuardCompactionWitnessBlock(t *testing.T) {
	row := newGuardCompactionWitness("sess-A", guardCompactionAnchorHead, compactionWitnessFixture(), time.Now())
	got := formatGuardCompactionWitness(row, "/tmp/led.jsonl")
	for _, want := range []string{
		"compaction witness (this session)",
		"sess-A",
		guardCompactionAnchorHead,
		"3 / 2 / 1", // fired / bailed / off
		"12345 tok (budget 60000)",
		"98765 tok", // cache_read priced AT THIS SESSION'S fires
		"anchor-starved",
		"x4",
		"solvency-forced",
		"x2 of 3 fired",
		"/tmp/led.jsonl", // the note names the ledger to re-read after exit
	} {
		if !strings.Contains(got, want) {
			t.Errorf("block missing %q in:\n%s", want, got)
		}
	}

	// A clean session: no starvation, no solvency override — those two rows must be
	// ABSENT rather than printed as a vacuous zero.
	clean := newGuardCompactionWitness("sess-clean", guardCompactionAnchorHead, gateway.AdjudicationSummary{CompactionFired: 1}, time.Now())
	cleanBlock := formatGuardCompactionWitness(clean, "/tmp/led.jsonl")
	for _, absent := range []string{"anchor-starved", "solvency-forced"} {
		if strings.Contains(cleanBlock, absent) {
			t.Errorf("clean session printed a vacuous %q row:\n%s", absent, cleanBlock)
		}
	}
}

// TestRecordGuardCompactionWitnessRendersFromDurableRow is the exit-funnel contract the
// guard banner depends on: record pins the row, then renders the block FROM the row that
// survived the round trip — so a non-empty block doubles as proof the witness is durable.
func TestRecordGuardCompactionWitnessRendersFromDurableRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compaction-health.jsonl")
	block := recordGuardCompactionWitness(path, "sess-A", compactionWitnessFixture(), time.Now())
	if block == "" {
		t.Fatal("record returned an empty block for a writable ledger and a live session")
	}
	if !strings.Contains(block, "sess-A") || !strings.Contains(block, "3 / 2 / 1") {
		t.Errorf("block does not carry the recorded numbers:\n%s", block)
	}
	// The row must actually be on disk — the banner's claim of durability is the point.
	rows, err := readGuardCompactionWitnesses(path)
	if err != nil || len(rows) != 1 {
		t.Fatalf("ledger round trip: rows=%d err=%v", len(rows), err)
	}
	if rows[0].Session != "sess-A" || rows[0].Fired != 3 {
		t.Errorf("durable row disagrees with the block: %+v", rows[0])
	}
}

// TestRecordGuardCompactionWitnessDegradesToSilence pins the best-effort contract: an
// observability witness must never fail a session's exit, so every unhappy path returns
// "" and the banner simply omits the block.
func TestRecordGuardCompactionWitnessDegradesToSilence(t *testing.T) {
	dir := t.TempDir()

	// No session id — nothing to key a row by.
	if got := recordGuardCompactionWitness(filepath.Join(dir, "a.jsonl"), "  ", compactionWitnessFixture(), time.Now()); got != "" {
		t.Errorf("empty session returned a block: %q", got)
	}
	// An unwritable ledger path: a regular file stands where a directory would have to
	// be, so both MkdirAll and OpenFile fail. Portable — no permission bits involved.
	blocker := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := recordGuardCompactionWitness(filepath.Join(blocker, "led.jsonl"), "sess-A", compactionWitnessFixture(), time.Now()); got != "" {
		t.Errorf("unwritable ledger returned a block: %q", got)
	}
	// An empty ledger path is refused by the writer, not panicked on.
	if got := recordGuardCompactionWitness("", "sess-A", compactionWitnessFixture(), time.Now()); got != "" {
		t.Errorf("empty ledger path returned a block: %q", got)
	}
}

// TestGuardCompactionAnchorModeName pins the flag→vocabulary mapping. The row carries
// which anchor ran because an anchor-starved count is uninterpretable without it.
func TestGuardCompactionAnchorModeName(t *testing.T) {
	if got := guardCompactionAnchorModeName(true); got != guardCompactionAnchorHead {
		t.Errorf("head-anchored = %q, want %q", got, guardCompactionAnchorHead)
	}
	if got := guardCompactionAnchorModeName(false); got != guardCompactionAnchorFirst {
		t.Errorf("legacy anchor = %q, want %q", got, guardCompactionAnchorFirst)
	}
}

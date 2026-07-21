package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// TestKnownBadCompactVerb builds a ledger with an expired signature and a live
// signature carrying a superseded row, then compacts: the expired + superseded rows
// drop, the live latest row survives, and a --dry-run first pass leaves the file
// untouched (#3471).
func TestKnownBadCompactVerb(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const t0 = int64(1_700_000_000)

	// sigA: short TTL -> expired by compact time.
	rec(t, ledger, t0, "--tree", "internal/a/**", "--reason", "build", "--ttl", "100")
	// sigB: durable (ttl 0), recorded TWICE so it carries a superseded row.
	rec(t, ledger, t0, "--tree", "internal/b/**", "--reason", "test", "--ttl", "0")
	rec(t, ledger, t0+1, "--tree", "internal/b/**", "--reason", "test", "--ttl", "0")

	if got := ledgerRows(t, ledger); got != 3 {
		t.Fatalf("setup: ledger has %d rows, want 3", got)
	}

	const now = t0 + 200 // past sigA's TTL, sigB is durable

	// --dry-run must not touch the file.
	var dOut, dErr bytes.Buffer
	if rc := runKnownBad(&dOut, &dErr, []string{"compact", "--ledger", ledger, "--dry-run"}, now); rc != 0 {
		t.Fatalf("compact --dry-run rc=%d stderr=%q", rc, dErr.String())
	}
	if got := ledgerRows(t, ledger); got != 3 {
		t.Fatalf("--dry-run rewrote the ledger: %d rows, want 3 untouched", got)
	}

	// Real compact: 3 -> 1 (drop sigA expired + sigB's superseded row).
	var cOut, cErr bytes.Buffer
	var res knownBadCompactResult
	if rc := runKnownBad(&cOut, &cErr, []string{"compact", "--ledger", ledger, "--json"}, now); rc != 0 {
		t.Fatalf("compact rc=%d stderr=%q", rc, cErr.String())
	}
	if err := json.Unmarshal(cOut.Bytes(), &res); err != nil {
		t.Fatalf("compact --json invalid: %v (%q)", err, cOut.String())
	}
	if !res.Wrote || res.Stats.KeptRows != 1 || res.Stats.InputRows != 3 {
		t.Fatalf("compact stats = %+v, want Wrote true Kept 1 Input 3", res.Stats)
	}
	if res.Stats.ExpiredDropped != 1 || res.Stats.SupersededDropped != 1 || res.Stats.LiveKept != 1 {
		t.Fatalf("compact breakdown = %+v", res.Stats)
	}

	// The one surviving row is the live durable sigB.
	recs, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(recs) != 1 || recs[0].ReasonClass != "test" {
		t.Fatalf("compacted ledger = %+v, want single reason=test row", recs)
	}

	// Match confirms the expired signature is gone and the live one still holds.
	if rc := matchRC(t, ledger, now, "internal/a/**"); rc != 0 {
		t.Errorf("expired tree still matches after compact (rc=%d, want 0)", rc)
	}
	if rc := matchRC(t, ledger, now, "internal/b/**"); rc != 3 {
		t.Errorf("live tree stopped matching after compact (rc=%d, want 3)", rc)
	}

	// A second compact is a clean no-op (already minimal): Wrote false, file unchanged.
	var c2Out, c2Err bytes.Buffer
	var res2 knownBadCompactResult
	if rc := runKnownBad(&c2Out, &c2Err, []string{"compact", "--ledger", ledger, "--json"}, now); rc != 0 {
		t.Fatalf("second compact rc=%d stderr=%q", rc, c2Err.String())
	}
	if err := json.Unmarshal(c2Out.Bytes(), &res2); err != nil {
		t.Fatalf("second compact --json invalid: %v", err)
	}
	if res2.Wrote {
		t.Errorf("second compact rewrote an already-minimal ledger: %+v", res2)
	}
}

// TestReadKnownBadLedgerCached proves the dispatch hot-path read returns the cached
// parse (same backing slice) when the file is unchanged, re-parses on an append
// (size changes), and clears the entry when the file disappears (#3471).
func TestReadKnownBadLedgerCached(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	mk := func(tree string, at int64) knownbad.Record {
		return knownbad.NewRecord("build", []string{tree}, "", "agent", "", at, 0)
	}
	if err := appendKnownBadRow(ledger, mk("internal/a/**", 1)); err != nil {
		t.Fatalf("append 1: %v", err)
	}

	r1, err := readKnownBadLedgerCached(ledger)
	if err != nil || len(r1) != 1 {
		t.Fatalf("first cached read = %d rows, err %v", len(r1), err)
	}
	// Unchanged file -> cache hit -> SAME backing array (no re-parse/alloc).
	r2, err := readKnownBadLedgerCached(ledger)
	if err != nil || len(r2) != 1 {
		t.Fatalf("second cached read = %d rows, err %v", len(r2), err)
	}
	if &r1[0] != &r2[0] {
		t.Errorf("cache miss on an unchanged ledger: got a fresh parse, want the cached slice")
	}

	// Append grows the size -> cache invalidated -> fresh parse with the new row.
	if err := appendKnownBadRow(ledger, mk("internal/b/**", 2)); err != nil {
		t.Fatalf("append 2: %v", err)
	}
	r3, err := readKnownBadLedgerCached(ledger)
	if err != nil || len(r3) != 2 {
		t.Fatalf("post-append cached read = %d rows, err %v", len(r3), err)
	}
	if &r3[0] == &r1[0] {
		t.Errorf("stale cache after append: returned the old slice")
	}

	// A vanished ledger reads as (nil, nil) and drops the stale entry.
	if err := os.Remove(ledger); err != nil {
		t.Fatalf("remove: %v", err)
	}
	r4, err := readKnownBadLedgerCached(ledger)
	if err != nil || r4 != nil {
		t.Errorf("missing ledger cached read = %v / %v, want nil/nil", r4, err)
	}
}

// rec records one known-bad row through the shell, failing the test on a non-zero rc.
func rec(t *testing.T, ledger string, now int64, extra ...string) {
	t.Helper()
	args := append([]string{"record", "--ledger", ledger}, extra...)
	var out, errb bytes.Buffer
	if rc := runKnownBad(&out, &errb, args, now); rc != 0 {
		t.Fatalf("record %v rc=%d stderr=%q", extra, rc, errb.String())
	}
}

// ledgerRows counts the parsed rows currently on the ledger.
func ledgerRows(t *testing.T, ledger string) int {
	t.Helper()
	recs, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	return len(recs)
}

// matchRC runs `knownbad match` and returns its exit code (3 = matched, 0 = clear).
func matchRC(t *testing.T, ledger string, now int64, tree string) int {
	t.Helper()
	var out, errb bytes.Buffer
	return runKnownBad(&out, &errb, []string{"match", "--tree", tree, "--ledger", ledger}, now)
}

package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// TestToDebezium_Mutation is the golden test for a coherence-bus mutation: op="u",
// the typed payload in `after`, `before` null, and ts_ms=0 (the bus is not
// wall-clock stamped). The exact JSON is pinned so a shape/tag drift fails loudly.
func TestToDebezium_Mutation(t *testing.T) {
	ev := CoherenceEvent{
		Kind:       "mutation",
		Seq:        42,
		Tool:       "write_file",
		Tags:       []string{"cache:index", "cache:plan"},
		WorldVer:   7,
		TrustEpoch: 3,
		principal:  "tenant-a",
	}
	const golden = `{"op":"u","ts_ms":0,"before":null,"after":{"kind":"mutation","tool":"write_file","tags":["cache:index","cache:plan"]},"source":{"connector":"fak","name":"fak.changes","ts_ms":0,"lsn":42,"world_ver":7,"trust_epoch":3,"principal":"tenant-a"}}`
	assertGoldenJSON(t, ToDebezium(ev), golden)
}

// TestToDebezium_Revocation is the golden test for a tombstone: op="d", the
// refuted witness (the delete key) in `before`, `after` null. This is the
// honesty rung — a delete carries the real revoked identity, never a fabricated
// after-image.
func TestToDebezium_Revocation(t *testing.T) {
	ev := CoherenceEvent{
		Kind:       "revocation",
		Seq:        43,
		Witness:    "sha256:deadbeef",
		Evicted:    5,
		WorldVer:   8,
		TrustEpoch: 4,
	}
	const golden = `{"op":"d","ts_ms":0,"before":{"kind":"revocation","witness":"sha256:deadbeef","evicted":5},"after":null,"source":{"connector":"fak","name":"fak.changes","ts_ms":0,"lsn":43,"world_ver":8,"trust_epoch":4}}`
	assertGoldenJSON(t, ToDebezium(ev), golden)
}

// TestRowToDebezium is the golden test for a durable journal row: op="c" (an
// append to the immutable ledger is a genuine insert), a REAL ts_ms sourced from
// Row.TSUnixNano (ns → ms), and the chain hashes carried so a sink can verify the
// ledger. This is the ts_ms-honesty contrast with the un-stamped coherence bus.
func TestRowToDebezium(t *testing.T) {
	r := journal.Row{
		Seq:        9,
		TSUnixNano: 1_700_000_000_123_456_789, // → 1_700_000_000_123 ms (integer ns/1e6)
		Kind:       "QUARANTINE",
		Tool:       "read_file",
		TraceID:    "sess-1",
		Verdict:    "REFUSE",
		Reason:     "REQUIRE_WITNESS",
		Witness:    "poisoned-readme",
		Hash:       "h2",
		PrevHash:   "h1",
	}
	const golden = `{"op":"c","ts_ms":1700000000123,"before":null,"after":{"kind":"QUARANTINE","tool":"read_file","trace_id":"sess-1","verdict":"REFUSE","reason":"REQUIRE_WITNESS","witness":"poisoned-readme","hash":"h2","prev_hash":"h1"},"source":{"connector":"fak","name":"fak.events","ts_ms":1700000000123,"lsn":9,"world_ver":0,"trust_epoch":0}}`
	assertGoldenJSON(t, RowToDebezium(r), golden)
}

// assertGoldenJSON marshals got and compares it byte-for-byte to want. Field
// order matters: Go marshals struct fields in declaration order, so the golden
// string doubles as a wire-shape lock (the Debezium envelope key order a sink
// sees).
func assertGoldenJSON(t *testing.T, got DebeziumEnvelope, want string) {
	t.Helper()
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(b) != want {
		t.Errorf("Debezium envelope mismatch\n got: %s\nwant: %s", b, want)
	}
}

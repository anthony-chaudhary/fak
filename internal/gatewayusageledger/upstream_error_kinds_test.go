package gatewayusageledger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// upstream_error_kinds_test.go holds the #5487 witnesses that are deliberately written
// WITHOUT naming Counters.UpstreamErrorKinds, so the whole file still compiles against the
// pre-fix ledger.go. That is what makes the fail-before/pass-after proof runnable: overlay
// ledger.go back to its HEAD content and these tests still BUILD, and the round-trip one
// FAILS with a real assertion rather than a build error. Every assertion here goes through
// the JSONL bytes, which is the durable contract that actually matters — the ledger is a
// file, and what a second reader sees is the line, not the struct.

// stalledLedgerLine is one persisted "exit" row from a `fak guard` session that hit three
// failed turns: TWO of them stalled (the idle-deadline detector in internal/agent fired)
// and one a 5xx. upstream_error_kinds is what makes the two stalls ATTRIBUTABLE after the
// process is gone. The `errored` counter beside it is deliberately an UNRELATED population
// — kernel adjudication ERROR verdicts, not upstream turn failures — which is why nothing
// in a pre-#5487 row moved at all when a turn stalled.
//
// Its row_key is synthetic and never asserted here (the round-trip test does not exercise
// the idempotency key); the key IS pinned, against real counters, in legacyLedgerLine below.
const stalledLedgerLine = `{"schema":"fak-gateway-usage-ledger/1","row_key":"synthetic-not-asserted","kind":"exit","session_type":"guard","context":"claude","session_id":"sess-stall","pid":4242,"unix_millis":1700000000000,"uptime_seconds":61.5,"counters":{"submits":0,"vdso_hits":0,"engine_calls":0,"denies":0,"transforms":0,"quarantines":0,"result_denies":0,"admitted":0,"total":9,"allowed":6,"denied":0,"transformed":0,"quarantined":0,"deferred":0,"escalated":0,"errored":3,"observed_turns":9,"input_tokens":100,"output_tokens":20,"cached_prompt_tokens":0,"cached_turns":0,"cache_creation_tokens":0,"kv_prefix_prompt_tokens":0,"kv_prefix_reused_tokens":0,"compaction_fired":0,"compaction_bailed":0,"compaction_off":0,"compaction_dropped_turns":0,"compaction_shed_tokens":0,"compaction_cache_read_tokens":0,"tool_prune_turns":0,"tool_prune_count":0,"deny_all_stops":1,"cache_ttl_upgrades_upgraded":0,"upstream_error_kinds":{"stalled":2,"status_5xx":1}},"generated_at":"2023-11-14T22:13:20Z"}`

// legacyLedgerLine is the SAME row as it was written BEFORE this field existed: identical
// in every other byte, with no upstream_error_kinds key at all. It is the back-compat
// fixture — a row persisted by an older binary must keep reading back with exactly its
// original meaning, and its idempotency key must not move. Its row_key is the REAL one the
// pre-#5487 computeRowKey stamps for these counters, so the pinning test below can compare
// the recomputed key against the row's own stamp rather than only against a constant.
const legacyLedgerLine = `{"schema":"fak-gateway-usage-ledger/1","row_key":"eaf4c3ed1b75d74f","kind":"exit","session_type":"guard","context":"claude","session_id":"sess-stall","pid":4242,"unix_millis":1700000000000,"uptime_seconds":61.5,"counters":{"submits":0,"vdso_hits":0,"engine_calls":0,"denies":0,"transforms":0,"quarantines":0,"result_denies":0,"admitted":0,"total":9,"allowed":6,"denied":0,"transformed":0,"quarantined":0,"deferred":0,"escalated":0,"errored":3,"observed_turns":9,"input_tokens":100,"output_tokens":20,"cached_prompt_tokens":0,"cached_turns":0,"cache_creation_tokens":0,"kv_prefix_prompt_tokens":0,"kv_prefix_reused_tokens":0,"compaction_fired":0,"compaction_bailed":0,"compaction_off":0,"compaction_dropped_turns":0,"compaction_shed_tokens":0,"compaction_cache_read_tokens":0,"tool_prune_turns":0,"tool_prune_count":0,"deny_all_stops":1,"cache_ttl_upgrades_upgraded":0},"generated_at":"2023-11-14T22:13:20Z"}`

// writeLedgerLines drops raw JSONL lines into a temp ledger file and returns its path, so
// a test can exercise the real ReadLedgerFile path over bytes it controls exactly.
func writeLedgerLines(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway-usage.jsonl")
	var buf []byte
	for _, l := range lines {
		buf = append(buf, l...)
		buf = append(buf, '\n')
	}
	if err := os.WriteFile(path, buf, 0o600); err != nil {
		t.Fatalf("write ledger fixture: %v", err)
	}
	return path
}

// countersMap re-marshals a parsed Row and hands back its "counters" object as a generic
// map. Going out through JSON (rather than reading struct fields) is the point: an unknown
// key is silently DROPPED by encoding/json on the way in, so this is what proves the kind
// actually survives read->write and is visible to any second reader of the file.
func countersMap(t *testing.T, r Row) map[string]any {
	t.Helper()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("re-marshal row: %v", err)
	}
	var generic map[string]any
	if err := json.Unmarshal(b, &generic); err != nil {
		t.Fatalf("decode re-marshaled row: %v", err)
	}
	c, ok := generic["counters"].(map[string]any)
	if !ok {
		t.Fatalf("re-marshaled row has no counters object: %s", b)
	}
	return c
}

// TestStalledUpstreamErrorKindSurvivesDurableRoundTrip is the #5487 fail-before/pass-after
// proof. The gateway already classifies a stall into its own kind (gateway.upstreamErrorKind
// returns "stalled"), but that classification lived only on the in-process /metrics counter
// and the stderr FAILED line — both of which die with a per-invocation `fak guard` gateway.
// The durable row had nowhere to put it, so afterwards a stall was invisible rather than
// merely coarse.
//
// Before the fix this FAILS: encoding/json drops the unknown upstream_error_kinds key on
// read, so the re-marshaled row loses it entirely and a stall is unattributable.
func TestStalledUpstreamErrorKindSurvivesDurableRoundTrip(t *testing.T) {
	rows := ReadLedgerFile(writeLedgerLines(t, stalledLedgerLine))
	if len(rows) != 1 {
		t.Fatalf("expected 1 parsed row, got %d", len(rows))
	}
	c := countersMap(t, rows[0])

	// The neighbouring adjudication counter is untouched. It is deliberately NOT asserted
	// to equal the kind total: `errored` counts kernel ERROR verdicts, a different
	// population from upstream turn failures, so reconciling the two would encode a
	// relationship production does not have.
	if got := c["errored"]; got != float64(3) {
		t.Fatalf("errored = %v, want 3 (an unrelated pre-existing counter must not shift)", got)
	}

	raw, ok := c["upstream_error_kinds"]
	if !ok {
		t.Fatalf("upstream_error_kinds is absent after a ledger round-trip: the durable row drops "+
			"the kind the gateway already classified, so a stall cannot be told apart from any other "+
			"upstream failure. counters=%v", c)
	}
	kinds, ok := raw.(map[string]any)
	if !ok {
		t.Fatalf("upstream_error_kinds = %T, want an object keyed by kind", raw)
	}

	// The whole point: "stalled" is its OWN attributable bucket, not folded into a generic one.
	if got := kinds["stalled"]; got != float64(2) {
		t.Fatalf("upstream_error_kinds[stalled] = %v, want 2 — the stall must round-trip as its own kind", got)
	}
	if got := kinds["status_5xx"]; got != float64(1) {
		t.Fatalf("upstream_error_kinds[status_5xx] = %v, want 1 — a co-occurring kind must stay distinct from the stall", got)
	}
	if len(kinds) != 2 {
		t.Fatalf("upstream_error_kinds = %v, want exactly the two observed kinds (nothing collapsed or invented)", kinds)
	}
}

// TestLegacyRowWithoutUpstreamErrorKindsReadsBackUnchanged is the durable-schema guarantee:
// a row written BEFORE this field existed must read back with its ORIGINAL meaning. It
// asserts the strongest available form of that — the parsed row re-marshals to a byte-for-byte
// equal JSON object (no key appears, none disappears, no value shifts), so an old row's
// interpretation is untouched and `upstream_error_kinds` stays ABSENT rather than being
// materialized as an empty object that would falsely assert "measured, zero failures".
func TestLegacyRowWithoutUpstreamErrorKindsReadsBackUnchanged(t *testing.T) {
	rows := ReadLedgerFile(writeLedgerLines(t, legacyLedgerLine))
	if len(rows) != 1 {
		t.Fatalf("expected 1 parsed legacy row, got %d", len(rows))
	}
	row := rows[0]

	// Every scalar keeps its original meaning.
	if row.Counters.Errored != 3 {
		t.Fatalf("legacy errored = %d, want 3", row.Counters.Errored)
	}
	if row.Counters.DenyAllStops != 1 {
		t.Fatalf("legacy deny_all_stops = %d, want 1 (a neighbouring counter must not shift)", row.Counters.DenyAllStops)
	}
	if row.Counters.ObservedTurns != 9 || row.Counters.InputTokens != 100 || row.Counters.OutputTokens != 20 {
		t.Fatalf("legacy counters shifted: %+v", row.Counters)
	}

	// And the WHOLE object re-serializes identically.
	reMarshaled, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("re-marshal legacy row: %v", err)
	}
	var before, after map[string]any
	if err := json.Unmarshal([]byte(legacyLedgerLine), &before); err != nil {
		t.Fatalf("decode legacy fixture: %v", err)
	}
	if err := json.Unmarshal(reMarshaled, &after); err != nil {
		t.Fatalf("decode re-marshaled legacy row: %v", err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("a pre-field row did not round-trip unchanged.\n before: %s\n  after: %s", legacyLedgerLine, reMarshaled)
	}
	if c, ok := after["counters"].(map[string]any); ok {
		if _, present := c["upstream_error_kinds"]; present {
			t.Fatalf("upstream_error_kinds was materialized on a legacy row — absent must keep reading NOT INSTRUMENTED, not zero failures")
		}
	}
}

// TestLegacyRowKeyIsUnmovedByTheNewField pins the idempotency key. RowKey hashes the
// json.Marshal of Counters (computeRowKey), so a NON-omitempty addition would rewrite every
// historical row's key and make already-folded rows look like new snapshots — silently
// double-counting a corpus nobody can re-derive. The golden below was captured from the
// pre-#5487 ledger.go; it must still hold, which is what proves the field is genuinely
// additive on the wire.
func TestLegacyRowKeyIsUnmovedByTheNewField(t *testing.T) {
	rows := ReadLedgerFile(writeLedgerLines(t, legacyLedgerLine))
	if len(rows) != 1 {
		t.Fatalf("expected 1 parsed legacy row, got %d", len(rows))
	}
	row := rows[0]
	const goldenPreFieldKey = "eaf4c3ed1b75d74f"
	got := computeRowKey(Schema, row.SessionID, row.PID, row.UnixMillis, row.Counters)
	if got != goldenPreFieldKey {
		t.Fatalf("RowKey over legacy counters = %q, want the pre-field golden %q — adding the kind map "+
			"moved the idempotency key, so every historical row would re-enter the fold as a new snapshot", got, goldenPreFieldKey)
	}
	// The same check stated against the row's OWN stamp: a persisted row must still hash to
	// the key it was written with, which is what DedupeByKey relies on to collapse a retried
	// flush instead of double-counting it.
	if got != row.RowKey {
		t.Fatalf("recomputed RowKey %q != the key stamped on the persisted row %q", got, row.RowKey)
	}
}

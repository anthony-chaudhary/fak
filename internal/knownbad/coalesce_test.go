package knownbad

import (
	"strings"
	"testing"
)

// crashEvents builds n identical crash events sharing one cause, spaced one second
// apart so the newest observation instant is deterministic.
func crashEvents(n int, class string, globs []string, hash string, startUnix int64) []CrashEvent {
	out := make([]CrashEvent, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, CrashEvent{
			ReasonClass: class,
			TreeGlobs:   globs,
			FailureHash: hash,
			AtUnix:      startUnix + int64(i),
			ObservedBy:  "worker-1",
		})
	}
	return out
}

// ACCEPTANCE 1 (#3586): 100 synthetic CHILD_CRASH events sharing a signature produce
// ONE deduped row with occurrence_count=100 — not 100 rows and not 100 issues.
func TestCoalesceHundredCrashesToOneCountedRow(t *testing.T) {
	const now int64 = 1_000_000
	// Deliberately supply the tree globs in a form that must normalize to the same
	// signature: a worker reporting "internal/foo/**" and one reporting
	// "internal/foo" are the same cause.
	events := crashEvents(100, "SIGNAL_CRASH", []string{"internal/foo/**"}, "sha256:deadbeef", now)

	rows, stats := CoalesceCrashes(nil, events, now, 900)

	if len(rows) != 1 {
		t.Fatalf("coalesced 100 same-cause crashes into %d rows, want exactly 1: %+v", len(rows), rows)
	}
	if got := rows[0].OccurrenceCount; got != 100 {
		t.Errorf("occurrence_count = %d, want 100", got)
	}
	if !rows[0].Coalesced() {
		t.Errorf("a 100-occurrence row must report Coalesced()")
	}
	// The row is a well-formed, live known-bad over the crashed tree.
	if rows[0].Status != StatusOpen || !rows[0].Live(now) {
		t.Errorf("coalesced row is not a live open record: status=%q live=%v", rows[0].Status, rows[0].Live(now))
	}
	if len(rows[0].TreeGlobs) != 1 || rows[0].TreeGlobs[0] != "internal/foo" {
		t.Errorf("tree globs = %v, want the normalized [internal/foo]", rows[0].TreeGlobs)
	}
	// The last-seen instant is the NEWEST event, not the first or the emit clock.
	if got, want := rows[0].LastSeenAtUnix, now+99; got != want {
		t.Errorf("last_seen_at_unix = %d, want %d (the newest folded event)", got, want)
	}
	if stats.Events != 100 || stats.Rows != 1 || stats.Signatures != 1 || stats.Opened != 1 || stats.Refreshed != 0 {
		t.Errorf("stats = %+v, want 100 events folded to 1 opened row", stats)
	}
	// The whole point: the operator surface shows ONE live signature, not 100.
	if live := LiveRecords(rows, now); len(live) != 1 {
		t.Errorf("LiveRecords over the coalesced output = %d rows, want 1", len(live))
	}
}

// ACCEPTANCE 2 (#3586): two DISTINCT signatures produce two rows — the coalescer
// dedups a cause, it never merges distinct causes. Each axis of the signature
// (reason class, tree, failure hash) must independently split.
func TestCoalesceDistinctSignaturesStaySeparate(t *testing.T) {
	const now int64 = 2_000_000
	base := crashEvents(3, "SIGNAL_CRASH", []string{"internal/foo"}, "sha256:aaa", now)

	cases := []struct {
		name  string
		other []CrashEvent
	}{
		{"different reason class", crashEvents(5, "OOM", []string{"internal/foo"}, "sha256:aaa", now)},
		{"different tree", crashEvents(5, "SIGNAL_CRASH", []string{"internal/bar"}, "sha256:aaa", now)},
		{"different failure hash", crashEvents(5, "SIGNAL_CRASH", []string{"internal/foo"}, "sha256:bbb", now)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows, stats := CoalesceCrashes(nil, append(append([]CrashEvent{}, base...), tc.other...), now, 900)
			if len(rows) != 2 {
				t.Fatalf("two distinct causes coalesced into %d rows, want 2: %+v", len(rows), rows)
			}
			if rows[0].Signature == rows[1].Signature {
				t.Fatalf("distinct causes collapsed onto one signature %q", rows[0].Signature)
			}
			if rows[0].OccurrenceCount != 3 || rows[1].OccurrenceCount != 5 {
				t.Errorf("counts = %d/%d, want 3/5 (each cause counted on its own)",
					rows[0].OccurrenceCount, rows[1].OccurrenceCount)
			}
			if stats.Signatures != 2 || stats.Opened != 2 {
				t.Errorf("stats = %+v, want 2 distinct signatures both opened", stats)
			}
		})
	}
}

// ACCEPTANCE 3 (#3586): re-emission WITHIN the window increments the count and does
// not duplicate — the refreshed row supersedes the prior one, so the live view still
// shows exactly one row per signature.
func TestCoalesceReEmissionIncrementsWithoutDuplicating(t *testing.T) {
	const now int64 = 3_000_000
	const ttl int64 = 900
	globs := []string{"internal/foo"}

	first, _ := CoalesceCrashes(nil, crashEvents(40, "OOM", globs, "", now), now, ttl)
	if len(first) != 1 || first[0].OccurrenceCount != 40 {
		t.Fatalf("first pass = %+v, want one row counting 40", first)
	}

	// The shell appends the coalesced row; a later pass reads it back as prior state.
	ledger := append([]Record{}, first...)

	// A second storm 60s later, still inside the 900s window.
	later := now + 60
	second, stats := CoalesceCrashes(ledger, crashEvents(60, "OOM", globs, "", later), later, ttl)
	if len(second) != 1 {
		t.Fatalf("re-emission produced %d rows, want 1 (a refresh, not a second signature)", len(second))
	}
	if got := second[0].OccurrenceCount; got != 100 {
		t.Errorf("occurrence_count after re-emission = %d, want 100 (40 + 60)", got)
	}
	if second[0].Signature != first[0].Signature {
		t.Errorf("refresh changed the signature: %q -> %q", first[0].Signature, second[0].Signature)
	}
	if stats.Refreshed != 1 || stats.Opened != 0 {
		t.Errorf("stats = %+v, want the signature refreshed rather than reopened", stats)
	}
	// The window stays anchored at FIRST sighting, so a sustained storm cannot renew
	// its own hold forever.
	if got, want := second[0].DiscoveredAtUnix, first[0].DiscoveredAtUnix; got != want {
		t.Errorf("refresh slid the window: discovered_at %d -> %d", want, got)
	}
	// Last-seen moved forward to the newest event of the second storm.
	if got, want := second[0].LastSeenAtUnix, later+59; got != want {
		t.Errorf("last_seen_at_unix = %d, want %d", got, want)
	}

	// NOT A DUPLICATE: appended to the ledger, the supersede-aware live view still
	// collapses to exactly ONE row for the cause — one issue, not two.
	ledger = append(ledger, second...)
	live := LiveRecords(ledger, later)
	if len(live) != 1 {
		t.Fatalf("ledger shows %d live rows for one cause, want 1 (append-to-supersede)", len(live))
	}
	if live[0].OccurrenceCount != 100 {
		t.Errorf("the live row's count = %d, want the cumulative 100", live[0].OccurrenceCount)
	}
}

// Once the window LAPSES, a fresh storm opens a NEW row starting a new count — the
// "per window" half of "one row per signature per window". A resolved signature
// likewise reopens: a crash after a resolve is evidence the failure came back.
func TestCoalesceOpensNewWindowAfterLapseOrResolve(t *testing.T) {
	const now int64 = 4_000_000
	const ttl int64 = 900
	globs := []string{"internal/foo"}

	first, _ := CoalesceCrashes(nil, crashEvents(10, "OOM", globs, "", now), now, ttl)

	// Well past the TTL: the prior row is no longer live, so the count restarts.
	after := now + ttl + 1
	expired, stats := CoalesceCrashes(first, crashEvents(7, "OOM", globs, "", after), after, ttl)
	if len(expired) != 1 || expired[0].OccurrenceCount != 7 {
		t.Fatalf("post-expiry pass = %+v, want a fresh row counting 7", expired)
	}
	if stats.Opened != 1 || stats.Refreshed != 0 {
		t.Errorf("stats = %+v, want a NEW window opened after the TTL lapsed", stats)
	}
	if expired[0].Signature != first[0].Signature {
		t.Errorf("a new window must keep the same signature (same cause)")
	}

	// A resolved signature also reopens on a fresh crash.
	resolved := append([]Record{}, first...)
	resolved = append(resolved, first[0].WithResolve("fixer", now+10, "tests"))
	reopened, rstats := CoalesceCrashes(resolved, crashEvents(4, "OOM", globs, "", now+20), now+20, ttl)
	if len(reopened) != 1 || reopened[0].OccurrenceCount != 4 || reopened[0].Status != StatusOpen {
		t.Fatalf("post-resolve pass = %+v, want one fresh OPEN row counting 4", reopened)
	}
	if rstats.Opened != 1 {
		t.Errorf("stats = %+v, want the resolved signature reopened", rstats)
	}
}

// An event whose tree globs all normalize away can never match a query, so it is
// dropped — but COUNTED, never silently swallowed. A blank reason class buckets into
// UNCLASSIFIED rather than splitting one cause across signatures.
func TestCoalesceDropsTreelessEventsLoudlyAndBucketsBlankClass(t *testing.T) {
	const now int64 = 5_000_000
	events := []CrashEvent{
		{ReasonClass: "OOM", TreeGlobs: []string{"**"}, AtUnix: now},        // bare star -> no tree
		{ReasonClass: "OOM", TreeGlobs: []string{"../escape"}, AtUnix: now}, // escapes the root
		{ReasonClass: "OOM", TreeGlobs: nil, AtUnix: now},                   // no tree at all
		{ReasonClass: "", TreeGlobs: []string{"internal/foo"}, AtUnix: now},
		{ReasonClass: "   ", TreeGlobs: []string{"internal/foo"}, AtUnix: now},
	}
	rows, stats := CoalesceCrashes(nil, events, now, 900)

	if stats.DroppedNoTree != 3 {
		t.Errorf("dropped_no_tree = %d, want 3 (drops are counted, not silent)", stats.DroppedNoTree)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1 (both blank classes bucket together): %+v", len(rows), rows)
	}
	if rows[0].ReasonClass != CrashClassUnclassified || rows[0].OccurrenceCount != 2 {
		t.Errorf("row = %+v, want the UNCLASSIFIED bucket counting 2", rows[0])
	}
}

// A crash event's Signature is the SAME fold the ledger, the dispatcher hold, and
// the resume backoff key on — so a parked class and a coalesced crash row name the
// same signature. Order and redundant glob stars must not change it.
func TestCrashEventSignatureMatchesLedgerSignature(t *testing.T) {
	ev := CrashEvent{
		ReasonClass: "SIGNAL_CRASH",
		TreeGlobs:   []string{"internal/bar/**", "internal/foo"},
		FailureHash: "sha256:abc",
	}
	want := Signature("SIGNAL_CRASH", []string{"internal/foo", "internal/bar"}, "sha256:abc")
	if got := ev.Signature(); got != want {
		t.Errorf("CrashEvent.Signature() = %q, want the ledger Signature() %q", got, want)
	}
	// A blank class folds to UNCLASSIFIED on BOTH sides of the seam.
	blank := CrashEvent{TreeGlobs: []string{"internal/foo"}}
	if got, want := blank.Signature(), Signature(CrashClassUnclassified, []string{"internal/foo"}, ""); got != want {
		t.Errorf("blank-class signature = %q, want the UNCLASSIFIED fold %q", got, want)
	}
}

// The load-bearing backward-compat property (same contract WithDerivedFrom holds):
// a row with no coalesced count is byte-identical to a pre-#3586 row — both new keys
// are omitempty — and a counted row survives a JSONL round-trip.
func TestOccurrenceFieldsAreBackwardCompatible(t *testing.T) {
	plain := NewRecord("build", []string{"internal/foo/**"}, "n", "a1", "", 42, 0)
	line, err := MarshalLine(plain)
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	if strings.Contains(line, "occurrence_count") || strings.Contains(line, "last_seen_at_unix") {
		t.Errorf("an uncoalesced row must omit both keys (not byte-identical to a pre-#3586 row): %s", line)
	}
	if plain.Coalesced() {
		t.Errorf("a zero-count row must not report Coalesced()")
	}

	counted := plain.WithOccurrences(300, 999)
	got := ParseLedger([]byte(mustLine(t, counted)))
	if len(got) != 1 || got[0].OccurrenceCount != 300 || got[0].LastSeenAtUnix != 999 {
		t.Fatalf("round-trip lost the coalesced count: %+v", got)
	}

	// Last-seen only moves forward; a stale batch cannot rewind it. A negative count clamps.
	if rewound := counted.WithOccurrences(301, 5); rewound.LastSeenAtUnix != 999 {
		t.Errorf("last_seen_at_unix rewound to %d, want it pinned at 999", rewound.LastSeenAtUnix)
	}
	if neg := counted.WithOccurrences(-4, 0); neg.OccurrenceCount != 0 {
		t.Errorf("a negative count = %d, want it clamped to 0", neg.OccurrenceCount)
	}
}

func mustLine(t *testing.T, rec Record) string {
	t.Helper()
	line, err := MarshalLine(rec)
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	return line
}

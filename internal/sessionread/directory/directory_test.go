package directory

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// fixtureIdentity is the durable UUID<->trace join set the directory reads. It stands in
// for resume_identity.jsonl (append-only), one row per observed pairing.
func fixtureIdentity() []IdentityRow {
	return []IdentityRow{
		{UUID: "uuid-A", Trace: "trace-A", Handle: "sesh-A", Account: "acct-1", Via: "guard SessionStart"},
		{UUID: "uuid-B", Trace: "trace-B", Handle: "sesh-B", Via: "guard SessionStart"},
		{UUID: "", Trace: "trace-half"}, // half row: must never fold a join
		{UUID: "uuid-half", Trace: ""},  // half row: the reverse
	}
}

// TestResolveIdentityBothDirections witnesses done-condition bullet 1: an external
// process resolves a Claude Code transcript UUID to a trace AND back, and an unknown id
// misses in the closed read-plane vocabulary (READ_UNKNOWN_TRACE).
func TestResolveIdentityBothDirections(t *testing.T) {
	rows := fixtureIdentity()

	// uuid -> trace
	fwd := ResolveIdentity(rows, "uuid-A")
	if !fwd.OK || fwd.Paired != "trace-A" || fwd.Direction != "uuid->trace" {
		t.Fatalf("uuid->trace resolve = %+v, want Paired=trace-A dir=uuid->trace OK", fwd)
	}
	if fwd.Row.Via != "guard SessionStart" || fwd.Row.Handle != "sesh-A" {
		t.Fatalf("provenance not surfaced: Via=%q Handle=%q", fwd.Row.Via, fwd.Row.Handle)
	}

	// trace -> uuid (the "and back" direction)
	rev := ResolveIdentity(rows, "trace-A")
	if !rev.OK || rev.Paired != "uuid-A" || rev.Direction != "trace->uuid" {
		t.Fatalf("trace->uuid resolve = %+v, want Paired=uuid-A dir=trace->uuid OK", rev)
	}

	// unknown id -> closed miss with READ_UNKNOWN_TRACE
	miss := ResolveIdentity(rows, "nope-nothing")
	if miss.OK || miss.Reason != sessionread.ReasonReadUnknownTrace {
		t.Fatalf("unknown resolve = %+v, want OK=false Reason=%s", miss, sessionread.ReasonReadUnknownTrace)
	}
	// blank id is also a closed miss, never a spurious join
	blank := ResolveIdentity(rows, "  ")
	if blank.OK || blank.Reason != sessionread.ReasonReadUnknownTrace {
		t.Fatalf("blank resolve = %+v, want OK=false Reason=%s", blank, sessionread.ReasonReadUnknownTrace)
	}

	// a half row (missing an endpoint) never resolves in either direction
	if m := ResolveIdentity(rows, "trace-half"); m.OK {
		t.Fatalf("half row trace-half must not resolve, got %+v", m)
	}
	if m := ResolveIdentity(rows, "uuid-half"); m.OK {
		t.Fatalf("half row uuid-half must not resolve, got %+v", m)
	}
}

// TestFoldIdentityLastWriteWins pins the append-only fold rule and half-row skip that
// the whole directory bridge rests on.
func TestFoldIdentityLastWriteWins(t *testing.T) {
	rows := []IdentityRow{
		{UUID: "u1", Trace: "t-old"},
		{UUID: "u1", Trace: "t-new"}, // re-paired: newest wins
		{UUID: "", Trace: "t-half"},  // skipped both ways
	}
	traceByUUID, uuidByTrace := FoldIdentity(rows)
	if traceByUUID["u1"] != "t-new" {
		t.Fatalf("last-write-wins failed: traceByUUID[u1]=%q want t-new", traceByUUID["u1"])
	}
	if uuidByTrace["t-new"] != "u1" {
		t.Fatalf("reverse map wrong: uuidByTrace[t-new]=%q want u1", uuidByTrace["t-new"])
	}
	if _, ok := uuidByTrace["t-half"]; ok {
		t.Fatalf("half row leaked into fold: %v", uuidByTrace)
	}
}

// TestDirectoryDoneConditionC4 is the centerpiece witness. It folds:
//   - a session present in BOTH stores (journal LIVE + drive RUNNING, same trace via
//     the identity join) -> ONE Source="both" row,
//   - a CRASHED lifecycle-only session -> Source="journal" (done-condition bullet 2a),
//   - a still-open drive-state-only session -> Source="drive" (done-condition bullet 2b),
//
// and asserts every row carries the OBSERVED qualifier, source tags are correct, and the
// lifecycle / run-state land on the right rows.
func TestDirectoryDoneConditionC4(t *testing.T) {
	identity := fixtureIdentity()

	journal := []sessionjournal.Classified{
		// uuid-A also has a drive row below (same trace via identity) -> "both"
		{Session: sessionjournal.Session{ID: "uuid-A"}, Status: sessionjournal.StatusLive, Reason: sessionjournal.ReasonLive},
		// uuid-B is lifecycle-only and CRASHED -> "journal"
		{Session: sessionjournal.Session{ID: "uuid-B"}, Status: sessionjournal.StatusCrashed, Reason: sessionjournal.ReasonPIDDead},
	}
	drive := []DriveRow{
		{TraceID: "trace-A", Run: "RUNNING", Priority: 5}, // pairs uuid-A -> "both"
		{TraceID: "trace-C", Run: "RUNNING", Priority: 9}, // still-open, drive-only -> "drive"
	}

	dir := Directory(journal, drive, identity)

	byTrace := map[string]DirectoryRow{}
	for _, r := range dir {
		// Invariant across the whole directory: every row is a live OBSERVED reading.
		if r.Evidence != sessionread.EvidenceObserved {
			t.Fatalf("row %+v: Evidence=%q want OBSERVED", r, r.Evidence)
		}
		byTrace[r.TraceID] = r
	}
	if len(dir) != 3 {
		t.Fatalf("directory has %d rows, want 3: %+v", len(dir), dir)
	}

	// (a) the merged "both" row: journal LIVE + drive RUNNING under one trace, UUID joined
	both := byTrace["trace-A"]
	if both.Source != SourceBoth {
		t.Fatalf("trace-A Source=%q want both", both.Source)
	}
	if both.UUID != "uuid-A" {
		t.Fatalf("trace-A UUID=%q want uuid-A (joined via identity map)", both.UUID)
	}
	if both.Lifecycle != sessionjournal.StatusLive || both.RunState != "RUNNING" {
		t.Fatalf("trace-A Lifecycle=%q RunState=%q want LIVE/RUNNING", both.Lifecycle, both.RunState)
	}
	if both.Priority != 5 {
		t.Fatalf("trace-A Priority=%d want 5", both.Priority)
	}

	// (b) the CRASHED lifecycle-only session, source-tagged journal
	crashed := byTrace["trace-B"]
	if crashed.Source != SourceJournal {
		t.Fatalf("trace-B Source=%q want journal", crashed.Source)
	}
	if crashed.Lifecycle != sessionjournal.StatusCrashed {
		t.Fatalf("trace-B Lifecycle=%q want CRASHED", crashed.Lifecycle)
	}
	if crashed.RunState != "" {
		t.Fatalf("trace-B RunState=%q want empty (no drive row)", crashed.RunState)
	}
	if crashed.UUID != "uuid-B" {
		t.Fatalf("trace-B UUID=%q want uuid-B (joined via identity map)", crashed.UUID)
	}

	// (c) the still-open drive-state-only session, source-tagged drive
	open := byTrace["trace-C"]
	if open.Source != SourceDrive {
		t.Fatalf("trace-C Source=%q want drive", open.Source)
	}
	if open.RunState != "RUNNING" {
		t.Fatalf("trace-C RunState=%q want RUNNING", open.RunState)
	}
	if open.Lifecycle != "" {
		t.Fatalf("trace-C Lifecycle=%q want empty (no journal row)", open.Lifecycle)
	}
}

// TestDirectoryUnknownTraceLookup witnesses the closed-miss addressing contract: a known
// trace or UUID resolves to its row; an unknown/blank id returns READ_UNKNOWN_TRACE.
func TestDirectoryUnknownTraceLookup(t *testing.T) {
	identity := fixtureIdentity()
	journal := []sessionjournal.Classified{
		{Session: sessionjournal.Session{ID: "uuid-A"}, Status: sessionjournal.StatusLive},
	}
	drive := []DriveRow{{TraceID: "trace-A", Run: "RUNNING"}}
	dir := Directory(journal, drive, identity)

	// address by trace
	if row, reason, ok := Lookup(dir, "trace-A"); !ok || reason != "" || row.Source != SourceBoth {
		t.Fatalf("Lookup(trace-A) = (%+v, %q, %v), want a both-row hit", row, reason, ok)
	}
	// address by transcript UUID (the join lets a caller reach the same row either way)
	if row, _, ok := Lookup(dir, "uuid-A"); !ok || row.TraceID != "trace-A" {
		t.Fatalf("Lookup(uuid-A) = (%+v, ok=%v), want the trace-A row", row, ok)
	}
	// unknown id: closed miss in the read-plane vocabulary
	if _, reason, ok := Lookup(dir, "ghost-trace"); ok || reason != sessionread.ReasonReadUnknownTrace {
		t.Fatalf("Lookup(unknown) reason=%q ok=%v, want READ_UNKNOWN_TRACE miss", reason, ok)
	}
	// blank id: same closed miss
	if _, reason, ok := Lookup(dir, ""); ok || reason != sessionread.ReasonReadUnknownTrace {
		t.Fatalf("Lookup(blank) reason=%q ok=%v, want READ_UNKNOWN_TRACE miss", reason, ok)
	}
}

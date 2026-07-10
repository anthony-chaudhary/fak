package transcriptfeed

// transcriptfeed_test.go witnesses the #4196 done-condition directly:
//
//   - "A peer tails a fixture session, receives turn/tool/decision events in order,
//     DROPS, and re-attaches by cursor with NO gap and NO duplicate."
//     -> TestReattachByCursorNoGapNoDuplicate
//   - "A test asserts a quarantined span never crosses the stream boundary."
//     -> TestQuarantinedSpanNeverCrossesBoundary
//   - "Reads are side-effect-free (observing cannot advance the loop)."
//     -> TestDrainIsSideEffectFree
//   - Principal-scoped re-attach stays gap-free via the cursor-over-all-retained invariant
//     -> TestPrincipalScopingCursorStaysMonotone
//   - Transcript records map to the four event kinds
//     -> TestEventsFromRecordsMapsTranscriptKinds
//
// The test is IN-package so it can witness the unexported cursor (f.seq) and routing key
// (ev.principal) the wire contract deliberately hides.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// --- re-attach with NO gap / NO duplicate -----------------------------------------------

func TestReattachByCursorNoGapNoDuplicate(t *testing.T) {
	f := NewFeed(0)

	// A peer tails from the head of the retained window.
	batch1 := []TranscriptEvent{
		{Kind: KindTurnOpen, UUID: "t1"},
		{Kind: KindDecision, UUID: "t2", Tool: "Read"},
		{Kind: KindToolTerminal, UUID: "t3", Tool: "Bash"},
	}
	for _, ev := range batch1 {
		f.Append(ev)
	}

	first, cursor := f.Drain("", 0)
	if len(first) != 3 {
		t.Fatalf("first drain returned %d events, want 3", len(first))
	}
	// Seq is feed-minted, monotone, and gap-free (1,2,3).
	for i, ev := range first {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("first drain event %d Seq=%d, want %d (monotone from 1)", i, ev.Seq, i+1)
		}
	}
	if cursor != 3 {
		t.Fatalf("first cursor = %d, want 3 (highest retained Seq)", cursor)
	}

	// The peer DROPS here, then more turns happen while it is away.
	batch2 := []TranscriptEvent{
		{Kind: KindTurnClose, UUID: "t4"},
		{Kind: KindTurnOpen, UUID: "t5"},
	}
	for _, ev := range batch2 {
		f.Append(ev)
	}

	// It RE-ATTACHES by its saved cursor. It must see ONLY the new events (no duplicate of
	// the first batch) and it must miss NONE of them (no gap).
	second, cursor2 := f.Drain("", cursor)
	if len(second) != 2 {
		t.Fatalf("re-attach drain returned %d events, want 2 (only the new turns)", len(second))
	}
	for _, ev := range second {
		if ev.Seq <= cursor {
			t.Fatalf("re-attach yielded Seq=%d <= saved cursor %d — a DUPLICATE", ev.Seq, cursor)
		}
	}
	if second[0].Seq != cursor+1 {
		t.Fatalf("re-attach first Seq=%d, want %d — a GAP", second[0].Seq, cursor+1)
	}
	if cursor2 != 5 {
		t.Fatalf("re-attach cursor = %d, want 5", cursor2)
	}

	// Concatenating both drains reconstructs EVERY appended event exactly once, in order,
	// with no gap and no duplicate across the drop boundary.
	all := append(append([]TranscriptEvent{}, first...), second...)
	if len(all) != 5 {
		t.Fatalf("first+second = %d events, want 5 (every appended event exactly once)", len(all))
	}
	wantUUIDs := []string{"t1", "t2", "t3", "t4", "t5"}
	for i, ev := range all {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("stitched event %d Seq=%d, want %d (monotone, gap-free)", i, ev.Seq, i+1)
		}
		if ev.UUID != wantUUIDs[i] {
			t.Fatalf("stitched event %d uuid=%q, want %q", i, ev.UUID, wantUUIDs[i])
		}
	}
}

// --- observing is side-effect-free ------------------------------------------------------

func TestDrainIsSideEffectFree(t *testing.T) {
	f := NewFeed(0)
	for _, ev := range []TranscriptEvent{
		{Kind: KindTurnOpen, UUID: "a"},
		{Kind: KindTurnClose, UUID: "b"},
	} {
		f.Append(ev)
	}

	seqBefore := f.seq
	ringLenBefore := len(f.ring)

	// Two identical reads from the same cursor return identical results...
	d1, c1 := f.Drain("", 0)
	d2, c2 := f.Drain("", 0)
	if c1 != c2 || len(d1) != len(d2) {
		t.Fatalf("repeated Drain differ: (%d,%v) vs (%d,%v)", c1, len(d1), c2, len(d2))
	}
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Fatalf("repeated Drain event %d differs: %+v vs %+v", i, d1[i], d2[i])
		}
	}
	// ...and neither read advanced the feed's own loop (Seq counter and ring untouched).
	if f.seq != seqBefore {
		t.Fatalf("Drain advanced f.seq from %d to %d — observing must not advance the loop", seqBefore, f.seq)
	}
	if len(f.ring) != ringLenBefore {
		t.Fatalf("Drain mutated the ring length from %d to %d", ringLenBefore, len(f.ring))
	}
}

// --- quarantined spans never cross the boundary -----------------------------------------

func TestQuarantinedSpanNeverCrossesBoundary(t *testing.T) {
	const secret = "SUPER-SECRET-PLAN-DO-NOT-LEAK-abc123"

	// One assistant turn whose text is the secret, plus a benign user turn. The secret turn
	// is derived from a sealed/tombstoned span.
	recs := []transcript.Record{
		{Type: "user", UUID: "safe", Message: &transcript.Message{Role: "user", Content: rawString("run the build please")}},
		{Type: "assistant", UUID: "tainted", Message: &transcript.Message{Role: "assistant", Content: rawString(secret)}},
	}
	quarantined := func(r transcript.Record) bool { return r.UUID == "tainted" }

	evs := EventsFromRecordsScreened(recs, "", quarantined)

	// The ingress screen produced a taint STUB for the quarantined record: kind emitted,
	// content withheld behind the read-plane marker, no payload bytes.
	var stub *TranscriptEvent
	for i := range evs {
		if evs[i].UUID == "tainted" {
			stub = &evs[i]
		}
	}
	if stub == nil {
		t.Fatal("no event produced for the quarantined record")
	}
	if !stub.Withheld {
		t.Fatalf("quarantined event not marked Withheld: %+v", *stub)
	}
	if stub.Reason != sessionread.ReasonReadTaintWithheld {
		t.Fatalf("stub Reason=%q, want %q (the read-plane taint marker)", stub.Reason, sessionread.ReasonReadTaintWithheld)
	}
	if stub.Summary != "" || stub.Tool != "" {
		t.Fatalf("stub carries payload bytes Summary=%q Tool=%q, want both empty", stub.Summary, stub.Tool)
	}
	// The kind is still emitted (structural metadata is safe to disclose).
	if stub.Kind != KindTurnClose {
		t.Fatalf("stub Kind=%q, want %q (kind survives the screen)", stub.Kind, KindTurnClose)
	}

	// Push through a real Feed and drain from every reachable cursor: the secret bytes must
	// be ABSENT from every drained event, on every re-attach.
	f := NewFeed(0)
	for _, ev := range evs {
		f.Append(ev)
	}
	for _, since := range []uint64{0, 1, 2} {
		drained, _ := f.Drain("", since)
		for _, ev := range drained {
			for _, field := range []string{ev.Kind, ev.UUID, ev.Tool, ev.Summary, ev.Reason} {
				if strings.Contains(field, secret) {
					t.Fatalf("secret content crossed the stream boundary in %+v (since=%d)", ev, since)
				}
			}
		}
	}
}

// --- principal scoping keeps a peer's cursor monotone (no re-scan) -----------------------

func TestPrincipalScopingCursorStaysMonotone(t *testing.T) {
	f := NewFeed(0)
	// Same feed, three producers: tenant A, tenant B, and a principal-less/global event.
	f.Append(TranscriptEvent{Kind: KindTurnOpen, UUID: "a1", principal: "A"})
	f.Append(TranscriptEvent{Kind: KindTurnOpen, UUID: "b1", principal: "B"})
	f.Append(TranscriptEvent{Kind: KindDecision, UUID: "g1", Tool: "Read", principal: ""})

	// B drains: it sees its OWN event + the global one, never A's — but its cursor still
	// advances PAST A's Seq (over all retained), so it will never re-scan it.
	bEvents, bCursor := f.Drain("B", 0)
	for _, ev := range bEvents {
		if ev.principal == "A" {
			t.Fatalf("principal B saw a peer (A) event: %+v", ev)
		}
	}
	if len(bEvents) != 2 {
		t.Fatalf("B drained %d events, want 2 (own b1 + global g1)", len(bEvents))
	}
	if bCursor != 3 {
		t.Fatalf("B cursor = %d, want 3 (advanced over ALL retained, including A's Seq)", bCursor)
	}

	// Re-attach from B's cursor: nothing new, and it does NOT re-scan A's already-elapsed
	// Seq (the cursor-over-all-retained invariant makes this gap-free AND re-scan-free).
	bEvents2, bCursor2 := f.Drain("B", bCursor)
	if len(bEvents2) != 0 {
		t.Fatalf("B re-attach yielded %d events, want 0 (no re-scan)", len(bEvents2))
	}
	if bCursor2 != bCursor {
		t.Fatalf("B re-attach cursor moved from %d to %d with no new events", bCursor, bCursor2)
	}

	// A, symmetrically, sees its own + global, never B's.
	aEvents, aCursor := f.Drain("A", 0)
	for _, ev := range aEvents {
		if ev.principal == "B" {
			t.Fatalf("principal A saw a peer (B) event: %+v", ev)
		}
	}
	if len(aEvents) != 2 || aCursor != 3 {
		t.Fatalf("A drained %d events cursor %d, want 2 events cursor 3", len(aEvents), aCursor)
	}

	// An empty (observer/admin) drainer sees everything.
	allEvents, _ := f.Drain("", 0)
	if len(allEvents) != 3 {
		t.Fatalf("empty-principal drain saw %d events, want all 3", len(allEvents))
	}
}

// --- transcript records map to the four event kinds -------------------------------------

func TestEventsFromRecordsMapsTranscriptKinds(t *testing.T) {
	// A small JSONL fixture session covering every kind: a user turn (turn-open), an
	// assistant tool choice with no result (decision), a paired tool_use+tool_result
	// (tool-terminal), and an assistant plain turn (turn-close).
	lines := []string{
		`{"type":"user","uuid":"u1","message":{"role":"user","content":"please run the build"}}`,
		`{"type":"assistant","uuid":"d1","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read"}]}}`,
		`{"type":"assistant","uuid":"x1","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash"},{"type":"tool_result","content":"exit 0"}]}}`,
		`{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":[{"type":"text","text":"the build passed"}]}}`,
		`{"type":"summary","uuid":"s1","message":{"role":"summary","content":"a control row"}}`,
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "session.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// Tail-a-fixture entrypoint the observe plane uses on live, appended transcripts.
	recs := transcript.LoadFileTail(path, 0)
	if len(recs) != 5 {
		t.Fatalf("loaded %d records, want 5", len(recs))
	}

	evs := EventsFromRecords(recs, "")

	// The control/summary row maps to no event; the four turns each map to their kind.
	byUUID := map[string]TranscriptEvent{}
	for _, ev := range evs {
		byUUID[ev.UUID] = ev
	}
	if len(evs) != 4 {
		t.Fatalf("produced %d events, want 4 (summary row is skipped): %+v", len(evs), evs)
	}
	wantKind := map[string]string{
		"u1": KindTurnOpen,
		"d1": KindDecision,
		"x1": KindToolTerminal,
		"a1": KindTurnClose,
	}
	for uuid, want := range wantKind {
		got, ok := byUUID[uuid]
		if !ok {
			t.Fatalf("no event for record %q", uuid)
		}
		if got.Kind != want {
			t.Fatalf("record %q -> kind %q, want %q", uuid, got.Kind, want)
		}
	}
	// The tool-terminal and decision events name their tool; the summary carries a bounded,
	// redacted descriptor (never full text is required — just that it round-trips).
	if byUUID["x1"].Tool != "Bash" {
		t.Fatalf("tool-terminal Tool=%q, want Bash", byUUID["x1"].Tool)
	}
	if byUUID["d1"].Tool != "Read" {
		t.Fatalf("decision Tool=%q, want Read", byUUID["d1"].Tool)
	}
	if byUUID["u1"].Summary != "please run the build" {
		t.Fatalf("turn-open Summary=%q, want %q", byUUID["u1"].Summary, "please run the build")
	}

	// End-to-end: the produced events drain from a live Feed in order, gap-free.
	f := NewFeed(0)
	for _, ev := range evs {
		f.Append(ev)
	}
	drained, cursor := f.Drain("", 0)
	if len(drained) != 4 || cursor != 4 {
		t.Fatalf("end-to-end drain = %d events cursor %d, want 4 / 4", len(drained), cursor)
	}
}

// rawString marshals a bare string into JSON content, mirroring the transcript package's
// own test helper so fixtures read identically.
func rawString(s string) json.RawMessage {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(b)
}

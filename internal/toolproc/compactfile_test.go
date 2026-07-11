package toolproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// writeJournal writes events as JSONL to a fresh file and returns its path.
func writeJournal(t *testing.T, events []Event) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	var buf []byte
	for _, ev := range events {
		line, err := json.Marshal(ev)
		if err != nil {
			t.Fatal(err)
		}
		buf = append(buf, append(line, '\n')...)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// readJournal parses a journal file back into events.
func readJournal(t *testing.T, path string) []Event {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evs, err := ParseEvents(f)
	if err != nil {
		t.Fatalf("journal must stay fold-clean after compaction: %v", err)
	}
	return evs
}

// Over the threshold, CompactJournalFile reclaims fully-terminal history, keeps
// the oldest un-exited spawn (even though it sits far outside any tail margin),
// and leaves the file fold-clean and smaller.
func TestCompactJournalFileBoundsOversized(t *testing.T) {
	events := []Event{{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "s", AtMS: 1}}
	for i := 0; i < 500; i++ {
		id := fmt.Sprintf("done-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(2 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
	}
	path := writeJournal(t, events)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	// Threshold of 1 byte forces the rewrite; keep only the last 8 events.
	compacted, err := CompactJournalFile(path, 1, 8)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if !compacted {
		t.Fatal("oversized journal with terminal history must be compacted")
	}

	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() >= before.Size() {
		t.Fatalf("compaction did not shrink the file: before=%d after=%d", before.Size(), after.Size())
	}

	got := readJournal(t, path)
	if len(got) >= len(events) {
		t.Fatalf("terminal history not reclaimed: in=%d out=%d", len(events), len(got))
	}
	// The oldest live spawn survives and still folds to a running proc.
	if id, running := hookResolveID("live", got); !running || id != "live" {
		t.Fatalf("live spawn dropped by file compaction: id=%q running=%v", id, running)
	}
}

// appendRaw appends raw bytes to an existing journal file — damage the helpers
// above could never write.
func appendRaw(t *testing.T, path string, chunks ...[]byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range chunks {
		if _, err := f.Write(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// A journal carrying rows the parser cannot decode — a single record past the
// scanner token cap and a torn JSON write — must still get bounded: the
// compaction read drops the bad rows and truncates forward instead of aborting,
// and the rewritten file is fold-clean (#3556).
func TestCompactJournalFileDropsUndecodableRows(t *testing.T) {
	events := []Event{{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "s", AtMS: 1}}
	for i := 0; i < 200; i++ {
		id := fmt.Sprintf("done-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(2 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
	}
	path := writeJournal(t, events)
	// One record past the token cap, one torn write, then a good row AFTER the
	// damage to prove the reader skips forward rather than stopping at it.
	goodTail, err := json.Marshal(Event{Kind: EvSpawn, CallID: "after-damage", Tool: "Bash", Session: "s", AtMS: 9999})
	if err != nil {
		t.Fatal(err)
	}
	appendRaw(t, path,
		append(bytes.Repeat([]byte("x"), maxEventLineBytes+1), '\n'),
		[]byte(`{"kind":"spawn","call_id":"torn"`+"\n"),
		append(goodTail, '\n'))

	compacted, err := CompactJournalFile(path, 1, 8)
	if err != nil {
		t.Fatalf("compaction must drop undecodable rows, not abort: %v", err)
	}
	if !compacted {
		t.Fatal("journal with undecodable rows was not rewritten")
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() > int64(maxEventLineBytes) {
		t.Fatalf("oversized record survived compaction: size=%d", after.Size())
	}
	got := readJournal(t, path) // strict ParseEvents: the result must be fold-clean
	if id, running := hookResolveID("live", got); !running || id != "live" {
		t.Fatalf("live spawn dropped: id=%q running=%v", id, running)
	}
	if id, running := hookResolveID("after-damage", got); !running || id != "after-damage" {
		t.Fatalf("decodable row after the damaged region lost: id=%q running=%v", id, running)
	}
}

// The "never recovers" shape from #3556: the oversized record alone holds the
// file over the threshold and every real row is a live spawn, so there is no
// terminal history to reclaim. The rewrite must still happen — expelling the
// bad row is the only way the file ever gets bounded.
func TestCompactJournalFileRewritesForDroppedRowsAlone(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "live-a", Tool: "Bash", Session: "s", AtMS: 1},
		{Kind: EvSpawn, CallID: "live-b", Tool: "Bash", Session: "s", AtMS: 2},
	}
	path := writeJournal(t, events)
	appendRaw(t, path, append(bytes.Repeat([]byte("x"), maxEventLineBytes+1), '\n'))

	compacted, err := CompactJournalFile(path, 1, 8)
	if err != nil {
		t.Fatalf("compaction must drop the oversized row, not abort: %v", err)
	}
	if !compacted {
		t.Fatal("journal held over the threshold by an undecodable row alone must still be rewritten")
	}
	got := readJournal(t, path)
	if len(got) != len(events) {
		t.Fatalf("live spawns not preserved: want %d events, got %d", len(events), len(got))
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() > int64(maxEventLineBytes) {
		t.Fatalf("oversized record survived compaction: size=%d", after.Size())
	}
}

// Under the threshold, CompactJournalFile is a stat-only no-op: the file's bytes
// are byte-for-byte unchanged.
func TestCompactJournalFileNoopUnderThreshold(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "s", AtMS: 1},
		{Kind: EvExit, CallID: "a", AtMS: 2, Status: "ok"},
	}
	path := writeJournal(t, events)
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	compacted, err := CompactJournalFile(path, JournalCompactThresholdBytes, JournalCompactTailKeep)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if compacted {
		t.Fatal("a journal under the threshold must not be rewritten")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("under-threshold journal must be byte-for-byte unchanged")
	}
}

// A missing journal is a clean no-op — a caller can invoke compaction
// unconditionally at a session boundary before any hook ever wrote a row.
func TestCompactJournalFileMissing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")
	compacted, err := CompactJournalFile(path, 1, 8)
	if err != nil {
		t.Fatalf("missing journal must be a clean no-op, got err=%v", err)
	}
	if compacted {
		t.Fatal("missing journal cannot be compacted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("compaction must not create a journal that did not exist")
	}
}

// Over the threshold but with nothing terminal to reclaim (every spawn is still
// live), CompactJournalFile leaves the file untouched rather than churn an
// identical rewrite — no live spawn is ever dropped to shrink the file.
func TestCompactJournalFileNoopWhenAllLive(t *testing.T) {
	var events []Event
	for i := 0; i < 50; i++ {
		events = append(events, Event{Kind: EvSpawn, CallID: fmt.Sprintf("live-%d", i), Tool: "Bash", Session: "s", AtMS: int64(i + 1)})
	}
	path := writeJournal(t, events)
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	compacted, err := CompactJournalFile(path, 1, 4)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if compacted {
		t.Fatal("an all-live journal has nothing terminal to reclaim; it must not be rewritten")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("all-live journal must be left byte-for-byte unchanged")
	}
}

// The bound CompactJournalFile enforces is a CONSTANT ceiling — the last
// tailKeep events plus every un-exited spawn — not merely "smaller than before".
// This is the disk-growth guarantee #3488 wired into production: the once-per-
// session stop firing in cmd/fak/toolproc.go compacts with JournalCompactTailKeep,
// so an ever-appended journal collapses to a fixed size rather than a
// proportionally-smaller-but-still-growing one. TestCompactJournalFileBoundsOversized
// proves the file shrinks for ONE input; this proves the ceiling is independent
// of how much terminal history was reclaimed — compact two journals whose
// terminal history differs 8x and the compacted results carry the same bounded
// event count and a same-order on-disk size. A regression that kept history
// proportional to the session's length would still pass "after < before" but
// fail this.
func TestCompactJournalFileBoundIsConstantCeiling(t *testing.T) {
	const tailKeep = 64
	build := func(pairs int) []Event {
		// One un-exited spawn at the front (kept regardless of age) followed by
		// `pairs` fully-terminal spawn/exit pairs — all reclaimable but for the tail.
		evs := []Event{{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "s", AtMS: 1}}
		for i := 0; i < pairs; i++ {
			id := fmt.Sprintf("done-%d", i)
			evs = append(evs,
				Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(2 + 2*i)},
				Event{Kind: EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
		}
		return evs
	}
	compact := func(pairs int) (events int, before, after int64) {
		path := writeJournal(t, build(pairs))
		bi, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		// Threshold 1 forces the rewrite; the ceiling is tailKeep events + the live spawn.
		if _, err := CompactJournalFile(path, 1, tailKeep); err != nil {
			t.Fatalf("compact(%d pairs): %v", pairs, err)
		}
		ai, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		got := readJournal(t, path) // strict ParseEvents: must stay fold-clean
		if id, running := hookResolveID("live", got); !running || id != "live" {
			t.Fatalf("live spawn dropped at %d pairs: id=%q running=%v", pairs, id, running)
		}
		return len(got), bi.Size(), ai.Size()
	}

	smallN, _, smallBytes := compact(500)
	largeN, largeBefore, largeBytes := compact(4000) // 8x the terminal history

	// The event ceiling is identical across an 8x difference in reclaimed history:
	// the last tailKeep events plus the one live spawn, nothing proportional to N.
	if smallN != largeN {
		t.Fatalf("compacted event count grew with history: %d (500 pairs) vs %d (4000 pairs) — bound is not a constant ceiling", smallN, largeN)
	}
	if want := tailKeep + 1; largeN != want {
		t.Fatalf("compacted to %d events, want tailKeep+live=%d", largeN, want)
	}
	// And it holds on disk: the 8x-larger input compacts to a file no bigger than
	// a small factor of the small one (bytes differ only by ID-string length), and
	// to a small fraction of its own pre-compaction size — the append-only write
	// path is genuinely bounded, not just trimmed.
	if largeBytes >= 2*smallBytes {
		t.Fatalf("compacted disk size scaled with history: %d bytes (4000 pairs) vs %d bytes (500 pairs)", largeBytes, smallBytes)
	}
	if largeBytes >= largeBefore/4 {
		t.Fatalf("compacted file not bounded on disk: %d bytes from a %d-byte journal", largeBytes, largeBefore)
	}
}

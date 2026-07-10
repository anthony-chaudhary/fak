package toolproc

import (
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

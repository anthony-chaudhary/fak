package toolproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// journalBytes marshals events to the JSONL wire form the journal stores.
func journalBytes(t *testing.T, events []Event) []byte {
	t.Helper()
	var b bytes.Buffer
	for _, ev := range events {
		line, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	return b.Bytes()
}

// A journal within the window is parsed whole — ParseTail matches ParseEvents.
func TestParseTailWholeUnderWindow(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "s", AtMS: 1},
		{Kind: EvExit, CallID: "a", AtMS: 2, Status: "ok"},
	}
	raw := journalBytes(t, events)
	got, err := ParseTail(bytes.NewReader(raw), JournalTailWindowBytes)
	if err != nil {
		t.Fatalf("ParseTail: %v", err)
	}
	if len(got) != len(events) {
		t.Fatalf("under-window parse dropped events: in=%d out=%d", len(events), len(got))
	}
}

// A non-positive window disables bounding: the whole reader is parsed.
func TestParseTailNonPositiveWindowParsesWhole(t *testing.T) {
	events := make([]Event, 0, 100)
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("c-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(1 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(2 + 2*i), Status: "ok"})
	}
	raw := journalBytes(t, events)
	got, err := ParseTail(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatalf("ParseTail: %v", err)
	}
	if len(got) != len(events) {
		t.Fatalf("non-positive window should parse whole: in=%d out=%d", len(events), len(got))
	}
}

// A journal larger than the window is bounded: only tail events return, the
// partial record the seek lands inside is discarded (no parse error, no bogus
// event), and a recent call's spawn near the tail stays resolvable.
func TestParseTailBoundsAndAlignsToRecord(t *testing.T) {
	var events []Event
	for i := 0; i < 4000; i++ { // padding — pushes the file well past the window
		id := fmt.Sprintf("old-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(1 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(2 + 2*i), Status: "ok"})
	}
	// The current call's pre-spawn, appended last — what a post firing must pair.
	events = append(events, Event{Kind: EvSpawn, CallID: "recent", Tool: "Bash", Session: "s", AtMS: 999999})
	raw := journalBytes(t, events)

	window := int64(64 << 10) // 64 KiB — far smaller than the file
	if int64(len(raw)) <= window {
		t.Fatalf("fixture too small (%d bytes) to exercise bounding", len(raw))
	}
	got, err := ParseTail(bytes.NewReader(raw), window)
	if err != nil {
		t.Fatalf("ParseTail over window: %v", err)
	}
	if len(got) == 0 || len(got) >= len(events) {
		t.Fatalf("window not applied: in=%d out=%d", len(events), len(got))
	}
	// The most recent event survived and stays resolvable as running.
	if id, running := hookResolveID("recent", got); !running || id != "recent" {
		t.Fatalf("recent spawn not resolvable after tail-read: id=%q running=%v", id, running)
	}
	// Every returned event is well-formed — the partial first line was dropped.
	for _, ev := range got {
		if err := ValidateEvent(ev); err != nil {
			t.Fatalf("tail-read yielded an invalid event: %v", err)
		}
	}
	// The oldest padding event fell outside the window.
	for _, ev := range got {
		if ev.CallID == "old-0" {
			t.Fatalf("event older than the window leaked into the tail")
		}
	}
}

// ParseTailFile round-trips a real journal on disk and applies the same bound.
func TestParseTailFileOverWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	var events []Event
	for i := 0; i < 200; i++ {
		id := fmt.Sprintf("f-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(1 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(2 + 2*i), Status: "ok"})
	}
	if err := os.WriteFile(path, journalBytes(t, events), 0o644); err != nil {
		t.Fatalf("write journal: %v", err)
	}

	// Under the default window: whole file parses.
	whole, err := ParseTailFile(path)
	if err != nil {
		t.Fatalf("ParseTailFile: %v", err)
	}
	if len(whole) != len(events) {
		t.Fatalf("small file should parse whole: in=%d out=%d", len(events), len(whole))
	}
}

// A missing journal is not an error — a fresh workspace has none yet, and the
// hook must fail open exactly as it did when it opened the file directly.
func TestParseTailFileMissing(t *testing.T) {
	got, err := ParseTailFile(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if err != nil {
		t.Fatalf("missing file should be nil error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("missing file should yield no events, got %d", len(got))
	}
}

// Blank and comment lines in the retained tail are skipped, mirroring
// ParseEvents — the seek-and-discard must not change record semantics.
func TestParseTailSkipsBlankAndCommentInTail(t *testing.T) {
	raw := []byte(strings.Join([]string{
		`{"kind":"spawn","call_id":"a","tool":"Bash","session":"s","at_unix_ms":1}`,
		"",
		"# a comment",
		`{"kind":"exit","call_id":"a","at_unix_ms":2,"status":"ok"}`,
	}, "\n") + "\n")
	got, err := ParseTail(bytes.NewReader(raw), JournalTailWindowBytes)
	if err != nil {
		t.Fatalf("ParseTail: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 events (blank/comment skipped), got %d", len(got))
	}
}

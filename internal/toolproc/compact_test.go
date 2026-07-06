package toolproc

import (
	"fmt"
	"testing"
)

// A live (un-exited) spawn must survive compaction even when it is the oldest
// event and the tail window would otherwise exclude it — the #3032 invariant.
func TestCompactJournalKeepsOldestLiveSpawn(t *testing.T) {
	events := []Event{{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "s", AtMS: 1}}
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("done-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(2 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
	}
	got := CompactJournal(events, 5)

	foundLive := false
	for _, ev := range got {
		if ev.CallID == "live" && ev.Kind == EvSpawn {
			foundLive = true
		}
	}
	if !foundLive {
		t.Fatalf("live spawn dropped by compaction; got %d events", len(got))
	}
	if len(got) >= len(events) {
		t.Fatalf("compaction did not reduce terminal history: in=%d out=%d", len(events), len(got))
	}
	if id, running := hookResolveID("live", got); !running || id != "live" {
		t.Fatalf("live identity not running after compaction: id=%q running=%v", id, running)
	}
}

// tailKeep >= len(events) returns the input unchanged.
func TestCompactJournalTailKeepAll(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "s", AtMS: 1},
		{Kind: EvExit, CallID: "a", AtMS: 2, Status: "ok"},
	}
	if got := CompactJournal(events, len(events)); len(got) != len(events) {
		t.Fatalf("tailKeep>=len should be unchanged: in=%d out=%d", len(events), len(got))
	}
}

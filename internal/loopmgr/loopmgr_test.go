package loopmgr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAppendLoadValidatesHashChainAndSummary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	now := fixedClock()
	events := []Event{
		{LoopID: "issue-dispatch/default", Kind: EventFire, Source: "schedule", Principal: "timer"},
		{LoopID: "issue-dispatch/default", Kind: EventAdmit, RunID: "run-1", Status: StatusAdmitted},
		{LoopID: "issue-dispatch/default", Kind: EventStart, RunID: "run-1"},
		{LoopID: "issue-dispatch/default", Kind: EventEnd, RunID: "run-1", Status: StatusClaimedDone},
		{LoopID: "issue-dispatch/default", Kind: EventWitness, RunID: "run-1", Status: StatusWitnessedDone, EvidenceRefs: []EvidenceRef{{Kind: "commit", Ref: "8469c56"}}},
		{LoopID: "issue-dispatch/default", Kind: EventNotify, RunID: "run-1", Reason: "DONE_WITNESSED"},
	}
	for _, ev := range events {
		if _, err := Append(path, ev, WithClock(now)); err != nil {
			t.Fatalf("Append(%s): %v", ev.Kind, err)
		}
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != len(events) {
		t.Fatalf("loaded %d events, want %d", len(loaded), len(events))
	}
	for i, ev := range loaded {
		if ev.Seq != uint64(i+1) {
			t.Fatalf("event %d seq = %d", i, ev.Seq)
		}
		if ev.Hash == "" {
			t.Fatalf("event %d missing hash", i)
		}
		if i > 0 && ev.PrevHash != loaded[i-1].Hash {
			t.Fatalf("event %d prev hash = %q, want prior %q", i, ev.PrevHash, loaded[i-1].Hash)
		}
	}

	st := Summarize(loaded, time.Unix(0, 7000).UTC())
	if st.Schema != SchemaStatus {
		t.Fatalf("status schema = %q, want %q", st.Schema, SchemaStatus)
	}
	if len(st.Loops) != 1 {
		t.Fatalf("loops = %d, want 1", len(st.Loops))
	}
	loop := st.Loops[0]
	if loop.LoopID != "issue-dispatch/default" || loop.Fires != 1 || loop.Admitted != 1 || loop.Started != 1 || loop.Ended != 1 || loop.Witnessed != 1 || loop.Notifications != 1 {
		t.Fatalf("summary = %+v", loop)
	}
	if loop.LastRun == nil || loop.LastRun.Status != StatusWitnessedDone || len(loop.LastRun.EvidenceRefs) != 1 {
		t.Fatalf("last run = %+v", loop.LastRun)
	}
}

func TestLoadRejectsTamperedLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	if _, err := Append(path, Event{LoopID: "loop-a", Kind: EventFire, Source: "schedule"}, WithClock(fixedClock())); err != nil {
		t.Fatalf("Append: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := strings.Replace(string(b), `"schedule"`, `"slack"`, 1)
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("Load tampered err = %v, want hash error", err)
	}
}

func TestLoadMissingLedgerIsEmpty(t *testing.T) {
	events, err := Load(filepath.Join(t.TempDir(), "missing.jsonl"))
	if err != nil {
		t.Fatalf("Load missing: %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("missing ledger events = %d, want 0", len(events))
	}
}

func TestSummarizeSortsLoops(t *testing.T) {
	st := Summarize([]Event{
		{Schema: SchemaEvent, Seq: 1, LoopID: "z-loop", Kind: EventFire},
		{Schema: SchemaEvent, Seq: 2, LoopID: "a-loop", Kind: EventFire},
	}, time.Unix(0, 1))
	if len(st.Loops) != 2 || st.Loops[0].LoopID != "a-loop" || st.Loops[1].LoopID != "z-loop" {
		t.Fatalf("loops sorted = %+v", st.Loops)
	}
}

func fixedClock() func() time.Time {
	return func() time.Time { return time.Unix(0, 1000).UTC() }
}

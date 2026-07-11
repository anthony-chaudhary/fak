package loopmgr

import (
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// richFoldStream is a synthetic multi-loop event slice that exercises every branch of the
// Summarize fold: the additive counters, a ConsecutiveRefusals streak that both grows and
// resets, last-writer-wins scalars (State/LastSeq/LastKind/CurrentRunID), LastRun's
// prior-run RunID inheritance (an EventEnd with an empty RunID inheriting the running
// run's id), Metrics per-key overwrite, and a run left in-flight (Start with no End) so
// Concurrent()=Started-Ended is non-zero. Seq/TSUnixNano advance monotonically as a real
// ledger's would; the fold reads them but validates nothing, so hand-built events are fine.
func richFoldStream() []Event {
	evs := []Event{
		{LoopID: "alpha", Kind: EventArmed, State: StateArmed},
		{LoopID: "beta", Kind: EventFire},
		{LoopID: "alpha", Kind: EventFire},
		{LoopID: "alpha", Kind: EventAdmit, Status: StatusRefused, Reason: "CADENCE_FLOOR"},
		{LoopID: "alpha", Kind: EventAdmit, Status: StatusRefused, Reason: "CADENCE_FLOOR"},
		{LoopID: "beta", Kind: EventStart, RunID: "x1"},
		{LoopID: "alpha", Kind: EventAdmit, Status: StatusAdmitted},
		{LoopID: "alpha", Kind: EventStart, RunID: "a1", Metrics: map[string]int64{"argc": 1}},
		{LoopID: "beta", Kind: EventEnd, RunID: "x1", Status: StatusFailed},
		// EventEnd with empty RunID must inherit a1 from the running run (setRun fallback).
		{LoopID: "alpha", Kind: EventEnd, Status: StatusClaimedDone, Metrics: map[string]int64{"argc": 2}},
		{LoopID: "alpha", Kind: EventWitness, Status: StatusWitnessedDone, Summary: "ok"},
		{LoopID: "beta", Kind: EventWitness, Status: StatusWitnessRefused},
		{LoopID: "alpha", Kind: EventNotify},
		{LoopID: "alpha", Kind: EventFire},
		{LoopID: "alpha", Kind: EventAdmit, Status: StatusRefused, Reason: "REFUSAL_STORM"},
		// A run left in-flight: Started increments with no matching End -> Concurrent()==1.
		{LoopID: "alpha", Kind: EventStart, RunID: "a2", Metrics: map[string]int64{"argc": 3, "guard": 1}},
	}
	for i := range evs {
		evs[i].Schema = SchemaEvent
		evs[i].Seq = uint64(i + 1)
		evs[i].TSUnixNano = int64(1700000000+i) * int64(time.Second)
		if evs[i].Source == "" {
			evs[i].Source = "test"
		}
	}
	return evs
}

// TestSummarizeFromResumesEqualsFromScratch is the fold-equivalence proof the rotation
// carried-snapshot design rests on: for EVERY split point k, seeding SummarizeFrom with the
// snapshot of the prefix and folding the suffix reproduces the from-scratch fold of the
// whole stream, field-for-field. It also asserts SummarizeFrom does not mutate or alias the
// seed the caller passed.
func TestSummarizeFromResumesEqualsFromScratch(t *testing.T) {
	now := time.Unix(1700009999, 0)
	stream := richFoldStream()
	want := Summarize(stream, now)

	for k := 0; k <= len(stream); k++ {
		prefix := stream[:k]
		suffix := stream[k:]

		seed := Summarize(prefix, now).Loops
		// A pristine second fold of the prefix: if SummarizeFrom mutates the seed it is
		// handed, seed will diverge from this after the call.
		pristine := Summarize(prefix, now).Loops

		got := SummarizeFrom(seed, suffix, now)

		if !reflect.DeepEqual(got.Loops, want.Loops) {
			t.Fatalf("split k=%d: SummarizeFrom(seed, suffix) diverged from Summarize(all)\n got=%+v\nwant=%+v",
				k, got.Loops, want.Loops)
		}
		if !reflect.DeepEqual(seed, pristine) {
			t.Fatalf("split k=%d: SummarizeFrom mutated the seed it was passed\n after=%+v\nbefore=%+v",
				k, seed, pristine)
		}
	}
}

// TestSummarizeFromConcurrentAcrossSeam pins the specific hazard rotation introduces: a run
// whose EventStart lands in the prefix (a would-be sealed segment) and whose EventEnd lands
// in the suffix (the active segment). The from-empty active-only fold would undercount
// Started and floor Concurrent() to 0; the seeded fold carries the baseline and stays exact.
func TestSummarizeFromConcurrentAcrossSeam(t *testing.T) {
	now := time.Unix(1700009999, 0)
	prefix := []Event{{LoopID: "l", Kind: EventStart, RunID: "A", Seq: 1}}
	suffix := []Event{
		{LoopID: "l", Kind: EventEnd, RunID: "A", Status: StatusClaimedDone, Seq: 1},
		{LoopID: "l", Kind: EventStart, RunID: "B", Seq: 2},
	}

	seeded := SummarizeFrom(Summarize(prefix, now).Loops, suffix, now)
	activeOnly := Summarize(suffix, now)

	sl := loopByID(t, seeded.Loops, "l")
	if sl.Started != 2 || sl.Ended != 1 || sl.Concurrent() != 1 {
		t.Fatalf("seeded fold: Started=%d Ended=%d Concurrent=%d, want 2/1/1", sl.Started, sl.Ended, sl.Concurrent())
	}
	al := loopByID(t, activeOnly.Loops, "l")
	if al.Concurrent() != 0 {
		t.Fatalf("active-only fold Concurrent=%d, want 0 (the seam underflow the seeded fold prevents)", al.Concurrent())
	}
}

// TestSnapshotFileAllSurvivesRotationSeam is the end-to-end form over a real Append+Rotate
// ledger: A starts (sealed by Rotate), then A ends and B starts in the fresh active segment.
// SnapshotFileAll folds the full history across the seam and reports the true in-flight count;
// SnapshotFile (active-only) underflows Concurrent() to 0 — the exact rotation hazard.
func TestSnapshotFileAllSurvivesRotationSeam(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	clk := func() time.Time { return time.Unix(1700000000, 0) }

	if _, err := Append(path, Event{LoopID: "l", Kind: EventStart, RunID: "A", Source: "s"}, WithClock(clk)); err != nil {
		t.Fatalf("append start A: %v", err)
	}
	if res, err := Rotate(path, 0); err != nil || !res.Rotated {
		t.Fatalf("Rotate: res=%+v err=%v", res, err)
	}
	if _, err := Append(path, Event{LoopID: "l", Kind: EventEnd, RunID: "A", Status: StatusClaimedDone, Source: "s"}, WithClock(clk)); err != nil {
		t.Fatalf("append end A: %v", err)
	}
	if _, err := Append(path, Event{LoopID: "l", Kind: EventStart, RunID: "B", Source: "s"}, WithClock(clk)); err != nil {
		t.Fatalf("append start B: %v", err)
	}

	all, err := SnapshotFileAll(path, clk())
	if err != nil {
		t.Fatalf("SnapshotFileAll: %v", err)
	}
	al := loopByID(t, all.Loops, "l")
	if al.Started != 2 || al.Ended != 1 || al.Concurrent() != 1 {
		t.Fatalf("SnapshotFileAll across seam: Started=%d Ended=%d Concurrent=%d, want 2/1/1", al.Started, al.Ended, al.Concurrent())
	}

	active, err := SnapshotFile(path, clk())
	if err != nil {
		t.Fatalf("SnapshotFile: %v", err)
	}
	act := loopByID(t, active.Loops, "l")
	if act.Concurrent() != 0 {
		t.Fatalf("active-only SnapshotFile Concurrent=%d, want 0 (undercount the cumulative reader fixes)", act.Concurrent())
	}
}

// TestSnapshotFileAllEqualsSnapshotFileWhenUnrotated mirrors TestLoadAllEqualsLoadWhenUnrotated
// at the fold level: with no sealed segments LoadAll==Load, so SnapshotFileAll and SnapshotFile
// must agree field-for-field (including LastSeq, which only diverges once a real rotation resets
// per-segment seq).
func TestSnapshotFileAllEqualsSnapshotFileWhenUnrotated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	clk := func() time.Time { return time.Unix(1700000000, 0) }
	for _, ev := range []Event{
		{LoopID: "l", Kind: EventStart, RunID: "A", Source: "s"},
		{LoopID: "l", Kind: EventEnd, RunID: "A", Status: StatusClaimedDone, Source: "s"},
		{LoopID: "l", Kind: EventStart, RunID: "B", Source: "s"},
	} {
		if _, err := Append(path, ev, WithClock(clk)); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	all, err := SnapshotFileAll(path, clk())
	if err != nil {
		t.Fatalf("SnapshotFileAll: %v", err)
	}
	one, err := SnapshotFile(path, clk())
	if err != nil {
		t.Fatalf("SnapshotFile: %v", err)
	}
	if !reflect.DeepEqual(all.Loops, one.Loops) {
		t.Fatalf("unrotated SnapshotFileAll != SnapshotFile\n all=%+v\none=%+v", all.Loops, one.Loops)
	}
}

func loopByID(t *testing.T, loops []LoopSnapshot, id string) LoopSnapshot {
	t.Helper()
	for _, l := range loops {
		if l.LoopID == id {
			return l
		}
	}
	t.Fatalf("loop %q not found in %+v", id, loops)
	return LoopSnapshot{}
}

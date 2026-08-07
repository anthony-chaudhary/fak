package main

// #5864 beat-while-working at the witness-sweep call site. The pure rungs are
// covered in internal/lanebeat; what these pin is the WIRING property the fix
// turns on and that no pure test can reach: the beat is emitted from the same
// per-worker branch that just read the process table, so it is impossible for a
// beat to be written for a worker the sweep did not observe running.

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/lanebeat"
)

// beatCall is one write the sweep asked the kernel for.
type beatCall struct {
	lane, owner, loopTS string
}

// withLaneBeatStubs replaces both `dos` seams for the duration of a test and
// returns pointers to the recorded reads and writes. Nothing here spawns a
// child, so the test pins the call site without a live kernel.
func withLaneBeatStubs(t *testing.T, live []lanebeat.Lease, readOK bool) (*int, *[]beatCall) {
	t.Helper()
	reads := 0
	var writes []beatCall
	origRead, origWrite := dispatchLaneBeatReader, dispatchLaneBeatWriter
	dispatchLaneBeatReader = func(string) ([]lanebeat.Lease, bool) {
		reads++
		return live, readOK
	}
	dispatchLaneBeatWriter = func(_ string, dec lanebeat.Decision) bool {
		writes = append(writes, beatCall{dec.Lane, dec.Owner, dec.LoopTS})
		return true
	}
	t.Cleanup(func() { dispatchLaneBeatReader, dispatchLaneBeatWriter = origRead, origWrite })
	return &reads, &writes
}

// laneBeatWorkerSlot lays down a worker slot whose pid is THIS process — the only
// pid a test can guarantee dispatchPIDAlive() reports alive — on the given lane,
// with a stem stamped `stamp` so the spawn time is recoverable.
func laneBeatWorkerSlot(t *testing.T, runsDir, stem, lane string, pid int) {
	t.Helper()
	writeWitnessWorker(t, runsDir, stem, "# fak-spawn issue=5864 lane="+lane+"\nworking\n", pid)
}

// laneBeatLease is a live lease record on lane, held by owner, acquired at
// `acquired` — the shape `dos lease-lane live` emits.
func laneBeatLease(lane, owner, loopTS string, acquired time.Time) lanebeat.Lease {
	host, _ := os.Hostname()
	if v := os.Getenv("DISPATCH_HOST_ID"); v != "" {
		host = v
	}
	return lanebeat.Lease{Lane: lane, Holder: owner, HostID: host, LoopTS: loopTS, AcquiredAt: acquired}
}

// THE HEADLINE. A worker the sweep re-observes running gets its lane lease
// refreshed, carrying the identity off the matched record.
func TestWitnessSweepBeatsTheLaneOfAStillRunningWorker(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	// Stamp the stem a few minutes back so the worker is inside both its quiet
	// window and its budget, and the lease can be acquired AFTER it started.
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	stem := "resolve-5864-" + spawned.Format("20060102-150405")
	laneBeatWorkerSlot(t, runsDir, stem, "gateway", os.Getpid())

	lease := laneBeatLease("gateway", "claude-5864", "2026-08-07T11:41:00Z", spawned.Add(time.Minute))
	_, writes := withLaneBeatStubs(t, []lanebeat.Lease{lease}, true)

	payload, records := witnessExitedWorkers(root, runsDir, true)
	if len(records) != 0 {
		t.Fatalf("a still-running worker must not be graded, got %+v", records)
	}
	if len(*writes) != 1 {
		t.Fatalf("beats written = %+v, want exactly one", *writes)
	}
	got := (*writes)[0]
	if got.lane != "gateway" || got.owner != "claude-5864" || got.loopTS != "2026-08-07T11:41:00Z" {
		t.Errorf("beat identity = %+v, want it copied off the matched lease record", got)
	}
	beat, _ := payload["lane_beat"].(map[string]any)
	if beat == nil || beat["beat"] != 1 {
		t.Errorf("payload lane_beat = %v, want one recorded beat", payload["lane_beat"])
	}
}

// A DEAD worker takes the release branch, never the beat branch. This is the
// wiring form of internal/lanebeat's HOLDER_DEAD rung: the sweep must not even
// reach the writer for a worker it has proven is gone.
func TestWitnessSweepNeverBeatsAFinishedWorker(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	stem := "resolve-5864-" + spawned.Format("20060102-150405")
	laneBeatWorkerSlot(t, runsDir, stem, "gateway", deadDispatchPID)

	lease := laneBeatLease("gateway", "claude-5864", "2026-08-07T11:41:00Z", spawned.Add(time.Minute))
	reads, writes := withLaneBeatStubs(t, []lanebeat.Lease{lease}, true)

	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
	payload, _ := witnessExitedWorkers(root, runsDir, true)
	if len(*writes) != 0 {
		t.Fatalf("a dead worker's lane must never be beaten, got %+v", *writes)
	}
	// And the sweep must not even ask the kernel for the live set on its behalf.
	if *reads != 0 {
		t.Errorf("live-lease reads = %d, want 0 for a sweep with no running worker", *reads)
	}
	if _, present := payload["lane_beat"]; present {
		t.Errorf("lane_beat must be absent when nothing was attested, got %v", payload["lane_beat"])
	}
}

// A dry-run sweep audits without mutating. The beat is a WAL write, so it obeys
// the same `live` gate the #4324 release and the .witness sidecar already do.
func TestWitnessSweepDryRunWritesNoBeat(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	stem := "resolve-5864-" + spawned.Format("20060102-150405")
	laneBeatWorkerSlot(t, runsDir, stem, "gateway", os.Getpid())

	lease := laneBeatLease("gateway", "claude-5864", "2026-08-07T11:41:00Z", spawned.Add(time.Minute))
	reads, writes := withLaneBeatStubs(t, []lanebeat.Lease{lease}, true)

	if _, _ = witnessExitedWorkers(root, runsDir, false); len(*writes) != 0 || *reads != 0 {
		t.Fatalf("a dry-run sweep must write no beat and read no live set: writes=%+v reads=%d", *writes, *reads)
	}
}

// The false-revival guard, end to end: a lease that was taken BEFORE this worker
// started cannot be this worker's, so a live process on the lane does not
// license reviving whatever orphan happens to sit there.
func TestWitnessSweepDoesNotReviveAnOrphanOnTheSameLane(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	stem := "resolve-5864-" + spawned.Format("20060102-150405")
	laneBeatWorkerSlot(t, runsDir, stem, "gateway", os.Getpid())

	orphan := laneBeatLease("gateway", "some-crashed-peer", "2026-07-16T12:26:31Z", spawned.Add(-72*time.Hour))
	_, writes := withLaneBeatStubs(t, []lanebeat.Lease{orphan}, true)

	payload, _ := witnessExitedWorkers(root, runsDir, true)
	if len(*writes) != 0 {
		t.Fatalf("a stranger's older orphan must not be revived, got %+v", *writes)
	}
	beat, _ := payload["lane_beat"].(map[string]any)
	counts, _ := beat["outcomes"].(map[string]any)
	if counts[lanebeat.ReasonLeasePredatesHolder] != 1 {
		t.Errorf("outcomes = %v, want one %s", counts, lanebeat.ReasonLeasePredatesHolder)
	}
}

// One live-lease read per sweep, not one per worker: the beat must not turn a
// busy runs dir into N `dos` children.
func TestWitnessSweepReadsTheLiveLeaseSetOncePerSweep(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	var live []lanebeat.Lease
	for i, lane := range []string{"gateway", "docs", "cmd"} {
		stem := fmt.Sprintf("resolve-586%d-%s", i, spawned.Format("20060102-150405"))
		laneBeatWorkerSlot(t, runsDir, stem, lane, os.Getpid())
		live = append(live, laneBeatLease(lane, "claude-"+lane, "2026-08-07T11:41:00Z", spawned.Add(time.Minute)))
	}
	reads, writes := withLaneBeatStubs(t, live, true)

	if _, _ = witnessExitedWorkers(root, runsDir, true); *reads != 1 {
		t.Errorf("live-lease reads = %d across 3 live workers, want exactly 1", *reads)
	}
	if len(*writes) != 3 {
		t.Errorf("beats written = %d, want one per live worker", len(*writes))
	}
}

// An unreadable kernel degrades to exactly the pre-#5864 behaviour: no beat, no
// error, the lease keeps its TTL — and the sweep still grades normally.
func TestWitnessSweepFailsOpenWhenTheLiveSetIsUnreadable(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	spawned := time.Now().UTC().Add(-5 * time.Minute)
	stem := "resolve-5864-" + spawned.Format("20060102-150405")
	laneBeatWorkerSlot(t, runsDir, stem, "gateway", os.Getpid())

	reads, writes := withLaneBeatStubs(t, nil, false)
	payload, _ := witnessExitedWorkers(root, runsDir, true)
	if len(*writes) != 0 {
		t.Fatalf("an unreadable live set must produce no beat, got %+v", *writes)
	}
	if *reads != 1 {
		t.Errorf("a failed read must be cached, not retried per worker: reads = %d", *reads)
	}
	beat, _ := payload["lane_beat"].(map[string]any)
	counts, _ := beat["outcomes"].(map[string]any)
	if counts["LIVE_LEASES_UNREADABLE"] != 1 {
		t.Errorf("outcomes = %v, want the read fault named", counts)
	}
}

func TestDispatchLaneBeatSpawnedAtParsesTheRunsStem(t *testing.T) {
	got := dispatchLaneBeatSpawnedAt(filepath.Join("x", "y", "resolve-5864-20260807-114100"))
	want := time.Date(2026, 8, 7, 11, 41, 0, 0, time.UTC)
	if !got.Equal(want) {
		t.Errorf("spawnedAt = %v, want %v", got, want)
	}
	// A stem the spawn did not mint yields the zero time, which lanebeat treats as
	// "no spawn evidence" — it disables the deadline rung rather than inventing one.
	if got := dispatchLaneBeatSpawnedAt("resolve-weird"); !got.IsZero() {
		t.Errorf("an unparseable stem must yield the zero time, got %v", got)
	}
}

func TestDispatchLaneBeatParseStampHandlesKernelShapes(t *testing.T) {
	want := time.Date(2026, 8, 7, 11, 41, 0, 0, time.UTC)
	for _, s := range []string{"2026-08-07T11:41:00Z", "2026-08-07T11:41:00", "2026-08-07T11:41Z"} {
		if got := dispatchLaneBeatParseStamp(s); !got.Equal(want) {
			t.Errorf("parse(%q) = %v, want %v", s, got, want)
		}
	}
	// Unparseable is UNPROVABLE, not ancient: the zero time makes lanebeat refuse.
	for _, s := range []string{"", "   ", "not-a-stamp"} {
		if got := dispatchLaneBeatParseStamp(s); !got.IsZero() {
			t.Errorf("parse(%q) = %v, want the zero time", s, got)
		}
	}
}

// The writer refuses to run a `dos` child for a decision that did not admit —
// the last line of defence if a caller ever ignores Decision.Beat.
func TestDispatchLaneBeatWriterRefusesANonAdmittingDecision(t *testing.T) {
	for _, dec := range []lanebeat.Decision{
		{Beat: false, Lane: "gateway", Owner: "claude-5864"},
		{Beat: true, Lane: "", Owner: "claude-5864"},
		{Beat: true, Lane: "gateway", Owner: ""},
	} {
		if dispatchLaneBeatWriteDos(t.TempDir(), dec) {
			t.Errorf("writer must refuse %+v without spawning dos", dec)
		}
	}
}

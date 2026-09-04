package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbudget"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestRegisterGardenTickLoopArmsDurableUnit proves the tick registers as a durable,
// armed loop unit (the #1281 precedent), so it is visible to `fak loop health` and
// re-arms at boot. Re-registering is idempotent (keeps the original CreatedUnixNano).
func TestRegisterGardenTickLoopArmsDurableUnit(t *testing.T) {
	registry := filepath.Join(t.TempDir(), "loop-registry.json")
	if err := registerGardenTickLoop(registry); err != nil {
		t.Fatalf("registerGardenTickLoop: %v", err)
	}
	reg, err := loopmgr.LoadRegistry(registry)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	job, ok := reg.Get(gardenTickLoopID)
	if !ok {
		t.Fatalf("loop %q not registered", gardenTickLoopID)
	}
	if !job.State.Armed() {
		t.Fatalf("loop %q state = %q, want armed", gardenTickLoopID, job.State)
	}
	if job.Schedule.IntervalSeconds != gardenTickIntervalSeconds {
		t.Fatalf("interval = %d, want %d", job.Schedule.IntervalSeconds, gardenTickIntervalSeconds)
	}
	created := job.CreatedUnixNano

	// Re-register: idempotent, original creation timestamp preserved.
	if err := registerGardenTickLoop(registry); err != nil {
		t.Fatalf("re-register: %v", err)
	}
	reg2, _ := loopmgr.LoadRegistry(registry)
	job2, _ := reg2.Get(gardenTickLoopID)
	if job2.CreatedUnixNano != created {
		t.Fatalf("re-register changed CreatedUnixNano %d -> %d", created, job2.CreatedUnixNano)
	}
}

func TestGardenTickRegisterAcceptsTildeDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", home)
	}

	var stdout, stderr bytes.Buffer
	registry := filepath.Join(t.TempDir(), "registry.json")
	code := runGardenTick(&stdout, &stderr, []string{
		"--register",
		"--registry", registry,
		"--dir", "~/leases",
	})
	if code != 0 {
		t.Fatalf("runGardenTick code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
}

// TestWitnessGardenTickRecordsRunEnd proves every tick appends the claim+verdict PAIR
// to the loop ledger under ONE run id: an EventEnd carrying the tick's own claim, and
// an EventWitness carrying the verdict the folded member envelopes prove.
func TestWitnessGardenTickRecordsRunEnd(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	plan := gardenbundle.PlanTick([]gardenbundle.MemberResult{
		{Key: "stale_leases", Label: "stale leases", State: "action"},
	}, false)

	witnessGardenTick(ledger, plan, 2, 1, 0, 0, 0, 3, 4)

	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		t.Fatalf("LoadPrefix: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("want 2 ledger events (end + witness), got %d", len(events))
	}
	ev := events[0]
	if ev.LoopID != gardenTickLoopID {
		t.Fatalf("LoopID = %q, want %q", ev.LoopID, gardenTickLoopID)
	}
	// The END channel carries the tick's own CLAIM. It must not wear the witness
	// vocabulary: loopmgr only counts EventWitness toward Witnessed, so a verdict
	// filed here reads as an unwitnessed run in `fak loop health`.
	if ev.Kind != loopmgr.EventEnd || ev.Status != loopmgr.StatusClaimedDone {
		t.Fatalf("kind/status = %s/%s, want end/claimed_done", ev.Kind, ev.Status)
	}
	if ev.Metrics["reaped_leases"] != 2 {
		t.Fatalf("reaped_leases metric = %d, want 2", ev.Metrics["reaped_leases"])
	}
	if ev.Metrics["reaped_intents"] != 3 {
		t.Fatalf("reaped_intents metric = %d, want 3", ev.Metrics["reaped_intents"])
	}
	if ev.Metrics["folded_sentinel_lines"] != 4 {
		t.Fatalf("folded_sentinel_lines metric = %d, want 4", ev.Metrics["folded_sentinel_lines"])
	}

	w := events[1]
	if w.Kind != loopmgr.EventWitness || w.Status != loopmgr.StatusWitnessedDone {
		t.Fatalf("witness kind/status = %s/%s, want witness/witnessed_done", w.Kind, w.Status)
	}
	// Same run id on both, or the ledger reads as two half-runs instead of one
	// claimed-and-witnessed run.
	if w.RunID != ev.RunID || w.RunID == "" {
		t.Fatalf("witness RunID = %q, want the end event's %q", w.RunID, ev.RunID)
	}
	if len(w.EvidenceRefs) == 0 {
		t.Fatal("witness carries no evidence ref for the folded member envelopes")
	}
}

// TestWitnessGardenTickWithholdsTheVerdictWhenAMemberErrored proves an unreadable
// member downgrades the verdict to witness_unavailable: the tick cannot prove it swept
// everything when a member produced no usable payload. The END claim is unaffected —
// the tick did run — so the run still counts, it just is not proven done.
func TestWitnessGardenTickWithholdsTheVerdictWhenAMemberErrored(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	plan := gardenbundle.PlanTick([]gardenbundle.MemberResult{
		{Key: "stale_leases", Label: "stale leases", State: "action"},
		{Key: "scorecard", Label: "scorecard control pane", State: "errored"},
	}, false)

	witnessGardenTick(ledger, plan, 0, 0, 0, 0, 0, 0, 0)

	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		t.Fatalf("LoadPrefix: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("want 2 ledger events (end + witness), got %d", len(events))
	}
	if events[0].Kind != loopmgr.EventEnd || events[0].Status != loopmgr.StatusClaimedDone {
		t.Fatalf("kind/status = %s/%s, want end/claimed_done", events[0].Kind, events[0].Status)
	}
	w := events[1]
	if w.Kind != loopmgr.EventWitness || w.Status != loopmgr.StatusWitnessUnavailable {
		t.Fatalf("witness kind/status = %s/%s, want witness/witness_unavailable", w.Kind, w.Status)
	}
	if w.Reason != "GARDEN_TICK_MEMBER_ERRORED" {
		t.Fatalf("witness reason = %q, want GARDEN_TICK_MEMBER_ERRORED", w.Reason)
	}
	if !strings.Contains(w.Summary, "1 member(s) errored") {
		t.Fatalf("witness summary = %q, want the errored-member count", w.Summary)
	}
}

// TestGardenTickUnmeasuredCountsOnlyErroredMembers pins the boundary the verdict turns
// on: a red or action member is a MEASURED finding the tick surfaced, so it must not
// withhold the verdict; only an unreadable ("errored") member does.
func TestGardenTickUnmeasuredCountsOnlyErroredMembers(t *testing.T) {
	plan := gardenbundle.PlanTick([]gardenbundle.MemberResult{
		{Key: "a", State: "ok"},
		{Key: "b", State: "action"},
		{Key: "c", State: "red"},
		{Key: "d", State: "errored"},
		{Key: "e", State: "errored"},
	}, false)
	if got := gardenTickUnmeasured(plan); got != 2 {
		t.Fatalf("gardenTickUnmeasured = %d, want 2 (only the errored members)", got)
	}
}

func TestInspectGardenReclaimSurfacesQueueWithoutMutatingRefs(t *testing.T) {
	dir, _ := wipReclaimFixture(t)
	before, err := gitWipOut(context.Background(), dir, nil, "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak/wip/")
	if err != nil {
		t.Fatal(err)
	}
	got := inspectGardenReclaim(io.Discard, dir)
	after, err := gitWipOut(context.Background(), dir, nil, "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak/wip/")
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("read-only inspection changed checkpoint refs:\nbefore=%s\nafter=%s", before, after)
	}
	if got.State != "action" || got.Counts["LAND_READY"] != 2 {
		t.Fatalf("result = %+v, want two land-ready lifecycle rows", got)
	}
	if !strings.Contains(got.Detail, "head=LAND_READY") || !strings.Contains(got.Detail, "fak wip reconcile adopt alpha") {
		t.Fatalf("detail = %q, want executable lifecycle head", got.Detail)
	}
}

func TestGardenTickIncludesReclaimInspection(t *testing.T) {
	old := gardenReclaimInspect
	t.Cleanup(func() { gardenReclaimInspect = old })
	gardenReclaimInspect = func(io.Writer, string) gardenbundle.MemberResult {
		return gardenbundle.MemberResult{Key: "commit_lifecycle", Label: "Commit lifecycle queue", State: "action", Detail: "2 non-terminal; head=LAND_READY next=fak wip reconcile adopt oldest"}
	}
	t.Setenv("FAK_GARDEN", "1")
	root := t.TempDir()
	var out, stderr bytes.Buffer
	code := runGardenTick(&out, &stderr, []string{"--workspace", root, "--dir", root, "--dry-run", "--timeout", "1", "--json"})
	if code != 0 {
		t.Fatalf("runGardenTick code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), `"key": "commit_lifecycle"`) || !strings.Contains(out.String(), "fak wip reconcile adopt oldest") {
		t.Fatalf("tick omitted reclaim queue: %s", out.String())
	}
}

func TestGardenTickCollectionCheckpointResumesWithoutReplayingPrefix(t *testing.T) {
	oldCollect := gardenCollectBounded
	oldInspect := gardenReclaimInspect
	t.Cleanup(func() {
		gardenCollectBounded = oldCollect
		gardenReclaimInspect = oldInspect
	})
	gardenReclaimInspect = func(io.Writer, string) gardenbundle.MemberResult {
		return gardenbundle.MemberResult{Key: gardenTickReclaimKey, Label: "Commit lifecycle queue", State: "ok"}
	}

	root := t.TempDir()
	cursor := filepath.Join(root, "tick-cursor.json")
	ledger := filepath.Join(root, "loops.jsonl")
	t.Setenv("FAK_GARDEN", "1")

	first := gardenbundle.MemberResult{Key: "scorecard", Label: "scorecard", State: "ok", OK: true, Verdict: "OK"}
	gardenCollectBounded = func(_ string, opt gardenbundle.CollectOptions) ([]gardenbundle.MemberResult, gardenbundle.CollectProgress) {
		if opt.Next != "" || len(opt.Prior) != 0 {
			t.Fatalf("first pass resume = next %q prior %d", opt.Next, len(opt.Prior))
		}
		results := []gardenbundle.MemberResult{first}
		if err := opt.Checkpoint("fresh_status", results); err != nil {
			t.Fatalf("checkpoint first member: %v", err)
		}
		return results, gardenbundle.CollectProgress{
			Total: 2, Completed: 1, Ran: []string{"scorecard"},
			Deferred: []string{"fresh_status"}, Next: "fresh_status", Exhausted: true,
		}
	}
	var stdout, stderr bytes.Buffer
	code := runGardenTick(&stdout, &stderr, []string{
		"--workspace", root, "--dir", root, "--cursor", cursor, "--ledger", ledger,
		"--budget", "1", "--timeout", "1", "--json",
	})
	if code != 0 {
		t.Fatalf("first pass code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var firstOut map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &firstOut); err != nil {
		t.Fatalf("first JSON: %v\n%s", err, stdout.String())
	}
	if firstOut["status"] != "partial" || firstOut["reason"] != "GARDEN_TICK_BUDGET_EXHAUSTED" {
		t.Fatalf("first output = %v", firstOut)
	}
	cur, err := gardenbudget.LoadCursor(cursor)
	if err != nil {
		t.Fatalf("load first cursor: %v", err)
	}
	state := decodeGardenTickState(cur.Payload)
	if cur.Stage != gardenTickStageCollect || cur.Next != "fresh_status" || len(state.Results) != 1 {
		t.Fatalf("first cursor = %+v state=%+v", cur, state)
	}

	second := gardenbundle.MemberResult{Key: "fresh_status", Label: "fresh status", State: "ok", OK: true, Verdict: "OK"}
	gardenCollectBounded = func(_ string, opt gardenbundle.CollectOptions) ([]gardenbundle.MemberResult, gardenbundle.CollectProgress) {
		if opt.Next != "fresh_status" || len(opt.Prior) != 1 || opt.Prior[0].Key != "scorecard" {
			t.Fatalf("second pass did not resume prefix: next=%q prior=%+v", opt.Next, opt.Prior)
		}
		results := append(append([]gardenbundle.MemberResult{}, opt.Prior...), second)
		if err := opt.Checkpoint("", results); err != nil {
			t.Fatalf("checkpoint completed collection: %v", err)
		}
		// Spend the rest of the one-second budget so the command checkpoints the
		// reclaim stage instead of entering a real action phase in this fixture.
		time.Sleep(1100 * time.Millisecond)
		return results, gardenbundle.CollectProgress{
			Total: 2, Completed: 2, Ran: []string{"fresh_status"}, Complete: true,
		}
	}
	stdout.Reset()
	stderr.Reset()
	code = runGardenTick(&stdout, &stderr, []string{
		"--workspace", root, "--dir", root, "--cursor", cursor, "--ledger", ledger,
		"--budget", "1", "--timeout", "1", "--json",
	})
	if code != 0 {
		t.Fatalf("second pass code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	cur, err = gardenbudget.LoadCursor(cursor)
	if err != nil {
		t.Fatalf("load second cursor: %v", err)
	}
	state = decodeGardenTickState(cur.Payload)
	if cur.Stage != gardenTickStageReclaim || cur.Next != gardenTickReclaimKey || len(state.Results) != 2 {
		t.Fatalf("second cursor = %+v state=%+v", cur, state)
	}
}

func TestGardenWatchdogDefaultLiveBoundIsShorterThanAMinute(t *testing.T) {
	if gardenWatchdogTimeoutSeconds < 30 || gardenWatchdogTimeoutSeconds > 60 {
		t.Fatalf("default watchdog timeout = %ds, want 30..60s", gardenWatchdogTimeoutSeconds)
	}
	if gardenWatchdogTickBudgetSeconds <= 0 || gardenWatchdogTickBudgetSeconds >= gardenWatchdogTimeoutSeconds {
		t.Fatalf("tick budget = %ds, want positive and below outer %ds",
			gardenWatchdogTickBudgetSeconds, gardenWatchdogTimeoutSeconds)
	}
}

func TestGardenWatchdogExplicitlyRefusesOverlap(t *testing.T) {
	root := t.TempDir()
	lock, acquired, err := acquireGardenWatchdogLock(root, time.Now(), time.Minute)
	if err != nil || !acquired {
		t.Fatalf("prime watchdog lock = (%v, %v)", acquired, err)
	}
	defer lock.release()

	var stdout, stderr bytes.Buffer
	code := runGardenWatchdogConfigured(&stdout, &stderr, gardenWatchdogConfig{
		Repo: root, AsJSON: true, Timeout: time.Second, TickBudget: 500 * time.Millisecond,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got gardenWatchdogEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("JSON: %v\n%s", err, stdout.String())
	}
	if !got.OverlapRefused || got.Status != "refused" || got.Reason != "SKIPPED_CONTENDED" {
		t.Fatalf("overlap result = %+v", got)
	}
}

const (
	gardenWatchdogHelperEnv      = "FAK_GARDEN_WATCHDOG_HELPER"
	gardenWatchdogHeartbeatEnv   = "FAK_GARDEN_WATCHDOG_HEARTBEAT"
	gardenWatchdogChildMode      = "child"
	gardenWatchdogGrandchildMode = "grandchild"
)

// TestGardenWatchdogHangingChildHelper turns the cmd/fak test binary into the
// hanging child/grandchild process tree used by the regression below. It is a
// normal no-op test in the parent process; the re-exec selects only this test.
func TestGardenWatchdogHangingChildHelper(t *testing.T) {
	t.Helper()
	if os.Getenv(gardenWatchdogHelperEnv) == "" {
		t.Skip("helper process")
		return
	}
	switch os.Getenv(gardenWatchdogHelperEnv) {
	case gardenWatchdogGrandchildMode:
		path := os.Getenv(gardenWatchdogHeartbeatEnv)
		for deadline := time.Now().Add(2 * time.Minute); time.Now().Before(deadline); {
			_ = os.WriteFile(path, []byte(time.Now().Format(time.RFC3339Nano)), 0o644)
			time.Sleep(80 * time.Millisecond)
		}
		os.Exit(0)
	case gardenWatchdogChildMode:
		gc := exec.Command(os.Args[0], "-test.run=^TestGardenWatchdogHangingChildHelper$")
		gc.Env = append(os.Environ(), gardenWatchdogHelperEnv+"="+gardenWatchdogGrandchildMode)
		if err := gc.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "spawn grandchild: %v\n", err)
			os.Exit(2)
		}
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	}
}

// TestGardenWatchdogTimeoutKeepsJSONAndReapsDescendantTree is the issue's
// hanging-child witness. The direct child spawns a grandchild that heartbeats a
// file. The outer timeout must return typed JSON and make that heartbeat stop;
// killing only the direct PID leaves the heartbeat advancing and fails.
func TestGardenWatchdogTimeoutKeepsJSONAndReapsDescendantTree(t *testing.T) {
	oldFactory := gardenWatchdogCommand
	t.Cleanup(func() { gardenWatchdogCommand = oldFactory })
	heartbeat := filepath.Join(t.TempDir(), "grandchild.heartbeat")
	gardenWatchdogCommand = func(_, _ string, _ time.Duration) *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=^TestGardenWatchdogHangingChildHelper$")
		cmd.Env = append(os.Environ(),
			gardenWatchdogHelperEnv+"="+gardenWatchdogChildMode,
			gardenWatchdogHeartbeatEnv+"="+heartbeat,
		)
		return cmd
	}

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := runGardenWatchdogConfigured(&stdout, &stderr, gardenWatchdogConfig{
		Repo: t.TempDir(), Live: true, AsJSON: true,
		Timeout: 2 * time.Second, TickBudget: time.Second,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if elapsed := time.Since(start); elapsed > 15*time.Second {
		t.Fatalf("2s watchdog returned after %s", elapsed)
	}
	var got gardenWatchdogEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("JSON did not survive timeout: %v\n%s", err, stdout.String())
	}
	if got.Garden.Status != "timeout" || got.Garden.Reason != "GARDEN_TICK_TIMEOUT" ||
		!got.Garden.TimedOut || got.Garden.Progress == nil {
		t.Fatalf("timeout garden result = %+v", got.Garden)
	}

	// The grandchild must have started and then gone stale after the tree reap.
	createDeadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(heartbeat); err == nil {
			break
		}
		if time.Now().After(createDeadline) {
			t.Fatalf("grandchild never produced heartbeat %s", heartbeat)
		}
		time.Sleep(100 * time.Millisecond)
	}
	staleDeadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(staleDeadline) {
		if info, err := os.Stat(heartbeat); err == nil && time.Since(info.ModTime()) > time.Second {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("grandchild heartbeat kept advancing after watchdog timeout; descendant tree was orphaned")
}

func TestGardenWatchdogGoPortSweepsOnlyOldEphemera(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	old := filepath.Join(root, ".dos", "markers", "old.jsonl")
	fresh := filepath.Join(root, ".dos", "markers", "fresh.jsonl")
	if err := os.MkdirAll(filepath.Dir(old), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(old, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fresh, []byte("fresh"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(old, now.Add(-10*24*time.Hour), now.Add(-10*24*time.Hour))
	_ = os.Chtimes(fresh, now.Add(-time.Hour), now.Add(-time.Hour))

	candidates := scanGardenWatchdogEphemera(root, now, 7)
	if len(candidates) != 1 || candidates[0].Rel != ".dos/markers/old.jsonl" {
		t.Fatalf("candidates = %+v", candidates)
	}
	swept := sweepGardenWatchdog(root, candidates, true)
	if swept.Files != 1 {
		t.Fatalf("swept = %+v", swept)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("old ephemera survived: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh ephemera was removed: %v", err)
	}
}

func TestGardenWatchdogGoPortPreservesTypedStaleGate(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	old := filepath.Join(root, ".dos", "markers", "old.jsonl")
	stuck := filepath.Join(root, ".dos", "stop-failures", "wedged.json")
	if err := os.MkdirAll(filepath.Dir(old), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(stuck), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(old, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stuck, []byte(`{"consecutive":4,"total":9}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(old, now.Add(-10*24*time.Hour), now.Add(-10*24*time.Hour))
	_ = os.Chtimes(stuck, now.Add(-time.Hour), now.Add(-time.Hour))

	var stdout, stderr bytes.Buffer
	code := runGardenWatchdogConfigured(&stdout, &stderr, gardenWatchdogConfig{
		Repo: root, MaxAgeDays: 7, StuckThreshold: 3, WIPStaleHours: 24,
		FailOnStale: true, AsJSON: true, Now: func() time.Time { return now },
	})
	if code != 2 {
		t.Fatalf("stale gate code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var got gardenWatchdogEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("JSON: %v\n%s", err, stdout.String())
	}
	if !got.HasStale || got.AgeGC.Files != 1 || len(got.Stuck) != 1 ||
		got.Stuck[0].Session != "wedged" || got.Garden.Reason != "DRY_RUN" {
		t.Fatalf("typed stale envelope = %+v", got)
	}
}

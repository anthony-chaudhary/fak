package main

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

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

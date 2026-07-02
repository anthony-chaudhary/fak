package main

// Witness tests for the shared region-admission seam: `fak loop drive` and the
// dispatch tick consulting ONE lease fabric + ONE decision (internal/regionadmit)
// so a loop, a dispatch worker, and a manual session are mutually visible before
// any of them mutates a file tree.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// initRegionTestRepo creates a temp git repo with a small dos.toml lane
// taxonomy and chdirs into it (the loop driver reads the lease store and the
// taxonomy from the cwd). Skips when git is unavailable.
func initRegionTestRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if out, err := exec.Command("git", "init", "-q", dir).CombinedOutput(); err != nil {
		t.Skipf("git init failed: %v %s", err, out)
	}
	toml := `[lanes]
concurrent = ["gateway", "docs"]
exclusive = ["release"]

[lanes.trees]
gateway = ["internal/gateway/**"]
docs = ["docs/**", "README.md"]
release = ["VERSION", "docs/releases/**"]
`
	if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(toml), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	return dir
}

func acquireRegionTestLease(t *testing.T, dir, id, holder string, tree []string) {
	t.Helper()
	store := leaseref.NewInDir(dir)
	_, verdict, err := store.AcquireFenced(context.Background(), leaseref.Record{
		ID: id, Holder: holder, TreeGlobs: tree, TTLSeconds: 600,
	}, time.Now())
	if err != nil {
		t.Fatalf("acquire test lease: %v", err)
	}
	if !verdict.OK {
		t.Fatalf("acquire test lease refused: %+v", verdict)
	}
}

func writeRegionGoal(t *testing.T, path, lane string) {
	t.Helper()
	body := fmt.Sprintf(`---
loop: region-loop
witness: commit-audit
lane: %s
budget: { max_iters: 1 }
---
# Objective
Prove the region admission.

# Plan
- [ ] one step

# Scratch / last-refusal
`, lane)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestLoopDriveRefusesOverlappingRegion is the headline witness: a live lease
// on the gateway tree (as a dispatch worker would hold) makes a gateway-lane
// loop drive REFUSE with COLLISION_RISK before spawning its child — the two
// surfaces are mutually visible for the first time.
func TestLoopDriveRefusesOverlappingRegion(t *testing.T) {
	dir := initRegionTestRepo(t)
	oldNewCommand := loopDriveNewCommand
	defer func() { loopDriveNewCommand = oldNewCommand }()

	acquireRegionTestLease(t, dir, "resolve-gateway", "dispatch-peer", []string{"internal/gateway/**"})

	goal := filepath.Join(dir, "GOAL.md")
	ledger := filepath.Join(dir, "loops.jsonl")
	writeRegionGoal(t, goal, "gateway")

	childRan := false
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		childRan = true
		return &loopDriveFakeCommand{}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--", "worker"})
	if code != 3 {
		t.Fatalf("drive code=%d, want 3 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if childRan {
		t.Fatal("the child must never spawn over a live overlapping lease")
	}
	if !strings.Contains(stderr.String(), "COLLISION_RISK") || !strings.Contains(stderr.String(), "resolve-gateway") {
		t.Fatalf("stderr must name the refusal and the conflicting lease: %s", stderr.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, ev := range events {
		if ev.Kind == loopmgr.EventAdmit && ev.Status == loopmgr.StatusRefused && ev.Reason == "COLLISION_RISK" {
			found = true
			if !strings.Contains(ev.Summary, "resolve-gateway") {
				t.Fatalf("refusal event must carry the conflicting lease as evidence: %+v", ev)
			}
		}
	}
	if !found {
		t.Fatalf("ledger missing COLLISION_RISK refused admit: %+v", events)
	}
}

// TestLoopDriveHoldsAndReleasesRegionLease proves the hold half: while the
// drive's child runs, the loop's region lease is LIVE on the shared fabric
// (visible to any dispatch tick or peer that reads it); after the drive exits
// the lease is released.
func TestLoopDriveHoldsAndReleasesRegionLease(t *testing.T) {
	dir := initRegionTestRepo(t)
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(dir, "GOAL.md")
	ledger := filepath.Join(dir, "loops.jsonl")
	writeRegionGoal(t, goal, "gateway")

	store := leaseref.NewInDir(dir)
	var liveDuringTurn []leaseref.Record
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{wait: func() error {
			live, _, err := store.Live(context.Background(), time.Now())
			if err != nil {
				return err
			}
			liveDuringTurn = live
			return nil
		}}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--", "worker"})
	if code != 0 {
		t.Fatalf("drive code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if len(liveDuringTurn) != 1 || liveDuringTurn[0].ID != "loop-region-loop" {
		t.Fatalf("the loop's region lease must be live on the fabric during the turn, got %+v", liveDuringTurn)
	}
	if got := liveDuringTurn[0].TreeGlobs; len(got) != 1 || got[0] != "internal/gateway/**" {
		t.Fatalf("the lease must carry the lane's canonical tree, got %v", got)
	}
	liveAfter, _, err := store.Live(context.Background(), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(liveAfter) != 0 {
		t.Fatalf("the region lease must be released when the drive exits, still live: %+v", liveAfter)
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatal(err)
	}
	foundEvidence := false
	for _, ev := range events {
		for _, ref := range ev.EvidenceRefs {
			if ref.Kind == "region_lease" && ref.Ref == "loop-region-loop" {
				foundEvidence = true
			}
		}
	}
	if !foundEvidence {
		t.Fatalf("ledger events must carry the held region lease as evidence: %+v", events)
	}
}

// TestLoopDriveWithoutRegionStaysUncoordinated pins the zero-config contract:
// no lane, no region, no flags = no lease acquired, no admission consulted.
func TestLoopDriveWithoutRegionStaysUncoordinated(t *testing.T) {
	dir := initRegionTestRepo(t)
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	// A live lease on the WHOLE tree would refuse any coordinated drive...
	acquireRegionTestLease(t, dir, "resolve-global", "peer", []string{"**/*"})

	goal := filepath.Join(dir, "GOAL.md")
	ledger := filepath.Join(dir, "loops.jsonl")
	writeLoopDriveGoal(t, goal, true, true)

	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--max-iters", "1", "--", "worker"})
	if code != 0 {
		t.Fatalf("an undeclared region must keep the historical uncoordinated drive, code=%d stderr=%s", code, stderr.String())
	}
}

// TestLoopRegionVerb proves the surface-neutral decision verb a manual session
// or super-loop enter path runs: admit on a free region, refuse with the lane
// serialization rung when the named lane is already held on the fabric.
func TestLoopRegionVerb(t *testing.T) {
	dir := initRegionTestRepo(t)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"region", "--lane", "gateway", "--actor", "session:me", "--dir", dir})
	if code != 0 {
		t.Fatalf("free lane must admit, code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "ADMIT") {
		t.Fatalf("stdout missing ADMIT: %s", stdout.String())
	}

	// A dispatch-shaped lease on the gateway lane's canonical tree: the verb
	// must refuse a same-lane request via lane serialization, not just geometry.
	acquireRegionTestLease(t, dir, "resolve-gateway", "dispatch-peer", []string{"internal/gateway/**"})

	stdout.Reset()
	stderr.Reset()
	code = runLoop(&stdout, &stderr, []string{"region", "--lane", "gateway", "--tree", "docs/gateway.md", "--actor", "session:me", "--dir", dir, "--json"})
	if code != leaserefRefused {
		t.Fatalf("held lane must refuse with exit %d, got %d stderr=%s stdout=%s", leaserefRefused, code, stderr.String(), stdout.String())
	}
	var payload struct {
		Admit    bool   `json:"admit"`
		Reason   string `json:"reason"`
		Rung     string `json:"rung"`
		Conflict struct {
			ID     string `json:"id"`
			Holder string `json:"holder"`
		} `json:"conflict"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if payload.Admit || payload.Reason != "COLLISION_RISK" || payload.Rung != "same_lane_live" {
		t.Fatalf("payload = %+v, want same_lane_live COLLISION_RISK refusal", payload)
	}
	if payload.Conflict.ID != "resolve-gateway" || payload.Conflict.Holder != "dispatch-peer" {
		t.Fatalf("conflict evidence must name the holder, got %+v", payload.Conflict)
	}
}

// TestLoopDriveRegionHoldRefusesDoppelganger is the pinned-identity witness:
// with a fleet-pinned FAK_LEASE_OWNER (standard in resume-wave ops) two
// concurrent drives of the SAME loop must not both hold the region — the
// process-unique holder makes the second one refuse at the fence instead of
// silently sharing the first one's lease.
func TestLoopDriveRegionHoldRefusesDoppelganger(t *testing.T) {
	initRegionTestRepo(t)
	t.Setenv("FAK_LEASE_OWNER", "pinned-fleet-owner")

	spec := loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}
	now := time.Now()

	first := newLoopDriveRegionHold(loopDriveOptions{}, spec)
	if refuse, err := first.ensure(now); err != nil || refuse != nil {
		t.Fatalf("first drive must hold: refuse=%+v err=%v", refuse, err)
	}
	defer first.release()

	second := newLoopDriveRegionHold(loopDriveOptions{}, spec)
	refuse, err := second.ensure(now)
	if err != nil {
		t.Fatal(err)
	}
	if refuse == nil {
		t.Fatal("a concurrent drive of the same loop under a pinned owner must refuse, not share the lease")
	}
	if refuse.Reason != leaseref.ReasonLeaseHeld {
		t.Fatalf("refusal reason = %q, want %q", refuse.Reason, leaseref.ReasonLeaseHeld)
	}
	// And the doppelganger's renew path must not hijack the region either:
	// its ensure never held, so the first drive's later renew still succeeds.
	if refuse, err := first.ensure(now.Add(time.Minute)); err != nil || refuse != nil {
		t.Fatalf("the true holder's renew must survive the doppelganger: refuse=%+v err=%v", refuse, err)
	}
}

// TestLoopDriveRegionHoldReacquiresAfterLapse: a lease that expired with no
// taker (one turn outran the TTL) is not a peer conflict — the next ensure
// reacquires instead of honest-stopping a contention-free drive.
func TestLoopDriveRegionHoldReacquiresAfterLapse(t *testing.T) {
	initRegionTestRepo(t)

	spec := loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}
	now := time.Now()
	hold := newLoopDriveRegionHold(loopDriveOptions{}, spec)
	if refuse, err := hold.ensure(now); err != nil || refuse != nil {
		t.Fatalf("acquire: refuse=%+v err=%v", refuse, err)
	}
	defer hold.release()

	// Two TTLs later the lease has lapsed untaken; ensure must self-heal.
	later := now.Add(time.Duration(2*loopDriveRegionTTLS) * time.Second)
	if refuse, err := hold.ensure(later); err != nil || refuse != nil {
		t.Fatalf("lapsed-untaken lease must reacquire, got refuse=%+v err=%v", refuse, err)
	}
	if !hold.held {
		t.Fatal("hold must be re-held after the lapse")
	}

	// But when a PEER takes over the lapsed lease (fenced expired-takeover,
	// generation bump), the drive's next renew must refuse — that region is
	// genuinely someone else's now.
	takeover := later.Add(time.Duration(2*loopDriveRegionTTLS) * time.Second)
	_, verdict, err := hold.store.AcquireFenced(context.Background(), leaseref.Record{
		ID: hold.id, Holder: "peer-holder", TreeGlobs: []string{"internal/gateway/**"}, TTLSeconds: 600,
	}, takeover)
	if err != nil || !verdict.OK {
		t.Fatalf("peer takeover of the lapsed lease: verdict=%+v err=%v", verdict, err)
	}
	refuse, err := hold.ensure(takeover.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if refuse == nil {
		t.Fatal("a peer holding the lease must refuse the drive's renew")
	}
	if refuse.Reason != leaseref.ReasonStaleLease {
		t.Fatalf("refusal reason = %q, want %q", refuse.Reason, leaseref.ReasonStaleLease)
	}
}

// TestDispatchAcquireHonorsLaneTaxonomy proves the dispatch side of the shared
// seam: the tick's lane-lease acquire now refuses on the SAME decision — here
// an exclusive-lane live lease (a release cut) blocks a disjoint-tree lane,
// which the historical geometry-only scan would have admitted.
func TestDispatchAcquireHonorsLaneTaxonomy(t *testing.T) {
	dir := initRegionTestRepo(t)
	acquireRegionTestLease(t, dir, "release-cut", "op", []string{"VERSION", "docs/releases/**"})

	lease := acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600)
	refused, _ := lease["refused"].(bool)
	if !refused {
		t.Fatalf("an exclusive-lane live lease must refuse every new region, got %+v", lease)
	}
	if lease["reason"] != "COLLISION_RISK" || lease["rung"] != "exclusive_lane_live" {
		t.Fatalf("refusal must carry the exclusive_lane_live rung, got %+v", lease)
	}

	// And the loop-held lease is visible to dispatch the same way: release the
	// exclusive lease, hold a loop lease on gateway, dispatch must refuse it.
	store := leaseref.NewInDir(dir)
	if err := store.Release(context.Background(), "release-cut"); err != nil {
		t.Fatal(err)
	}
	acquireRegionTestLease(t, dir, "loop-nightly", "loop:nightly@host", []string{"internal/gateway/**"})
	lease = acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600)
	if refused, _ := lease["refused"].(bool); !refused {
		t.Fatalf("a loop-held region must refuse a dispatch spawn on the same tree, got %+v", lease)
	}
}

// TestDispatchAcquireRecordsDecidedTree pins decide/record consistency: an
// empty requested tree with a named lane is admitted on the lane's canonical
// taxonomy tree, so the WRITTEN lease must carry that same tree — never an
// empty unknown-blast-radius record after a permissive admit.
func TestDispatchAcquireRecordsDecidedTree(t *testing.T) {
	dir := initRegionTestRepo(t)
	lease := acquireDispatchLaneLease(dir, "resolve-docs", "docs", nil, 600)
	if acquired, _ := lease["acquired"].(bool); !acquired {
		t.Fatalf("free lane with empty tree must acquire on the lane's canonical tree, got %+v", lease)
	}
	live, _, err := leaseref.NewInDir(dir).Live(context.Background(), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(live) != 1 {
		t.Fatalf("live = %+v", live)
	}
	if got := live[0].TreeGlobs; len(got) != 2 || got[0] != "docs/**" || got[1] != "README.md" {
		t.Fatalf("recorded tree = %v, want the docs lane's canonical tree", got)
	}
}

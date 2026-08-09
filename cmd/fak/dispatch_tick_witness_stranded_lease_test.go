package main

// dispatch_tick_witness_stranded_lease_test.go pins #5570 at the stranded-revert rung:
// it must attribute a dirty file by WHO HOLDS THE LANE, not merely by whether the path
// falls inside the dead worker's declared lease-tree globs.
//
// THE DEFECT THESE TESTS CLOSE. dispatchWitnessRevertStranded's only attribution input
// was the DEAD worker's own .tree sidecar, so a live worker editing inside the same globs
// was indistinguishable from the dead worker's strand — and the rung's strongest guard,
// a scoped `go build` that FAILS, is exactly what a live worker at a non-compiling
// intermediate state trips routinely. Its files were then archived and stashed out from
// under it. Recoverable, but a worker whose edits revert mid-run reports nonsense.
//
// WHY THE FIX IS ONLY POSSIBLE NOW. The distinguishing evidence is the lease record, and
// until acquireDispatchLaneLease started stamping leaseref.Record.SessionID every dispatch
// lease took ClassifyLiveness's SessionID == "" branch — peer-unknown for the whole fleet,
// with no way to tell a heartbeating holder from a dead one. The peer-dead escape hatch
// asserted below is that binding's first read-side consumer inside the sweep.
//
// THE SHAPE OF THE TWO HALVES. TestWitnessStrandedRevertYieldsToLiveLaneHolder is the
// fix: a contested lane stands the rung down whole, and peer-unknown fails CLOSED.
// TestWitnessStrandedRevertKeepsFiringOnItsOwnLane is the CONTROL, and it is not a
// restatement of the implementation — a fix that merely refused whenever ANY live lease
// existed would pass the first test and red every case of the second, because the sweep
// reverts BEFORE it hands the dead worker's own lease back, so that lane is still live
// right there.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// seedContestingLaneLease publishes a real lane lease over the stranded repo through the
// production acquire — the same call site that stamps the session binding — so what the
// rung reads back is a genuine refs/fak/locks record and not a hand-rolled fixture.
func seedContestingLaneLease(t *testing.T, root, id, lane string, tree []string, session string) map[string]any {
	t.Helper()
	t.Setenv("FAK_LEASE_OWNER", "holder-of-"+lane)
	t.Setenv("FAK_SESSION_ID", session)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "")
	lease := acquireDispatchLaneLease(root, id, lane, tree, 1800, "")
	if acquired, _ := lease["acquired"].(bool); !acquired {
		t.Fatalf("seeding lane lease %q must acquire on a free lane, got %+v", id, lease)
	}
	return lease
}

// claimStrandedLeaseForWorker writes the .lease-id and #4324 .lease-fence.json sidecars
// that make a lease PROVABLY the swept worker's own — id, holder and generation together.
// Without all three the rung cannot tell the dead worker's un-released lane from the next
// worker's re-acquire of the same lane id.
func claimStrandedLeaseForWorker(t *testing.T, runsDir, stem, id string, lease map[string]any) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchLeaseIDSidecarSuffix), []byte(id), 0o644); err != nil {
		t.Fatalf("write lease-id sidecar: %v", err)
	}
	if p := writeDispatchLeaseFenceSidecar(filepath.Join(runsDir, stem+".log"), lease); p == "" {
		t.Fatalf("an acquired lease must persist a fencing token: %+v", lease)
	}
}

// assertStrandPreserved is the stand-down witness: the poison bytes are still dirty on
// disk, no stash was pushed, and no archive dir was written.
func assertStrandPreserved(t *testing.T, root, runsDir, stem string) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, "cmd", "lane", "tool.go"))
	if err != nil || !strings.Contains(string(b), "func broken(") {
		t.Fatalf("in-lane file = %q err=%v, want left dirty — the rung must not revert a contested lane", b, err)
	}
	if got := strings.TrimSpace(runDispatchGit(t, root, "stash", "list")); got != "" {
		t.Fatalf("a contested lane must produce no stash, stash list = %q", got)
	}
	if _, err := os.Stat(filepath.Join(runsDir, stem+witnessStrandedArchiveSuffix)); !os.IsNotExist(err) {
		t.Fatalf("a contested lane must write no archive, stat err = %v", err)
	}
}

// assertStrandReverted is the rung-still-works witness: the tracked strand is back to its
// committed bytes, via a recoverable stash, with the poison archived first.
func assertStrandReverted(t *testing.T, root, runsDir, stem string) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, "cmd", "lane", "tool.go"))
	if err != nil || string(b) != "package lane\n" {
		t.Fatalf("in-lane file = %q err=%v, want reverted to committed content", b, err)
	}
	if got := runDispatchGit(t, root, "stash", "list"); !strings.Contains(got, "stash@{0}") {
		t.Fatalf("the revert must be a recoverable stash, stash list = %q", got)
	}
	arch := filepath.Join(runsDir, stem+witnessStrandedArchiveSuffix, "cmd", "lane", "tool.go")
	if b, err := os.ReadFile(arch); err != nil || !strings.Contains(string(b), "( {") {
		t.Fatalf("archive = %q err=%v, want the stranded poison bytes written before the stash", b, err)
	}
}

// TestWitnessStrandedRevertYieldsToLiveLaneHolder is the #5570 fix witness. The scene is
// the one the ticket names: a provably-dead no-commit worker declared lane `cmd/**`, its
// strand is dirty and reds a scoped build (stubbed RED, so the build gate is NOT what
// stops anything here), and the lane is now held by a DIFFERENT live lease. The rung must
// touch nothing — and must not even ask the build gate, because attribution is decided
// before evidence-of-poison is gathered.
//
// The three sub-cases are the three liveness classes that are not positive death:
// peer-live is the hazard by name, and both peer-unknown kinds (a bound session whose
// descriptor was never published, and a wholly unbound record) fail CLOSED — the explicit
// decision the ticket asked for, matching this rung's archive-before-destroy posture.
func TestWitnessStrandedRevertYieldsToLiveLaneHolder(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	for _, tc := range []struct {
		name     string
		session  string
		publish  string
		wantLive string
	}{
		{"peer-live holder must never be swept", "holder-sess-live", "RUNNING", leaseref.LivenessPeerLive},
		{"bound holder with no descriptor fails closed", "holder-sess-ghost", "", leaseref.LivenessPeerUnknown},
		{"unbound holder fails closed", "", "", leaseref.LivenessPeerUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, runsDir, stem := seedStrandedTrunk(t)
			seedContestingLaneLease(t, root, "resolve-cmd-peer", "cmd", []string{"cmd/**"}, tc.session)
			now := time.Now()
			if tc.publish != "" {
				dispatchSessionPublish(t, root, tc.session, tc.publish, now, 3600)
			}
			// Precondition: the lane really is held, and really classifies the way this
			// case claims. Without it a broken fixture would stand the rung down for the
			// wrong reason and the test would guard nothing.
			if got := dispatchClassifiedLease(t, root, "resolve-cmd-peer", now); got.Liveness != tc.wantLive {
				t.Fatalf("contesting lease liveness = %q, want %q (evidence: %s)", got.Liveness, tc.wantLive, got.Evidence)
			}

			withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
			builds := withWitnessStrandedBuildStub(t, true, true) // the strand DOES red the build

			_, records := witnessExitedWorkers(root, runsDir, true)

			if len(records) != 1 {
				t.Fatalf("records = %+v, want the one swept slot", records)
			}
			if len(*builds) != 0 {
				t.Fatalf("scoped build ran %v; a contested lane must stand the rung down BEFORE the build gate", *builds)
			}
			assertStrandPreserved(t, root, runsDir, stem)
		})
	}
}

// TestWitnessStrandedRevertKeepsFiringOnItsOwnLane is the CONTROL — every case here is
// green both with and without the #5570 gate, and together they are what stops the fix
// from being "refuse whenever any live lease exists", which would retire the rung.
//
//   - own-lane: the sweep reverts BEFORE dispatchLeaseReleaser hands the lease back, so
//     the dead worker's lane is still live at this exact point. It is recognized by the
//     #4324 fencing token (id AND holder AND generation), which is also what distinguishes
//     it from a re-acquire — the lease id is derived from the LANE, so the next worker
//     reuses it and only the bumped generation tells the two records apart.
//   - peer-dead: a holder whose session published terminal STOPPED has no live writer to
//     protect. Reachable only because the acquire now stamps Record.SessionID; unbound,
//     this same lease would classify peer-unknown and (correctly, then) block.
//   - disjoint: a heartbeating holder of `docs/**` says nothing about `cmd/**`.
func TestWitnessStrandedRevertKeepsFiringOnItsOwnLane(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	for _, tc := range []struct {
		name string
		seed func(t *testing.T, root, runsDir, stem string)
	}{
		{
			name: "its own un-released lane does not contest itself",
			seed: func(t *testing.T, root, runsDir, stem string) {
				lease := seedContestingLaneLease(t, root, "resolve-cmd", "cmd", []string{"cmd/**"}, "swept-worker-sess")
				dispatchSessionPublish(t, root, "swept-worker-sess", "RUNNING", time.Now(), 3600)
				claimStrandedLeaseForWorker(t, runsDir, stem, "resolve-cmd", lease)
			},
		},
		{
			name: "a positively dead holder does not contest",
			seed: func(t *testing.T, root, runsDir, stem string) {
				seedContestingLaneLease(t, root, "resolve-cmd-peer", "cmd", []string{"cmd/**"}, "holder-sess-stopped")
				now := time.Now()
				dispatchSessionPublish(t, root, "holder-sess-stopped", "STOPPED", now, 3600)
				if got := dispatchClassifiedLease(t, root, "resolve-cmd-peer", now); got.Liveness != leaseref.LivenessPeerDead {
					t.Fatalf("holder liveness = %q, want peer-dead (evidence: %s)", got.Liveness, got.Evidence)
				}
			},
		},
		{
			name: "a live holder of a disjoint lane does not contest",
			seed: func(t *testing.T, root, runsDir, stem string) {
				seedContestingLaneLease(t, root, "resolve-docs", "docs", []string{"docs/**"}, "holder-sess-docs")
				dispatchSessionPublish(t, root, "holder-sess-docs", "RUNNING", time.Now(), 3600)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, runsDir, stem := seedStrandedTrunk(t)
			tc.seed(t, root, runsDir, stem)

			withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
			builds := withWitnessStrandedBuildStub(t, true, true)

			_, records := witnessExitedWorkers(root, runsDir, true)

			if len(records) != 1 {
				t.Fatalf("records = %+v, want the one swept slot", records)
			}
			if len(*builds) != 1 || len((*builds)[0]) != 1 || (*builds)[0][0] != "./cmd/lane" {
				t.Fatalf("scoped build pkgs = %v, want one call with [./cmd/lane] — an uncontested lane must still reach the build gate", *builds)
			}
			assertStrandReverted(t, root, runsDir, stem)
			// The out-of-lane peer WIP is untouched in every case, exactly as before.
			if b, err := os.ReadFile(filepath.Join(root, "docs", "peer.md")); err != nil || string(b) != "peer wip - must survive\n" {
				t.Fatalf("out-of-lane WIP = %q err=%v, want untouched", b, err)
			}
		})
	}
}

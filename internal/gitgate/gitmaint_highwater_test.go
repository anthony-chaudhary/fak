package gitgate

// gitmaint_highwater_test.go — the #5084 witnesses for the HIGH-WATER-triggered fold and
// its non-reduction incident.
//
// #4602 Phase 0 landed the predicate (LooseBacklogHigh) as a read-only witness that gated
// nothing, and Phase 1 unblocked the fold tier from session leases — but nothing ever
// fired a fold on the high-water mark, so the reported ~2-minute cold-git stalls stood.
// MaintOptions.RequireBacklogHigh is that trigger, and the two things worth pinning about
// a trigger are that it fires when it should and stays silent when it should not: a gate
// that only ever ran on the high side would be indistinguishable from no gate at all.
//
// Every case runs through the injected MaintRunner — no git, no repo — the same pattern
// TestRunMaintLooseBacklogHighFromPreRunCount uses.

import (
	"context"
	"path/filepath"
	"testing"
)

// gracefulCalls returns the mutating fold argv the run actually issued. The always-safe
// steps are deliberately excluded: the gate must never hold THEM back, so counting them
// as "the fold ran" would make the below-threshold case pass vacuously.
func gracefulCalls(calls [][]string) [][]string {
	var out [][]string
	for _, c := range calls {
		if len(c) < 2 {
			continue
		}
		switch args := c[1:]; {
		case args[0] == "prune-packed",
			args[0] == "maintenance" && len(args) > 1 && args[1] == "run":
			out = append(out, args)
		}
	}
	return out
}

// alwaysSafeRan reports whether the add-only tier executed — the invariant the gate must
// not disturb on a below-threshold clone.
func alwaysSafeRan(res MaintResult) bool { return ranTier(res.Steps, "always-safe") }

// TestHighWaterGateFiresOnlyAboveTheThreshold is the both-directions witness: with
// RequireBacklogHigh set, a pre-run count at/above LooseBacklogThreshold folds, and a
// count below it issues no mutating fold argv at all — while the always-safe tier runs
// either way.
func TestHighWaterGateFiresOnlyAboveTheThreshold(t *testing.T) {
	cases := []struct {
		name      string
		loose     int
		wantFold  bool
		wantRefus MaintReason
	}{
		{"at-threshold-fires", LooseBacklogThreshold, true, ""},
		{"far-above-threshold-fires", LooseBacklogThreshold + 47_751, true, ""},
		{"just-below-threshold-holds", LooseBacklogThreshold - 1, false, MaintReasonBacklogLow},
		{"quiet-clone-holds", 100, false, MaintReasonBacklogLow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gitDir := scratchGit(t)
			f := &fakeMaint{posture: safePosture(), loose: tc.loose, inPack: 500, packs: 4}
			res := RunMaint(context.Background(), f.run, MaintOptions{
				RepoRoot:           filepath.Dir(gitDir),
				GitCommonDir:       gitDir,
				Apply:              true,
				RequireBacklogHigh: true,
			})

			folded := len(gracefulCalls(f.calls)) > 0
			if folded != tc.wantFold {
				t.Errorf("loose=%d (threshold %d): fold ran=%v want %v; argv=%v",
					tc.loose, LooseBacklogThreshold, folded, tc.wantFold, gracefulCalls(f.calls))
			}
			if res.GraceRefused != tc.wantRefus {
				t.Errorf("loose=%d: GraceRefused=%q want %q", tc.loose, res.GraceRefused, tc.wantRefus)
			}
			if !alwaysSafeRan(res) {
				t.Errorf("loose=%d: the always-safe tier must stay unconditional under the gate", tc.loose)
			}
			if !tc.wantFold && res.Incident {
				t.Errorf("loose=%d: a below-threshold clone is healthy, not an incident", tc.loose)
			}
		})
	}
}

// TestUngatedRunStillFoldsALowBacklog pins that the gate is opt-in: the operator-invoked
// `fak git-maint` verb (RequireBacklogHigh unset) still folds a quiet clone on demand. An
// always-on gate would silently turn the manual verb into a no-op below 10k.
func TestUngatedRunStillFoldsALowBacklog(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 4}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true,
	})

	if got := gracefulCalls(f.calls); len(got) == 0 {
		t.Fatalf("an ungated run must still fold a 100-loose clone; issued no fold argv")
	}
	if res.GraceRefused != "" {
		t.Fatalf("an ungated run must not report a backlog refusal, got %q", res.GraceRefused)
	}
}

// TestHighWaterGateIsFailClosedOnAnUnavailableCount: when `git count-objects` cannot be
// read at all the backlog is UNKNOWN, and the gate must read unknown as "do not sweep"
// rather than sweeping blind — the same fail-closed polarity LooseBacklogHigh already has.
func TestHighWaterGateIsFailClosedOnAnUnavailableCount(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: LooseBacklogThreshold * 5, inPack: 500, packs: 4}
	// countBlind fails only the count probe; every other verb behaves normally.
	countBlind := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "count-objects" {
			return "", 1, nil
		}
		return f.run(ctx, dir, args...)
	}
	res := RunMaint(context.Background(), countBlind, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, RequireBacklogHigh: true,
	})

	if res.GraceRefused != MaintReasonBacklogLow {
		t.Fatalf("an unreadable count must hold the gated fold, got %q", res.GraceRefused)
	}
	if got := gracefulCalls(f.calls); len(got) != 0 {
		t.Fatalf("a blind run must issue no fold argv; got %v", got)
	}
	if res.LooseBacklogIncident != "" {
		t.Fatalf("a run that never folded cannot raise a non-reduction incident, got %q", res.LooseBacklogIncident)
	}
}

// TestLooseBacklogHighIncidentOnANonReducingSweep is the escalation witness. A backlog
// that is entirely UNREACHABLE cannot be folded away — the fold relocates reachable loose
// objects into packs and removes nothing else — so the sweep runs, the count stands, and
// that measured non-reduction is exactly the LOOSE_BACKLOG_HIGH signal #5079 (grace-prune)
// exists to answer. Silence here is the failure mode the issue reports: maintenance that
// "ran fine" every day while the stall never went away.
func TestLooseBacklogHighIncidentOnANonReducingSweep(t *testing.T) {
	gitDir := scratchGit(t)
	backlog := LooseBacklogThreshold + 47_751
	f := &fakeMaint{posture: safePosture(), loose: backlog, unreachable: backlog, inPack: 500, packs: 4}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, RequireBacklogHigh: true,
	})

	if len(gracefulCalls(f.calls)) == 0 {
		t.Fatalf("sanity: the gate should have fired on a %d-loose backlog", backlog)
	}
	if res.After.Count < res.Before.Count {
		t.Fatalf("sanity: an all-unreachable backlog cannot fold: before=%d after=%d", res.Before.Count, res.After.Count)
	}
	if res.LooseBacklogIncident != MaintReasonLooseBacklogHigh {
		t.Fatalf("a sweep that ran and did not reduce %d loose objects must raise LOOSE_BACKLOG_HIGH, got %q",
			backlog, res.LooseBacklogIncident)
	}
	if !res.Incident {
		t.Fatal("LOOSE_BACKLOG_HIGH must surface as an incident, not just a field")
	}
	if res.GracePruneRefused != MaintReasonPruneOff {
		t.Fatalf("this leaf folds and escalates; it must not reach for prune itself (got %q)", res.GracePruneRefused)
	}
}

// TestLooseBacklogHighIncidentStaysSilentWhenUnearned covers every way the observation
// would be unearned. Each of these left the count where it found it too, so a naive
// "before <= after" check would raise all four — the point is that only a fold that
// actually RAN and APPLIED against a HIGH, READABLE backlog can prove non-reduction.
func TestLooseBacklogHighIncidentStaysSilentWhenUnearned(t *testing.T) {
	backlog := LooseBacklogThreshold + 2_000
	cases := []struct {
		name  string
		lock  string // a lock path to seed under the common dir
		apply bool
		loose int
		why   string
	}{
		{"dry-run-mutated-nothing", "", false, backlog, "a dry run plans; it never had the chance to reduce anything"},
		{"locked-tier-never-ran", "refs/heads/main.lock", true, backlog, "a deferred fold proves nothing about whether folding works"},
		{"low-backlog-nothing-to-prove", "", true, 100, "a clone below the high-water mark is not carrying the reported backlog"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gitDir := scratchGit(t)
			if tc.lock != "" {
				writeLock(t, gitDir, tc.lock)
			}
			f := &fakeMaint{posture: safePosture(), loose: tc.loose, unreachable: tc.loose, inPack: 500, packs: 4}
			res := RunMaint(context.Background(), f.run, MaintOptions{
				RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: tc.apply, RequireBacklogHigh: true,
			})
			if res.LooseBacklogIncident != "" {
				t.Fatalf("%s: got %q — %s", tc.name, res.LooseBacklogIncident, tc.why)
			}
		})
	}
}

// TestPostureDriftOutranksTheBacklogGate: a drifted shared config is a config-health
// incident an operator must repair, so it must still be reported on a quiet clone rather
// than being masked by the cheaper BACKLOG_LOW hold.
func TestPostureDriftOutranksTheBacklogGate(t *testing.T) {
	gitDir := scratchGit(t)
	drifted := safePosture()
	drifted["gc.auto"] = "6700" // an auto-gc that could prune-race a peer mid-commit
	f := &fakeMaint{posture: drifted, loose: 100, inPack: 500, packs: 4}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, RequireBacklogHigh: true,
	})

	if res.GraceRefused != MaintReasonPostureDrift {
		t.Fatalf("posture drift must outrank BACKLOG_LOW so the operator still sees it, got %q", res.GraceRefused)
	}
	if !res.Incident {
		t.Fatal("posture drift stays an incident under the high-water gate")
	}
}

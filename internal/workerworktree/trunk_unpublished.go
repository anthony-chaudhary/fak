package workerworktree

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	trunkUnpublishedGrace = 30 * time.Minute
	trunkMainRef          = "refs/heads/main"
	trunkOriginMainRef    = "refs/remotes/origin/main"
)

// trunkUnpublishedNow is a test seam for deterministic grace-window tests.
var trunkUnpublishedNow = time.Now

// checkTrunkUnpublishedOrDiverged prevents hook-bypassing Git plumbing
// (commit-tree + update-ref) from stacking commits onto a local main that has
// already diverged from origin/main or stayed unpublished past the grace window.
//
// Probing uses `rev-list --left-right --count` (never `rev-parse`) and fails
// open when origin/main is absent or unparseable, so offline/standalone repos
// and unit-test fakes are unaffected.
func checkTrunkUnpublishedOrDiverged(root, target, wtPath string, git GitRunner) *Result {
	if target != "" && target != trunkMainRef {
		return nil
	}
	ahead, behind, ok := trunkAheadBehind(root, git)
	if !ok || ahead == 0 {
		return nil
	}
	refuse, age := shouldRefuseTrunkStacking(root, ahead, behind, git)
	if !refuse {
		return nil
	}
	// One bounded refresh before refusing so a stale local remote-tracking ref
	// cannot false-refuse after another publisher already pushed origin/main.
	if fetchRC, _ := run(git, root, []string{"fetch", "--quiet", "origin", "main"}); fetchRC == 0 {
		if a2, b2, ok2 := trunkAheadBehind(root, git); ok2 {
			ahead, behind = a2, b2
			if ahead == 0 {
				return nil
			}
			refuse, age = shouldRefuseTrunkStacking(root, ahead, behind, git)
			if !refuse {
				return nil
			}
		}
	}
	var reason string
	if behind > 0 {
		reason = fmt.Sprintf(
			"trunk %s is diverged from %s (ahead %d, behind %d); refusing to stack more commits via plumbing — reconcile with fak-sync repo converge-diverged --apply before landing",
			trunkMainRef, trunkOriginMainRef, ahead, behind,
		)
	} else {
		reason = fmt.Sprintf(
			"trunk %s has %d unpublished commit(s) older than %s (oldest %s); refusing to stack more commits via plumbing — publish trunk before landing",
			trunkMainRef, ahead, trunkUnpublishedGrace, age.Round(time.Second),
		)
	}
	r := isolatedLandReconciliationResult(wtPath, Result{
		OK:     false,
		Code:   LandResultTrunkUnpublished,
		Reason: reason,
	})
	return &r
}

func trunkAheadBehind(root string, git GitRunner) (ahead, behind int, ok bool) {
	rc, out := run(git, root, []string{"rev-list", "--left-right", "--count", trunkMainRef + "..." + trunkOriginMainRef})
	if rc != 0 {
		return 0, 0, false
	}
	fields := strings.Fields(out)
	if len(fields) != 2 {
		return 0, 0, false
	}
	a, errA := strconv.Atoi(fields[0])
	b, errB := strconv.Atoi(fields[1])
	if errA != nil || errB != nil || a < 0 || b < 0 {
		return 0, 0, false
	}
	return a, b, true
}

func shouldRefuseTrunkStacking(root string, ahead, behind int, git GitRunner) (bool, time.Duration) {
	if ahead <= 0 {
		return false, 0
	}
	if behind > 0 {
		return true, 0
	}
	rc, out := run(git, root, []string{"log", "--reverse", "--format=%ct", "-n", "1", trunkOriginMainRef + ".." + trunkMainRef})
	if rc != 0 {
		return false, 0
	}
	ts, err := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	if err != nil || ts <= 0 {
		return false, 0
	}
	age := trunkUnpublishedNow().Sub(time.Unix(ts, 0))
	return age > trunkUnpublishedGrace, age
}

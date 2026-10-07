package workerworktree

import (
	"strings"
	"testing"
)

// prospectiveReuseFake stubs the bounded CAS retry SB-33 is about: a peer lands in
// the compare-and-swap gap, so the FIRST attempt loses and the second re-seeds from
// the peer's new HEAD and wins. diff-tree is the peer's trunk delta — the only input
// the reuse decision may rest on.
func prospectiveReuseFake(deltaPaths string) *fakeGit {
	g := isolatedHappyFake().
		replyOnce("rev-parse", 0, "oldhead000\n").
		replyOnce("rev-parse", 0, "newhead111\n").
		replyOnce("update-ref", 1, "fatal: update_ref failed: ref moved").
		replyOnce("update-ref", 0, "")
	if deltaPaths != "" {
		g.replyOnce("diff-tree", 0, deltaPaths)
	}
	return g
}

func countPhase(receipt *LandCostReceipt, phase string) int {
	n := 0
	for _, p := range receipt.Phases {
		if p.Phase == phase {
			n++
		}
	}
	return n
}

// runProspectiveReuseLand drives the seam through a lost CAS and reports the cost
// receipt the worker's Result.Cost carries.
func runProspectiveReuseLand(t *testing.T, g *fakeGit, closure []string, ran *int) (Result, *LandCostReceipt) {
	t.Helper()
	stubCASSleep(t)
	cfg := landConfig{
		resources:     func() landResourceSample { return landResourceSample{} },
		verifyClosure: closure,
	}
	tracker := newLandProgressTracker(cfg)
	cfg.tracker = tracker
	prospective := func(string, error) Result {
		*ran++
		return Result{OK: true}
	}
	res, handled := landIsolatedProspectiveVerified("/trunk", "/wt",
		"diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"},
		prospective, nil, g.run, g.runEnv, cfg, "base")
	if !handled {
		t.Fatalf("prospective land must be handled: %+v", res)
	}
	return res, tracker.receipt()
}

// A peer landing in the CAS gap whose commit touches NOTHING inside the
// verification's declared dependency closure must not re-run the verification:
// the prior verdict still describes the candidate.
func TestLandProspectiveVerifyReusedWhenTrunkDeltaLeavesClosureUntouched(t *testing.T) {
	ran := 0
	res, receipt := runProspectiveReuseLand(t,
		prospectiveReuseFake("docs/notes.md\ndocs/GO-RUN.md\n"),
		[]string{"pkg/calc/**", "internal/workerworktree/**"}, &ran)
	if !res.OK || !res.Committed || !res.Applied {
		t.Fatalf("retry after a lost CAS must land: %+v", res)
	}
	if ran != 1 {
		t.Fatalf("a disjoint trunk delta must reuse the verdict, but the prospective verifier ran %d times", ran)
	}
	if n := countPhase(receipt, "prospective-commit-verification"); n != 1 {
		t.Fatalf("cost receipt must report exactly ONE prospective-commit-verification phase on a disjoint retry, got %d: %+v", n, receipt.Phases)
	}
}

// The fail-closed half: when the peer's trunk delta DOES touch the verification's
// dependency closure, the candidate is a different program and must be re-verified.
func TestLandProspectiveVerifyRerunsWhenTrunkDeltaTouchesClosure(t *testing.T) {
	ran := 0
	res, receipt := runProspectiveReuseLand(t,
		prospectiveReuseFake("pkg/calc/calc.go\n"),
		[]string{"pkg/calc/**", "internal/workerworktree/**"}, &ran)
	if !res.OK || !res.Committed {
		t.Fatalf("retry after a lost CAS must land: %+v", res)
	}
	if ran != 2 {
		t.Fatalf("a trunk delta inside the closure must re-verify, but the prospective verifier ran %d times", ran)
	}
	if n := countPhase(receipt, "prospective-commit-verification"); n != 2 {
		t.Fatalf("cost receipt must report TWO prospective-commit-verification phases, got %d: %+v", n, receipt.Phases)
	}
}

// No declared closure means no disjointness was ever proven, so the only sound
// answer is to re-verify. A moved base alone is never grounds to skip a check.
func TestLandProspectiveVerifyRerunsWithoutDeclaredClosure(t *testing.T) {
	ran := 0
	res, _ := runProspectiveReuseLand(t,
		prospectiveReuseFake("docs/notes.md\n"), nil, &ran)
	if !res.OK || !res.Committed {
		t.Fatalf("retry after a lost CAS must land: %+v", res)
	}
	if ran != 2 {
		t.Fatalf("an undeclared closure must re-verify, but the prospective verifier ran %d times", ran)
	}
}

// An unreadable trunk delta cannot prove disjointness either.
func TestLandProspectiveVerifyRerunsWhenTrunkDeltaUnreadable(t *testing.T) {
	ran := 0
	g := prospectiveReuseFake("")
	g.replyOnce("diff-tree", 1, "fatal: bad revision")
	res, _ := runProspectiveReuseLand(t, g, []string{"pkg/calc/**"}, &ran)
	if !res.OK || !res.Committed {
		t.Fatalf("retry after a lost CAS must land: %+v", res)
	}
	if ran != 2 {
		t.Fatalf("an unreadable trunk delta must re-verify, but the prospective verifier ran %d times", ran)
	}
}

// The reuse is disclosed, never silent: the receipt says the verdict was reused.
func TestLandProspectiveVerifyReuseIsDisclosed(t *testing.T) {
	ran := 0
	_, receipt := runProspectiveReuseLand(t,
		prospectiveReuseFake("docs/notes.md\n"), []string{"pkg/calc/**"}, &ran)
	if ran != 1 {
		t.Fatalf("precondition: the disjoint retry must reuse the verdict, ran %d", ran)
	}
	if !receipt.Reused {
		t.Fatalf("a reused verdict must be disclosed in the cost receipt: %+v", receipt)
	}
	if !strings.Contains(receipt.CacheState, "prospective") {
		t.Fatalf("cache state must name what was reused: %+v", receipt)
	}
}

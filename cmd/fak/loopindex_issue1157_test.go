package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopgate"
	"github.com/anthony-chaudhary/fak/internal/loopindex"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestLoopIndexIssue1157FalseDoneRefusedDefault(t *testing.T) {
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	verify := rep.StageDetail[3]
	if verify.Name != loopindex.StageVerify {
		t.Fatalf("stage[3] = %q, want verify", verify.Name)
	}

	probes := map[string]bool{}
	for _, p := range verify.Probes {
		probes[p.Name] = p.Pass
	}
	for _, name := range []string{
		"stop_seam_mechanism",
		"false_done_refused_default",
		"review_residual_surface",
	} {
		if !probes[name] {
			t.Fatalf("verify probe %s = false; issue #1157 is not witnessed by the tree", name)
		}
	}

	var verifyKPI loopindex.KPI
	for _, k := range rep.KPIs {
		if k.Name == loopindex.StageVerify {
			verifyKPI = k
			break
		}
	}
	if !verifyKPI.Wired || verifyKPI.Debt != 0 {
		t.Fatalf("verify KPI = %+v, want wired with no debt for #1157", verifyKPI)
	}
}

func TestLoopDriveIssue1157UnwitnessedDoneIsRefusedAtStopSeam(t *testing.T) {
	got := loopDriveWitnessFromGate(loopgate.Decision{
		Verdict: loopgate.VerdictNotYet,
		Reason:  loopgate.ReasonDoneUnwitnessed,
		Summary: "commit claim was not witnessed by its diff",
		Request: loopgate.Request{Kind: loopgate.CriterionCommitAudit, Ref: "HEAD"},
		Witness: "subject-only",
	})
	if got.Status != loopmgr.StatusWitnessRefused || got.ExitCode != 1 {
		t.Fatalf("witness result = %+v, want witness_refused exit 1", got)
	}
	if got.Reason != loopgate.ReasonDoneUnwitnessed {
		t.Fatalf("reason = %q, want %q", got.Reason, loopgate.ReasonDoneUnwitnessed)
	}
}

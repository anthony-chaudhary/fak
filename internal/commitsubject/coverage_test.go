package commitsubject

import "testing"

var sampleSubjects = []string{
	"fix(tools): add a noun-form rung to the closure auditor (fak tools)",
	"feat(model): MiniMax-M3 witness oracle (fak model)",
	"Merge branch 'main' of origin",
	"v0.32.0: cut the release",
	"ci: add the dispatch tool-test cluster to the gate (fak ci)",
}

func TestIsExempt(t *testing.T) {
	for _, s := range []string{"Merge branch 'main'", "Revert \"feat: x\"", "v1.2.3: ship it"} {
		if !IsExempt(s) {
			t.Fatalf("%q should be exempt", s)
		}
	}
	if IsExempt("fix(tools): add a rung") {
		t.Fatal("real ship subject marked exempt")
	}
}

func TestFoldCountsAndFraction(t *testing.T) {
	cov := Fold(sampleSubjects)
	if cov.Total != 3 || cov.Gradeable != 2 || cov.Abstain != 1 {
		t.Fatalf("coverage=%+v", cov)
	}
	if cov.Coverage == nil || *cov.Coverage < 0.666 || *cov.Coverage > 0.667 {
		t.Fatalf("coverage fraction=%v", cov.Coverage)
	}
	if len(cov.AbstainSubjects) != 1 || cov.AbstainSubjects[0].Subject == "" {
		t.Fatalf("abstains=%+v", cov.AbstainSubjects)
	}
}

func TestFoldEmptyWindow(t *testing.T) {
	cov := Fold([]string{"Merge x", "v1.0.0: bump"})
	if cov.Total != 0 || cov.Coverage != nil {
		t.Fatalf("coverage=%+v", cov)
	}
}

func TestFoldReusesCommitMsgVerbSet(t *testing.T) {
	cov := Fold([]string{"feat(tools): thing without a verb"})
	if cov.Abstain != 1 {
		t.Fatalf("noun-led subject should abstain under commit-msg verb set: %+v", cov)
	}
}

func TestBuildPayload(t *testing.T) {
	floor80 := 80.0
	p := BuildPayload("r", Fold(sampleSubjects), &floor80)
	if p.OK || p.Verdict != "BELOW_FLOOR" {
		t.Fatalf("payload=%+v", p)
	}
	floor50 := 50.0
	p = BuildPayload("r", Fold(sampleSubjects), &floor50)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("payload=%+v", p)
	}
	p = BuildPayload("r", Fold(sampleSubjects), nil)
	if !p.OK || p.CoveragePct == nil || *p.CoveragePct != 66.7 {
		t.Fatalf("payload=%+v", p)
	}
	p = BuildPayload("r", Fold([]string{"Merge x"}), &floor80)
	if !p.OK || p.Verdict != "NO_GRADEABLE_COMMITS" {
		t.Fatalf("payload=%+v", p)
	}
}

func TestCollectUsesInjectedFetcher(t *testing.T) {
	floor := 80.0
	p := Collect("r", 5, &floor, func(root string, n int) []string { return sampleSubjects })
	if p.Verdict != "BELOW_FLOOR" || p.Total != 3 {
		t.Fatalf("payload=%+v", p)
	}
}

package disambiguation

import "testing"

func TestRunGoSourceSelfTest(t *testing.T) {
	report, err := RunGoSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Candidates) != 3 || !report.Deterministic || !report.TestsExcluded || !report.GeneratedExcluded || !report.UnexportedExcluded {
		t.Fatalf("report=%#v", report)
	}
	got := map[string]GoCandidateKind{}
	for _, candidate := range report.Candidates {
		got[candidate.Name] = candidate.Kind
	}
	if got["PublicType"] != GoCandidateSymbol || got["PublicFunc"] != GoCandidateSymbol || got["demo.v1"] != GoCandidateCapability {
		t.Fatalf("candidates=%#v", report.Candidates)
	}
	if _, ok := got["TestOnlyExport"]; ok {
		t.Fatal("test symbol included")
	}
	if _, ok := got["GeneratedExport"]; ok {
		t.Fatal("generated symbol included")
	}
	if _, ok := got["hidden"]; ok {
		t.Fatal("unexported symbol included")
	}
}

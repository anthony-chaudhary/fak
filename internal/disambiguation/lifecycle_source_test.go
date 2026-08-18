package disambiguation

import "testing"

func TestRunLifecycleSourceSelfTest(t *testing.T) {
	report, err := RunLifecycleSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Ladders) != 3 || !report.IncompatibleSpellingsRejected {
		t.Fatalf("report=%#v", report)
	}
	for _, ladder := range report.Ladders {
		if ladder.CanonicalTerm == "" || ladder.SourcePath == "" || len(ladder.Spellings) == 0 {
			t.Errorf("ladder=%#v", ladder)
		}
	}
}

func TestResolveLadderSpellingPreservesDomain(t *testing.T) {
	got, err := ResolveLadderSpelling(LadderActivation, "shadow")
	if err != nil {
		t.Fatal(err)
	}
	if got.Entry.Identity.CanonicalTerm != "activation posture" {
		t.Fatalf("term=%q", got.Entry.Identity.CanonicalTerm)
	}
}

package disambiguation

import "testing"

func TestRunSessionSourceSelfTestResolvesTermsAndRejectsConflation(t *testing.T) {
	report, err := RunSessionSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != SessionSourceSelfTestSchemaVersion || report.IndexVersion != PublicIndexVersion {
		t.Fatalf("versions = %#v", report)
	}
	if len(report.Resolutions) != 5 {
		t.Fatalf("resolutions = %d, want 5", len(report.Resolutions))
	}
	want := map[string]string{"session": "agent session", "resume": "session resume", "recovery": "session recovery", "compaction": "context compaction", "checkpoint": "recovery checkpoint"}
	for _, resolution := range report.Resolutions {
		if resolution.CanonicalTerm != want[resolution.Input] {
			t.Errorf("%s resolved to %q, want %q", resolution.Input, resolution.CanonicalTerm, want[resolution.Input])
		}
		if resolution.SourcePath == "" {
			t.Errorf("%s has no source", resolution.Input)
		}
	}
	if !report.ResumeRecoveryConflation || !report.CompactionCheckpointConflation {
		t.Fatalf("conflation proof = %#v", report)
	}
}

func TestSessionSourceRequiredContrastsExplainDifferentMechanisms(t *testing.T) {
	pairs := [][2]string{{"session resume", "session recovery"}, {"context compaction", "recovery checkpoint"}}
	for _, pair := range pairs {
		ok, err := requiredForbiddenPair(pair[0], pair[1])
		if err != nil || !ok {
			t.Errorf("pair %q/%q: ok=%t err=%v", pair[0], pair[1], ok, err)
		}
	}
}

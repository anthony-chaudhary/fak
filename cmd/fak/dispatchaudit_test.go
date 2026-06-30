package main

import "testing"

func TestDispatchAuditIssueLabelsMarkTriageOnly(t *testing.T) {
	got := dispatchAuditIssueLabels()
	want := []string{"dispatch", "observability", "needs-triage", "triage-only"}
	if len(got) != len(want) {
		t.Fatalf("labels = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("labels = %+v, want %+v", got, want)
		}
	}
}

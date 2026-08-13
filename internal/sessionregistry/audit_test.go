package sessionregistry

import "testing"

func TestAuditAllRegisteredNestedGraph(t *testing.T) {
	rows := []Record{
		{RegistrationID: "root", RootRegistrationID: "root", AttemptID: "a", Identity: Identity{SessionID: "s-root"}},
		{RegistrationID: "child", ParentRegistrationID: "root", RootRegistrationID: "root", AttemptID: "b", Identity: Identity{SessionID: "s-child"}},
		{RegistrationID: "micro", ParentRegistrationID: "child", RootRegistrationID: "root", AttemptID: "c", Identity: Identity{SessionID: "s-micro"}},
	}
	got := Audit(rows, []string{"s-root", "s-child", "s-micro", "s-child"})
	if !got.Complete || got.Coverage != 1 || got.RegisteredObserved != 3 || len(got.Issues) != 0 {
		t.Fatalf("audit=%+v", got)
	}
}

func TestAuditRefusesIncompleteAmbiguousAndBrokenLineage(t *testing.T) {
	rows := []Record{
		{RegistrationID: "root-a", RootRegistrationID: "root-a", Identity: Identity{SessionID: "shared"}},
		{RegistrationID: "root-b", RootRegistrationID: "root-b", Identity: Identity{SessionID: "shared"}},
		{RegistrationID: "orphan", ParentRegistrationID: "missing-parent", RootRegistrationID: "missing-root", Identity: Identity{SessionID: "known"}},
		{RegistrationID: "cycle-a", ParentRegistrationID: "cycle-b", RootRegistrationID: "root-a"},
		{RegistrationID: "cycle-b", ParentRegistrationID: "cycle-a", RootRegistrationID: "root-a"},
	}
	got := Audit(rows, []string{"shared", "known", "absent"})
	if got.Complete || got.Coverage != 1.0/3.0 || got.RegisteredObserved != 1 || got.Unregistered != 1 || got.Ambiguous != 1 {
		t.Fatalf("audit=%+v", got)
	}
	codes := map[string]bool{}
	for _, issue := range got.Issues {
		codes[issue.Code] = true
	}
	for _, code := range []string{"ambiguous_session_root", "missing_root", "orphan_parent", "parent_cycle", "unregistered_observed"} {
		if !codes[code] {
			t.Errorf("missing issue %s: %+v", code, got.Issues)
		}
	}
}

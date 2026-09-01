package wipinventory

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func TestJoinIssueSessionsConservesRootChildrenAndResumes(t *testing.T) {
	history := issueHistory("wip:v1:00000000000000000000000000000001", "owner/repo", 10437)
	bindings := sessionregistry.WIPBindingReport{Bindings: []sessionregistry.WIPExecutionBinding{{
		RootRegistrationID: "root", Issue: &sessionregistry.WIPIssueIdentity{Repository: "owner/repo", Number: 10437},
		RegistrationIDs: []string{"child", "resume", "root"}, AttemptIDs: []string{"a0", "a1", "a2"}, SessionIDs: []string{"s0", "s1", "s2"}, Status: sessionregistry.WIPBindingJoined,
	}}}
	got, err := JoinIssueSessions([]History{history}, bindings)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Units) != 1 || len(got.Debt) != 0 {
		t.Fatalf("join = %#v", got)
	}
	if len(got.Units[0].AttemptIDs) != 3 || got.Units[0].Status != IssueSessionJoined {
		t.Fatalf("unit = %#v", got.Units[0])
	}
}

func TestJoinIssueSessionsKeepsSimilarIssuesSeparate(t *testing.T) {
	histories := []History{
		issueHistory("wip:v1:00000000000000000000000000000001", "owner/repo", 10437),
		issueHistory("wip:v1:00000000000000000000000000000002", "owner/repo", 10438),
	}
	bindings := sessionregistry.WIPBindingReport{Bindings: []sessionregistry.WIPExecutionBinding{
		{RootRegistrationID: "one", Issue: &sessionregistry.WIPIssueIdentity{Repository: "owner/repo", Number: 10437}, Status: sessionregistry.WIPBindingJoined},
		{RootRegistrationID: "two", Issue: &sessionregistry.WIPIssueIdentity{Repository: "owner/repo", Number: 10438}, Status: sessionregistry.WIPBindingJoined},
	}}
	got, err := JoinIssueSessions(histories, bindings)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Units) != 2 || got.Units[0].UnitID == got.Units[1].UnitID {
		t.Fatalf("units = %#v", got.Units)
	}
}

func TestJoinIssueSessionsPreservesTypedDebt(t *testing.T) {
	history := issueHistory("wip:v1:00000000000000000000000000000001", "owner/repo", 10437)
	bindings := sessionregistry.WIPBindingReport{Bindings: []sessionregistry.WIPExecutionBinding{
		{RootRegistrationID: "legacy", RegistrationIDs: []string{"legacy"}, Status: sessionregistry.WIPBindingMissing, Details: []string{"lineage has no durable issue binding"}},
		{RootRegistrationID: "conflict", RegistrationIDs: []string{"conflict"}, Status: sessionregistry.WIPBindingConflicting, Details: []string{"lineage carries conflicting issue bindings"}},
		{RootRegistrationID: "stale", Issue: &sessionregistry.WIPIssueIdentity{Repository: "owner/repo", Number: 10437}, Status: sessionregistry.WIPBindingStale},
	}}
	got, err := JoinIssueSessions([]History{history}, bindings)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Debt) != 2 || got.Debt[0].Status != IssueSessionConflicting || got.Debt[1].Status != IssueSessionMissing {
		t.Fatalf("debt = %#v", got.Debt)
	}
	if len(got.Units) != 1 || got.Units[0].Status != IssueSessionStale {
		t.Fatalf("units = %#v", got.Units)
	}
}

func TestJoinIssueSessionsDoesNotSilentlyDeduplicateConflictingUnits(t *testing.T) {
	histories := []History{
		issueHistory("wip:v1:00000000000000000000000000000001", "owner/repo", 10437),
		issueHistory("wip:v1:00000000000000000000000000000002", "owner/repo", 10437),
	}
	got, err := JoinIssueSessions(histories, sessionregistry.WIPBindingReport{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Units) != 1 || got.Units[0].Status != IssueSessionAmbiguous || len(got.Units[0].Details) == 0 {
		t.Fatalf("units = %#v", got.Units)
	}
}

func issueHistory(id WIPUnitID, repository string, number int) History {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	return History{Schema: WIPUnitSchema, Transitions: []Transition{
		{Kind: TransitionCreate, Timestamp: now, Source: "test", Provenance: Provenance{Actor: "test", Mechanism: "fixture"}, Successors: []WIPUnitID{id}, Witness: "fixture:create"},
		{Kind: TransitionBind, Timestamp: now.Add(time.Second), Source: "test", Provenance: Provenance{Actor: "test", Mechanism: "fixture"}, Predecessors: []WIPUnitID{id}, Successors: []WIPUnitID{id}, References: []SurfaceReference{{Kind: SurfaceIssue, Issue: &IssueReference{Repository: repository, Number: number}}}, Witness: "fixture:bind"},
	}}
}

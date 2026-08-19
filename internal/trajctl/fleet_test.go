package trajctl

import "testing"

func TestFleetWorstFirstAndCountsSessions(t *testing.T) {
	s := State{Objectives: map[string]Objective{"a": {ID: "a", Statement: "healthy", Status: StatusActive}, "b": {ID: "b", Statement: "stalled", Status: StatusActive}, "bc": {ID: "bc", ParentID: "b", Statement: "child", Status: StatusActive}}, Scores: []ScoreRow{{ObjectiveID: "a", Method: CommitScorerMethod, Value: .8, Witness: W3, SessionID: "s1"}, {ObjectiveID: "b", Method: CommitScorerMethod, Value: .3, Witness: W3, SessionID: "s2"}, {ObjectiveID: "bc", Method: CommitScorerMethod, Value: .2, Witness: W3, SessionID: "s3"}}}
	r := s.Fleet()
	if len(r.Objectives) != 2 || r.Objectives[0].ObjectiveID != "b" || r.Objectives[0].Sessions != 2 || r.Objectives[0].OpenDescendants != 1 {
		t.Fatalf("fleet=%+v", r)
	}
}

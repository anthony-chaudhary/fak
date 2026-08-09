package commitlifecycle

import (
	"reflect"
	"strings"
	"testing"
)

func TestFoldCoversLifecycle(t *testing.T) {
	tests := []struct {
		name  string
		facts Facts
		state State
		tool  string
		args  []string
		gated bool
	}{
		{"editing", Facts{DirtyPaths: []string{"a.go"}}, Editing, "", nil, true},
		{"commit ready", Facts{DirtyPaths: []string{"a.go"}, CommitArgs: []string{"--path", "a.go", "-m", "fix(x): y (#1) (fak x)"}}, CommitReady, "fak", []string{"commit", "--path", "a.go", "-m", "fix(x): y (#1) (fak x)"}, false},
		{"committed", Facts{LocalCommit: "abc"}, CommittedUnpushed, "fak", []string{"sync", "push"}, false},
		{"parked", Facts{Checkpoint: "session-1", CheckpointLive: true}, Parked, "", nil, true},
		{"reclaim", Facts{Checkpoint: "session-1"}, Reclaim, "fak", []string{"wip", "reconcile", "--reclaim"}, false},
		{"checkpoint land", Facts{Checkpoint: "session-1", CheckpointApply: true}, LandReady, "fak", []string{"wip", "land", "session-1", "--apply"}, false},
		{"worker land", Facts{WorkerPath: `C:\\worker`, WorkerLandReady: true}, LandReady, "fak", []string{"worktree", "worker", "land", "--path", `C:\\worker`}, false},
		{"landed", Facts{LandedCommit: "def"}, LandedUnpushed, "fak", []string{"sync", "push"}, false},
		{"local shipped", Facts{LocalCommit: "abc", LocalOnRemote: true}, Shipped, "", nil, false},
		{"landed shipped", Facts{LandedCommit: "def", LandedOnRemote: true}, Shipped, "", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Fold(tt.facts)
			if got.State != tt.state || got.Action.Tool != tt.tool || got.Action.NeedsOperator != tt.gated || !reflect.DeepEqual(got.Action.Args, tt.args) {
				t.Fatalf("Fold(%+v) = %+v, want state=%s tool=%q args=%q gated=%v", tt.facts, got, tt.state, tt.tool, tt.args, tt.gated)
			}
			if got.State != Shipped && got.Action.Tool == "" && !got.Action.NeedsOperator {
				t.Fatalf("non-terminal row has neither argv nor operator gate: %+v", got)
			}
			if got.State == Shipped && (got.Action.Tool != "" || got.Action.NeedsOperator) {
				t.Fatalf("SHIPPED must be terminal: %+v", got)
			}
		})
	}
}

func TestFoldContradictionsFailClosed(t *testing.T) {
	for name, facts := range map[string]Facts{
		"remote without local":  {LocalOnRemote: true},
		"remote without landed": {LandedOnRemote: true},
		"checkpoint without id": {CheckpointApply: true},
		"worker without path":   {WorkerLandReady: true},
		"contract without dirt": {CommitArgs: []string{"--path", "a.go"}},
		"live checkpoint apply": {Checkpoint: "s", CheckpointLive: true, CheckpointApply: true},
	} {
		t.Run(name, func(t *testing.T) {
			got := Fold(facts)
			if got.State != Unknown || !got.Action.NeedsOperator || got.Action.Reason == "" {
				t.Fatalf("contradiction did not fail closed: %+v", got)
			}
		})
	}
}

func TestActionsUseOnlySafeExistingEntrypoints(t *testing.T) {
	facts := []Facts{
		{DirtyPaths: []string{"a"}, CommitArgs: []string{"--path", "a", "-m", "m"}},
		{Checkpoint: "s"},
		{Checkpoint: "s", CheckpointApply: true},
		{WorkerPath: "worker", WorkerLandReady: true},
		{LocalCommit: "abc"},
		{LandedCommit: "def"},
	}
	for _, f := range facts {
		a := Fold(f).Action
		joined := strings.ToLower(a.Tool + " " + strings.Join(a.Args, " "))
		for _, forbidden := range []string{"add -a", "add -a", "--force", "worktree add"} {
			if strings.Contains(joined, forbidden) {
				t.Fatalf("unsafe action %q", joined)
			}
		}
		if a.Tool == "fak" {
			if len(a.Args) == 0 || (a.Args[0] != "commit" && a.Args[0] != "wip" && a.Args[0] != "worktree" && a.Args[0] != "sync") {
				t.Fatalf("non-allowlisted fak action: %q", joined)
			}
		} else {
			t.Fatalf("non-allowlisted action: %q", joined)
		}
	}
}

package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchWaveFinishesAttemptedIssueBeforeFreshIssue captures the full default-on spine:
// a durable resolve sidecar is folded into runsSnapshot.latest, carried through wave pricing,
// and causes already-started work to be selected before a never-attempted peer.
func TestDispatchWaveFinishesAttemptedIssueBeforeFreshIssue(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runs, 0o755); err != nil {
		t.Fatal(err)
	}
	attempt := filepath.Join(runs, "resolve-10-previous.witness")
	if err := os.WriteFile(attempt, []byte("finished\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	attemptedAt := time.Now().Add(-3 * time.Hour).Truncate(time.Second)
	if err := os.Chtimes(attempt, attemptedAt, attemptedAt); err != nil {
		t.Fatal(err)
	}

	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"old-wip": {Count: 1, Issues: []int{10}, Tree: []string{"internal/old"}},
			"fresh":   {Count: 1, Issues: []int{20}, Tree: []string{"internal/fresh"}},
		},
	}
	price, err := priceDispatchWavePayload(root, router, 1, 1, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 1 {
		t.Fatalf("run targets = %+v, want exactly one", price.RunTargets)
	}
	if got := price.RunTargets[0].Issue; got != 10 {
		t.Fatalf("selected issue = %d, want attempted issue 10 before fresh issue 20; candidates=%+v", got, price.Candidates)
	}
}

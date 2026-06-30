package main

import (
	"encoding/json"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchTickPicksPriorityOverNewer pins #1395: within the chosen lane the
// tick picks the heaviest priority/P* issue even when a newer (higher-numbered)
// unlabeled issue exists, instead of pure recency. The fixture's cmd lane carries
// a newer unlabeled #1500 and an older #300 that holds priority/P1; P1 wins.
func TestDispatchTickPicksPriorityOverNewer(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	// Disable the guard so this test isolates the priority pick-ORDER (#1395) from the
	// #1397 self-modify hold (cmd/** is fak's own source, so a GUARDED tick would hold).
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"cmd": {
					Tree:     []string{"cmd/**"},
					Issues:   []int{300, 1500},
					Count:    2,
					Priority: map[int]int{300: dispatchtick.PriorityWeightP1},
				},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	root := t.TempDir()
	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["target_issue"] != float64(300) {
		t.Fatalf("priority target = %v, want 300 (priority/P1 beats newer #1500)", got["target_issue"])
	}
}

func TestPickDispatchLaneUsesStepBudgetBeforeIssueCount(t *testing.T) {
	old := dispatchRouteIssues
	// Both lanes are NON-self-source (docs/**, tools/**) so this test exercises the real
	// default-GUARDED auto-pick and isolates the step-budget>issue-count ordering (#1395)
	// without the #1397 self-source skip firing -- the prior `gateway` lane (internal/**)
	// is now skipped under guard, which is a DIFFERENT property; tools/** is shippable
	// under guard, so the ordering assertion (larger step budget wins) is what's tested.
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"docs": {
					Tree:       []string{"docs/**"},
					Issues:     []int{10, 11, 12},
					Count:      3,
					StepBudget: 3,
				},
				"tools": {
					Tree:       []string{"tools/**"},
					Issues:     []int{20, 21},
					Count:      2,
					StepBudget: 9,
				},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	pick, err := pickDispatchLane(t.TempDir(), io.Discard, "", nil, false)
	if err != nil {
		t.Fatalf("pickDispatchLane: %v", err)
	}
	if pick.Lane != "tools" {
		t.Fatalf("lane = %q, want tools because it has the larger step budget", pick.Lane)
	}
	if pick.ByLaneStepBudget["tools"] != 9 || pick.ByLaneStepBudget["docs"] != 3 {
		t.Fatalf("step budgets = %+v, want tools=9 docs=3", pick.ByLaneStepBudget)
	}
	if len(pick.Numbers) != 2 || pick.Numbers[0] != 20 || pick.Numbers[1] != 21 {
		t.Fatalf("picked numbers = %+v, want tools issues ordered oldest-first", pick.Numbers)
	}
}

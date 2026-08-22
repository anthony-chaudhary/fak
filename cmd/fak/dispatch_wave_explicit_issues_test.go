package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestDispatchWaveExplicitIssuesAreTheOnlyCandidates(t *testing.T) {
	requested := []int{8203, 8204, 8205, 8206}
	router := explicitIssueRouter(append(append([]int(nil), requested...), 7824))
	router.Lanes["lane-7824"] = dispatchtick.RouterLaneGroup{
		Tree:       []string{"internal/unrelated/**"},
		Issues:     []int{7824},
		Count:      1,
		StepBudget: 10_000,
		Priority:   map[int]int{7824: dispatchtick.PriorityWeightP0},
	}

	price, err := priceDispatchWaveExplicitIssues(t.TempDir(), router, requested, 4, 4, "", nil, 0, nil, 4, nil, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(price.RequestedIssues, requested) {
		t.Fatalf("requested issues = %v, want %v", price.RequestedIssues, requested)
	}
	if !reflect.DeepEqual(sortedInts(price.SelectedIssues), requested) {
		t.Fatalf("selected issues = %v, want exactly %v", price.SelectedIssues, requested)
	}
	if len(price.RefusedIssues) != 0 {
		t.Fatalf("refused issues = %+v, want none", price.RefusedIssues)
	}
	for _, cand := range price.Candidates {
		if cand.Issue == 7824 {
			t.Fatalf("unrelated backlog issue substituted into explicit set: %+v", cand)
		}
		if !containsInt(requested, cand.Issue) {
			t.Fatalf("candidate issue %d is outside requested set %v", cand.Issue, requested)
		}
	}
}

func TestDispatchWaveExplicitIssuesReturnTypedRefusalsWithoutSubstitution(t *testing.T) {
	requested := []int{100, 200, 300, 400, 500}
	router := explicitIssueRouter([]int{100, 200, 300, 500, 999})
	intentHolds := map[int]string{300: "claimed by peer session"}

	price, err := priceDispatchWaveExplicitIssues(t.TempDir(), router, requested, 1, 1, "", nil, 0, map[int]bool{500: true}, 4, intentHolds, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.SelectedIssues) != 1 || !containsInt([]int{100, 200}, price.SelectedIssues[0]) {
		t.Fatalf("selected issues = %v, want one eligible requested issue", price.SelectedIssues)
	}
	if containsInt(price.SelectedIssues, 999) {
		t.Fatalf("unrelated issue 999 substituted into selected set %v", price.SelectedIssues)
	}
	assertIssueRefusal(t, price.RefusedIssues, 300, dispatchWaveIssueRefusalIntent, leaseref.ReasonIntentCollision)
	assertIssueRefusal(t, price.RefusedIssues, 400, dispatchWaveIssueRefusalRouting, dispatchWaveReasonIssueUnroutable)
	assertIssueRefusal(t, price.RefusedIssues, 500, dispatchWaveIssueRefusalEligibility, dispatchWaveReasonIssueIneligible)
	capacityIssue := 100
	if price.SelectedIssues[0] == 100 {
		capacityIssue = 200
	}
	assertIssueRefusal(t, price.RefusedIssues, capacityIssue, dispatchWaveIssueRefusalCapacity, dispatchWaveReasonCapacity)
}

func TestDispatchWaveExplicitIssuesFoldPrelaunchRefusal(t *testing.T) {
	requested := []int{8203, 8204}
	price, err := priceDispatchWaveExplicitIssues(t.TempDir(), explicitIssueRouter(requested), requested, 2, 2, "", nil, 0, nil, 2, nil, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 2 {
		t.Fatalf("run targets = %+v, want two before prelaunch audit", price.RunTargets)
	}
	price = dispatchWaveApplyAuditIssueOutcomes(price, []dispatchWaveExecutionAudit{
		{Rank: 0, OK: true, Target: dispatchWaveLaunchTarget(price.RunTargets[0])},
		{Rank: 1, OK: false, Verdict: leaseref.ReasonIntentCollision, Reason: "claimed by peer", Target: dispatchWaveLaunchTarget(price.RunTargets[1])},
	})
	if !reflect.DeepEqual(price.SelectedIssues, []int{price.RunTargets[0].Issue}) {
		t.Fatalf("selected issues after audit = %v, run targets=%+v", price.SelectedIssues, price.RunTargets)
	}
	refusedIssue := 8203
	if price.SelectedIssues[0] == 8203 {
		refusedIssue = 8204
	}
	assertIssueRefusal(t, price.RefusedIssues, refusedIssue, dispatchWaveIssueRefusalIntent, leaseref.ReasonIntentCollision)
}

func TestDispatchWaveExplicitIssueJSONReceiptRoundTripsIdentities(t *testing.T) {
	requested := []int{8203, 9999}
	router := explicitIssueRouter([]int{8203, 7824})
	price, err := priceDispatchWaveExplicitIssues(t.TempDir(), router, requested, 2, 2, "", nil, 0, nil, 2, nil, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	rec := newDispatchWaveRecord(t.TempDir(), false, "codex", "engineering", "", dispatchGoalProfileThroughput, 2, 2, map[string]any{})
	rec["granted"] = 2
	rec["price"] = price
	rec["ok"] = true
	rec["prelaunch_gate"] = dispatchWavePrelaunchGate{OK: true, Action: "LAUNCH", TargetCount: 1, ReadyCount: 1}
	dispatchWaveAttachExplicitIssueReceipt(rec, price)

	var stdout, stderr bytes.Buffer
	if code := writeDispatchWaveResult(&stdout, &stderr, rec, true); code != 0 {
		t.Fatalf("write receipt code=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Requested []int                      `json:"requested_issues"`
		Selected  []int                      `json:"selected_issues"`
		Refused   []dispatchWaveIssueRefusal `json:"refused_issues"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v\n%s", err, stdout.String())
	}
	if !reflect.DeepEqual(got.Requested, requested) || !reflect.DeepEqual(got.Selected, []int{8203}) {
		t.Fatalf("receipt identities requested=%v selected=%v, want %v/[8203]", got.Requested, got.Selected, requested)
	}
	assertIssueRefusal(t, got.Refused, 9999, dispatchWaveIssueRefusalRouting, dispatchWaveReasonIssueUnroutable)
}

func TestParseDispatchWaveIssueNumbers(t *testing.T) {
	got, err := parseDispatchWaveIssueNumbers([]string{"#8203", "8204, issue-8205", "8203"})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{8203, 8204, 8205}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("issues = %v, want %v", got, want)
	}
	if _, err := parseDispatchWaveIssueNumbers([]string{"not-an-issue"}); err == nil {
		t.Fatal("invalid explicit issue must fail parsing")
	}
}

func TestDispatchWaveReadsLiveIntentHolds(t *testing.T) {
	root := t.TempDir()
	if out, err := exec.Command("git", "-C", root, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	store := leaseref.NewInDir(root)
	if _, verdict, err := store.ClaimIntent(context.Background(), leaseref.IntentRecord{
		Target: "issue #8205", Holder: "peer-worker", TTLSeconds: 600,
	}, time.Now()); err != nil || !verdict.OK {
		t.Fatalf("claim intent: err=%v verdict=%+v", err, verdict)
	}

	holds, err := dispatchWaveReadIntentHolds(root, []int{8205, 8206})
	if err != nil {
		t.Fatal(err)
	}
	if holds[8205] == "" || holds[8206] != "" {
		t.Fatalf("intent holds = %v, want only issue 8205", holds)
	}
}

func explicitIssueRouter(issues []int) dispatchtick.RouterPayload {
	router := dispatchtick.RouterPayload{Lanes: map[string]dispatchtick.RouterLaneGroup{}}
	for _, issue := range issues {
		lane := "lane-" + strconv.Itoa(issue)
		path := "internal/lane" + strconv.Itoa(issue) + "/work.go"
		router.Issues = append(router.Issues, dispatchtick.IssueRoute{
			Number: issue, Lane: lane, Paths: []string{path}, ExpectedSteps: issue,
		})
		router.Lanes[lane] = dispatchtick.RouterLaneGroup{
			Tree:       []string{path},
			Issues:     []int{issue},
			Count:      1,
			StepBudget: issue,
			Priority:   map[int]int{issue: dispatchtick.PriorityWeightDefault},
		}
	}
	return router
}

func assertIssueRefusal(t *testing.T, rows []dispatchWaveIssueRefusal, issue int, class, reason string) {
	t.Helper()
	for _, row := range rows {
		if row.Issue == issue {
			if row.Class != class || row.Reason != reason {
				t.Fatalf("issue %d refusal = %+v, want class=%q reason=%q", issue, row, class, reason)
			}
			return
		}
	}
	t.Fatalf("no refusal for issue %d in %+v", issue, rows)
}

func containsInt(xs []int, want int) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func sortedInts(xs []int) []int {
	out := append([]int(nil), xs...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

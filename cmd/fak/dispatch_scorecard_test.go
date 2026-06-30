package main

import (
	"encoding/json"
	"io"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchScorecardJSONWitnessesPlanningFloor(t *testing.T) {
	out, errb, code := runDispatchAt("scorecard", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchPlanScorecard
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if !got.OK || got.Verdict != "OK" || got.Corpus.DispatchPlanDebt != 0 || got.Corpus.Score != 100 {
		t.Fatalf("scorecard header = %+v, want OK debt=0 score=100", got)
	}
	if got.Live != nil {
		t.Fatalf("default scorecard should not include live telemetry: %+v", got.Live)
	}
	probes := map[string]bool{}
	for _, k := range got.KPIs {
		probes[k.Name] = k.Pass
	}
	for _, name := range []string{
		"collision_price_serializes",
		"price_wave_schedule",
		"scoped_parallelism_gain",
		"wave_execution_plan",
		"prelaunch_audit_gate",
		"repartition_advice",
		"issue_scoped_same_lane_parallelism",
		"router_carries_path_scope",
		"reactive_lease_overlap_floor",
	} {
		if !probes[name] {
			t.Fatalf("probe %s missing or failed in %+v", name, got.KPIs)
		}
	}
}

func sameStringsUnordered(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	left := append([]string(nil), a...)
	right := append([]string(nil), b...)
	sort.Strings(left)
	sort.Strings(right)
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func TestDispatchScorecardLiveRouterTelemetryIsNonGating(t *testing.T) {
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema:  dispatchtick.RouterSchema,
			OK:      true,
			Verdict: "OK",
			Reason:  "stubbed live router",
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"gateway": {
					Tree:   []string{"internal/gateway/**"},
					Issues: []int{40, 41},
					Count:  2,
				},
			},
			Counts: dispatchtick.RouterCounts{Open: 2, Routed: 2, ByConfidence: map[string]int{"path-confirmed": 2}},
			Issues: []dispatchtick.IssueRoute{
				{Number: 40, Title: "gateway http", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/http.go"}},
				{Number: 41, Title: "gateway mcp", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/mcp.go"}},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })

	out, errb, code := runDispatchAt("scorecard", "--workspace", t.TempDir(), "--live-router", "--count", "2", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchPlanScorecard
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if !got.OK || got.Corpus.DispatchPlanDebt != 0 {
		t.Fatalf("live telemetry should not add deterministic scorecard debt: %+v", got)
	}
	if got.Live == nil || !got.Live.OK || got.Live.Verdict != "OK" {
		t.Fatalf("live telemetry = %+v, want OK block", got.Live)
	}
	if got.Live.CandidateCount != 2 || got.Live.ScopeCoveragePct != 100 ||
		got.Live.SafeConcurrencyPct != 100 || got.Live.SameLaneParallelism != 1 {
		t.Fatalf("live metrics = %+v, want two scoped disjoint same-lane targets", got.Live)
	}
	if got.Live.WaveCount != 1 || got.Live.LaneSerialWaveCount != 2 ||
		got.Live.ScopedParallelGain != 1 || got.Live.CollisionWavePenalty != 0 {
		t.Fatalf("live wave score = %+v, want one scoped wave beating two lane-serial waves", got.Live)
	}
	if len(got.Live.Waves) != 1 || !sameStringsUnordered(got.Live.Waves[0].Agents, []string{"gateway#40", "gateway#41"}) {
		t.Fatalf("live waves = %+v, want one same-lane scoped wave", got.Live.Waves)
	}
	if len(got.Live.LaunchPlan) != 1 || len(got.Live.LaunchPlan[0].Targets) != 2 {
		t.Fatalf("live launch plan = %+v, want one two-target scoped wave", got.Live.LaunchPlan)
	}
	leases := map[string]bool{}
	for _, target := range got.Live.LaunchPlan[0].Targets {
		if target.Lane != "gateway" || !target.Scoped || target.ScopeSource != "issue" ||
			target.LeaseID == "" || len(target.Tree) != 1 {
			t.Fatalf("live launch target = %+v, want resolved issue-scoped gateway target", target)
		}
		for _, want := range []string{"--lane", target.Lane, "--target-issue", "--lease-id", target.LeaseID, "--lease-tree", strings.Join(target.Tree, ",")} {
			if !containsString(target.TickArgs, want) {
				t.Fatalf("live dispatch_tick_args = %#v, missing %q for target %+v", target.TickArgs, want, target)
			}
		}
		if leases[target.LeaseID] {
			t.Fatalf("live launch plan reused lease id in %+v", got.Live.LaunchPlan[0].Targets)
		}
		leases[target.LeaseID] = true
	}
	if !strings.Contains(got.Live.NextAction, "same-lane targets") {
		t.Fatalf("live next action = %q, want same-lane launch guidance", got.Live.NextAction)
	}
}

func TestDispatchScorecardHumanNamesSameLaneParallelism(t *testing.T) {
	out, errb, code := runDispatchAt("scorecard")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	for _, want := range []string{"dispatch planning scorecard", "score=100", "issue_scoped_same_lane_parallelism", "reactive_lease_overlap_floor"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human scorecard missing %q:\n%s", want, out)
		}
	}
}

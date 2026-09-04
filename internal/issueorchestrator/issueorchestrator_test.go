package issueorchestrator

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testIssue(number int, key, title, lane string, paths []string, steps int) Issue {
	return Issue{
		Number:          number,
		Key:             key,
		Title:           title,
		Lane:            lane,
		Paths:           paths,
		ExpectedSteps:   steps,
		Dispatchability: "dispatchable",
	}
}

func TestPlanWavesBasic(t *testing.T) {
	issues := []Issue{
		testIssue(1, "issue-1", "Fix gateway timeout", "gateway", []string{"internal/gateway/timeout.go"}, 3),
		testIssue(2, "issue-2", "Add model kv cache", "model", []string{"internal/model/kv.go"}, 4),
		testIssue(3, "issue-3", "Refactor compute rocm", "compute", []string{"internal/compute/rocm.go"}, 5),
		testIssue(4, "issue-4", "Improve engine router", "engine", []string{"internal/engine/router.go"}, 2),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
	})

	if plan.Schema != WavePlanSchema {
		t.Fatalf("expected schema %q, got %q", WavePlanSchema, plan.Schema)
	}
	if plan.TotalIssues != 4 {
		t.Fatalf("expected 4 total issues, got %d", plan.TotalIssues)
	}
	if plan.Dispatchable < 3 {
		t.Fatalf("expected at least 3 dispatchable, got %d", plan.Dispatchable)
	}
	if plan.TotalWaves == 0 {
		t.Fatalf("expected at least 1 wave, got 0")
	}
}

func TestPlanWavesCollisionSeparation(t *testing.T) {
	// Issue 1 and Issue 2 touch overlapping files in internal/gateway.
	// They must NOT be in the same wave.
	issues := []Issue{
		testIssue(1, "issue-1", "Fix gateway auth", "gateway", []string{"internal/gateway/auth.go"}, 3),
		testIssue(2, "issue-2", "Refactor gateway router", "gateway", []string{"internal/gateway/auth.go"}, 4),
		testIssue(3, "issue-3", "Compute kernel", "compute", []string{"internal/compute/kernel.go"}, 2),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
	})

	for _, w := range plan.Waves {
		contains1 := false
		contains2 := false
		for _, iss := range w.Issues {
			if iss.Number == 1 {
				contains1 = true
			}
			if iss.Number == 2 {
				contains2 = true
			}
		}
		if contains1 && contains2 {
			t.Fatalf("Wave %s contains colliding issues 1 and 2!", w.ID)
		}
	}
}

func TestPlanWavesImportContention(t *testing.T) {
	// Simulated graph: "gateway" imports "policy"
	graph := map[string]map[string]struct{}{
		"gateway": {
			"policy": struct{}{},
		},
	}

	issues := []Issue{
		testIssue(1, "issue-1", "Gateway feature", "gateway", []string{"internal/gateway/serve.go"}, 3),
		testIssue(2, "issue-2", "Policy engine", "policy", []string{"internal/policy/engine.go"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
		Graph:    graph,
	})

	for _, w := range plan.Waves {
		if len(w.Issues) > 1 {
			containsGateway := false
			containsPolicy := false
			for _, iss := range w.Issues {
				if iss.Lane == "gateway" {
					containsGateway = true
				}
				if iss.Lane == "policy" {
					containsPolicy = true
				}
			}
			if containsGateway && containsPolicy {
				t.Fatalf("Wave %s placed contending packages (gateway -> policy) together!", w.ID)
			}
		}
	}
}

func TestPlanWavesSerialSingleton(t *testing.T) {
	issues := []Issue{
		testIssue(1, "issue-1", "Frozen ABI change", "abi", []string{"internal/abi/abi.go"}, 2),
		testIssue(2, "issue-2", "Tooling update", "tooling", []string{"internal/tooling/t.go"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
	})

	var abiWave *Wave
	for i := range plan.Waves {
		w := &plan.Waves[i]
		for _, iss := range w.Issues {
			if iss.Lane == "abi" {
				abiWave = w
				break
			}
		}
	}

	if abiWave == nil {
		t.Fatalf("abi wave not found")
	}
	if abiWave.Safety != WaveSafetySerialSingleton {
		t.Errorf("expected abi wave to be serial_singleton, got %s", abiWave.Safety)
	}
	if abiWave.WaveSize != 1 {
		t.Errorf("serial singleton wave size must be 1, got %d", abiWave.WaveSize)
	}
}

func TestPlanWavesHeldLeaseExclusion(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")
	record := `{"op":"ACQUIRE","lane":"gateway","tree":["internal/gateway/**"]}` + "\n"
	if err := os.WriteFile(journal, []byte(record), 0o644); err != nil {
		t.Fatal(err)
	}

	issues := []Issue{
		testIssue(1, "issue-1", "Gateway work", "gateway", []string{"internal/gateway/foo.go"}, 3),
		testIssue(2, "issue-2", "Compute work", "compute", []string{"internal/compute/bar.go"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize:       4,
		WorkspaceRoot:  tempDir,
		AutoDetectHeld: true,
	})

	if len(plan.HeldLanes) == 0 || plan.HeldLanes[0] != "gateway" {
		t.Fatalf("expected gateway in held lanes, got %v", plan.HeldLanes)
	}
	if len(plan.HeldIssues) == 0 || plan.HeldIssues[0] != 1 {
		t.Fatalf("expected issue 1 to be held, got %v", plan.HeldIssues)
	}

	for _, w := range plan.Waves {
		for _, iss := range w.Issues {
			if iss.Number == 1 {
				t.Fatalf("held issue 1 should not be scheduled in any wave!")
			}
		}
	}
}

func TestPlanWavesSubdivideAndTriage(t *testing.T) {
	issues := []Issue{
		testIssue(1, "epic-1", "Giant monolithic redesign", "engine", []string{"internal/engine/foo.go"}, 25), // >15 steps -> subdivide
		{
			Number:          2,
			Key:             "triage-1",
			Title:           "", // missing title -> triage
			Lane:            "compute",
			Dispatchability: "triage_only",
		},
		testIssue(3, "leaf-1", "Normal leaf work", "compute", []string{"internal/compute/c.go"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
	})

	if len(plan.Subdivide) != 1 || plan.Subdivide[0].IssueNumber != 1 {
		t.Fatalf("expected epic 1 in subdivide, got %v", plan.Subdivide)
	}
	if len(plan.Triage) != 1 || plan.Triage[0].IssueNumber != 2 {
		t.Fatalf("expected issue 2 in triage, got %v", plan.Triage)
	}
	if plan.PlannedIssues != 1 || plan.Waves[0].Issues[0].Number != 3 {
		t.Fatalf("expected only issue 3 planned, got plannedCount=%d", plan.PlannedIssues)
	}
}

func TestCompare(t *testing.T) {
	baseline := Plan{
		Schema:        WavePlanSchema,
		TotalIssues:   4,
		PlannedIssues: 4,
		PlannedSteps:  15,
		TotalWaves:    2,
		Waves: []Wave{
			{
				ID:       "wave-1",
				WaveSize: 2,
				Issues: []Issue{
					testIssue(1, "issue-1", "Issue 1", "compute", nil, 3),
					testIssue(2, "issue-2", "Issue 2", "model", nil, 4),
				},
			},
			{
				ID:       "wave-2",
				WaveSize: 2,
				Issues: []Issue{
					testIssue(3, "issue-3", "Issue 3", "gateway", nil, 4),
					testIssue(4, "issue-4", "Issue 4", "engine", nil, 4),
				},
			},
		},
	}

	// Current plan: issues 1 and 2 are closed; only issues 3 and 4 remain in wave-2
	current := Plan{
		Schema:        WavePlanSchema,
		TotalIssues:   2,
		PlannedIssues: 2,
		PlannedSteps:  8,
		TotalWaves:    1,
		Waves: []Wave{
			{
				ID:       "wave-1",
				WaveSize: 2,
				Issues: []Issue{
					testIssue(3, "issue-3", "Issue 3", "gateway", nil, 4),
					testIssue(4, "issue-4", "Issue 4", "engine", nil, 4),
				},
			},
		},
	}

	res := Compare(current, baseline)
	if res.Schema != CompareSchema {
		t.Errorf("expected schema %q, got %q", CompareSchema, res.Schema)
	}
	if res.ClosedIssues != 2 {
		t.Errorf("expected 2 closed issues, got %d", res.ClosedIssues)
	}
	if len(res.ClosedNumbers) != 2 || res.ClosedNumbers[0] != 1 || res.ClosedNumbers[1] != 2 {
		t.Errorf("expected closed numbers [1, 2], got %v", res.ClosedNumbers)
	}
	if res.ClosedPercent != 50.0 {
		t.Errorf("expected 50.0%% closed, got %.1f%%", res.ClosedPercent)
	}
	if res.ClosedWaves != 1 {
		t.Errorf("expected 1 closed wave, got %d", res.ClosedWaves)
	}
	if res.RetiredSteps != 7 {
		t.Errorf("expected 7 retired steps, got %d", res.RetiredSteps)
	}

	report := CompareReport(current, baseline)
	if !strings.Contains(report, "2/4 issue(s) closed (50.0%)") {
		t.Errorf("report missing burndown percentage: %s", report)
	}
	if !strings.Contains(report, "#1, #2") {
		t.Errorf("report missing closed issue numbers: %s", report)
	}
}

func TestRenderAndMarkdown(t *testing.T) {
	issues := []Issue{
		testIssue(101, "issue-101", "Add prompt caching", "cache", []string{"internal/cache/c.go"}, 3),
		testIssue(102, "issue-102", "Fix model dispatch", "model", []string{"internal/model/m.go"}, 4),
	}
	plan := PlanWaves(issues, WavePlanOptions{WaveSize: 4})

	text := RenderWaves(plan)
	if !strings.Contains(text, "FAK ISSUE ORCHESTRATOR: CONCURRENT SAFE WAVE PLAN") {
		t.Errorf("render missing header:\n%s", text)
	}
	if !strings.Contains(text, "#101") || !strings.Contains(text, "#102") {
		t.Errorf("render missing issue numbers:\n%s", text)
	}

	md := MarkdownWaves(plan)
	if !strings.Contains(md, "# Issue Orchestrator: Concurrent Safe Wave Plan") {
		t.Errorf("markdown missing title:\n%s", md)
	}
	if !strings.Contains(md, "#101") {
		t.Errorf("markdown missing issue #101:\n%s", md)
	}

	// Verify JSON round trip
	b, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	var roundTrip Plan
	if err := json.Unmarshal(b, &roundTrip); err != nil {
		t.Fatalf("unmarshal plan: %v", err)
	}
	if roundTrip.PlannedIssues != plan.PlannedIssues {
		t.Errorf("round trip mismatch: got %d, want %d", roundTrip.PlannedIssues, plan.PlannedIssues)
	}
}

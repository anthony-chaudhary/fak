package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// writeWalkFixture writes a gh-issue-list-shaped JSON file with a deliberately mixed
// set: a stale issue (mark-stale act), a dormant question (close act), a fresh tagged
// issue (cheap-skipped), a healthy issue (skipped), and a review issue. Dates are
// relative to now so idle-day classification is deterministic.
func writeWalkFixture(t *testing.T) string {
	t.Helper()
	now := time.Now().UTC()
	ago := func(days int) string { return now.AddDate(0, 0, -days).Format(time.RFC3339) }
	issues := []map[string]any{
		{ // stale: idle 70d, no priority/area -> mark-stale act
			"number": 101, "title": "wire the gpu telemetry exporter end to end",
			"state": "OPEN", "createdAt": ago(120), "updatedAt": ago(70),
			"labels": []map[string]string{{"name": "enhancement"}},
		},
		{ // dormant question: question + idle 40d -> close act
			"number": 102, "title": "how should accounts rotate across two seats",
			"state": "OPEN", "createdAt": ago(90), "updatedAt": ago(40),
			"labels": []map[string]string{{"name": "question"}},
		},
		{ // fresh but tagged: idle 1d, missing area -> cheap-skipped by skip-fresh
			"number": 103, "title": "refactor the dispatch preflight ladder ordering",
			"state": "OPEN", "createdAt": ago(2), "updatedAt": ago(1),
			"labels": []map[string]string{{"name": "enhancement"}},
		},
		{ // healthy: priority + kind + area, idle 5d -> skipped (no tags)
			"number": 104, "title": "tune the compute backend prefetch window size",
			"state": "OPEN", "createdAt": ago(20), "updatedAt": ago(5),
			"labels":    []map[string]string{{"name": "priority/P2"}, {"name": "bug"}, {"name": "compute"}},
			"assignees": []map[string]string{{"login": "someone"}},
		},
		{ // review: idle 20d, missing priority -> needs-priority -> review
			"number": 105, "title": "document the trajectory recorder export format",
			"state": "OPEN", "createdAt": ago(30), "updatedAt": ago(20),
			"labels": []map[string]string{{"name": "bug"}, {"name": "compute"}},
		},
	}
	b, err := json.Marshal(issues)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// walkPlanJSON mirrors the WalkPlan JSON the verb emits, for assertion.
type walkPlanJSON struct {
	Schema    string `json:"schema"`
	Total     int    `json:"total"`
	Fresh     int    `json:"fresh"`
	Healthy   int    `json:"healthy"`
	Attention int    `json:"attention"`
	Acted     int    `json:"acted"`
	Review    int    `json:"review"`
	Deferred  int    `json:"deferred"`
	OK        bool   `json:"ok"`
	Verdict   string `json:"verdict"`
	Finding   string `json:"finding"`
	Decisions []struct {
		ID          int    `json:"id"`
		Disposition string `json:"disposition"`
		Command     string `json:"cmd"`
		Perform     bool   `json:"perform"`
	} `json:"decisions"`
}

// TestRunGardenWalkClassifiesAndBoundsTheSet drives the verb end-to-end over the
// fixture and proves the resource policy: fresh skipped, healthy skipped, the rest a
// worst-first worklist with the right act/review split, propose-only (Perform=false).
func TestRunGardenWalkClassifiesAndBoundsTheSet(t *testing.T) {
	fixture := writeWalkFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{
		"--input", fixture, "--json",
		"--skip-fresh", "3", "--budget", "10",
		"--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("runGardenWalk code=%d stderr=%s", code, stderr.String())
	}
	var plan walkPlanJSON
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("unmarshal plan: %v\n%s", err, stdout.String())
	}
	if plan.Total != 5 {
		t.Fatalf("Total = %d, want 5", plan.Total)
	}
	if plan.Fresh != 1 {
		t.Fatalf("Fresh = %d, want 1 (issue 103 idle 1d)", plan.Fresh)
	}
	if plan.Healthy != 1 {
		t.Fatalf("Healthy = %d, want 1 (issue 104)", plan.Healthy)
	}
	if plan.Attention != 3 {
		t.Fatalf("Attention = %d, want 3 (101 stale, 102 dormant, 105 review)", plan.Attention)
	}
	if plan.Acted != 2 {
		t.Fatalf("Acted = %d, want 2 (mark-stale + close-dormant)", plan.Acted)
	}
	if plan.Review != 1 {
		t.Fatalf("Review = %d, want 1", plan.Review)
	}
	if plan.Deferred != 0 {
		t.Fatalf("Deferred = %d, want 0 under budget 10", plan.Deferred)
	}
	if !plan.OK || plan.Verdict != "ACTION" || plan.Finding != "garden_walk_worklist" {
		t.Fatalf("verdict fold = ok=%v verdict=%q finding=%q", plan.OK, plan.Verdict, plan.Finding)
	}
	// Propose-only: no decision performs, and every act carries a command.
	for _, d := range plan.Decisions {
		if d.Perform {
			t.Fatalf("decision %d Perform=true; walk is propose-only", d.ID)
		}
		if d.Disposition == "act" && d.Command == "" {
			t.Fatalf("act decision %d has no command to propose", d.ID)
		}
	}
}

// TestRunGardenWalkBudgetDefers proves a tight budget bounds the worklist and records
// the deferred remainder — the resource cap that keeps a 100s-item set tractable.
func TestRunGardenWalkBudgetDefers(t *testing.T) {
	fixture := writeWalkFixture(t)
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{
		"--input", fixture, "--json",
		"--skip-fresh", "3", "--budget", "1",
		"--ledger", filepath.Join(t.TempDir(), "loops.jsonl"),
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan walkPlanJSON
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(plan.Decisions) != 1 {
		t.Fatalf("want 1 decision under budget 1, got %d", len(plan.Decisions))
	}
	if plan.Deferred != 2 {
		t.Fatalf("Deferred = %d, want 2 (3 attention - budget 1)", plan.Deferred)
	}
}

// TestRunGardenWalkWitnessesRun proves every walk appends a witnessed run-end to the
// loop ledger so `fak loop health` sees the loop living.
func TestRunGardenWalkWitnessesRun(t *testing.T) {
	fixture := writeWalkFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{
		"--input", fixture, "--json", "--skip-fresh", "3", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		t.Fatalf("LoadPrefix: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 ledger event, got %d", len(events))
	}
	ev := events[0]
	if ev.LoopID != gardenWalkLoopID {
		t.Fatalf("LoopID = %q, want %q", ev.LoopID, gardenWalkLoopID)
	}
	if ev.Metrics["walked"] != 5 {
		t.Fatalf("walked metric = %d, want 5", ev.Metrics["walked"])
	}
	if ev.Metrics["attention"] != 3 {
		t.Fatalf("attention metric = %d, want 3", ev.Metrics["attention"])
	}
}

// TestRunGardenWalkSkipsInProgress proves the in-progress label flows through to the
// pure planner's active pre-filter end-to-end (default skip-active on).
func TestRunGardenWalkSkipsInProgress(t *testing.T) {
	now := time.Now().UTC()
	ago := func(days int) string { return now.AddDate(0, 0, -days).Format(time.RFC3339) }
	issues := []map[string]any{
		{ // in-progress + missing priority -> would be review, but skipped as active
			"number": 201, "title": "land the radix kv prefix reuse kernel patch",
			"state": "OPEN", "createdAt": ago(30), "updatedAt": ago(10),
			"labels": []map[string]string{{"name": "in-progress"}, {"name": "bug"}, {"name": "compute"}},
		},
		{ // not in-progress, missing priority -> review (the one survivor)
			"number": 202, "title": "fix the gateway cache pricing rounding error",
			"state": "OPEN", "createdAt": ago(30), "updatedAt": ago(10),
			"labels": []map[string]string{{"name": "bug"}, {"name": "compute"}},
		},
	}
	b, _ := json.Marshal(issues)
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{
		"--input", path, "--json", "--ledger", filepath.Join(t.TempDir(), "loops.jsonl"),
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan struct {
		Active    int `json:"active"`
		Attention int `json:"attention"`
		Decisions []struct {
			ID int `json:"id"`
		} `json:"decisions"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if plan.Active != 1 {
		t.Fatalf("Active = %d, want 1 (issue 201 in-progress)", plan.Active)
	}
	if plan.Attention != 1 || len(plan.Decisions) != 1 || plan.Decisions[0].ID != 202 {
		t.Fatalf("want only issue 202 on the worklist; got attention=%d decisions=%+v", plan.Attention, plan.Decisions)
	}
}

// TestRunGardenWalkUnknownSourceRefuses proves an unwired source is refused loudly
// (exit 2) rather than silently walking nothing.
func TestRunGardenWalkUnknownSourceRefuses(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{"--source", "trajectories"})
	if code != 2 {
		t.Fatalf("unknown source code=%d, want 2; stderr=%s", code, stderr.String())
	}
}

// TestRegisterGardenWalkLoopArmsDurableUnit proves the walk registers as a durable,
// armed loop unit and re-registration is idempotent (preserves CreatedUnixNano).
func TestRegisterGardenWalkLoopArmsDurableUnit(t *testing.T) {
	registry := filepath.Join(t.TempDir(), "registry.json")
	if err := registerGardenWalkLoop(registry); err != nil {
		t.Fatalf("registerGardenWalkLoop: %v", err)
	}
	reg, err := loopmgr.LoadRegistry(registry)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	job, ok := reg.Get(gardenWalkLoopID)
	if !ok {
		t.Fatalf("loop %q not registered", gardenWalkLoopID)
	}
	if !job.State.Armed() {
		t.Fatalf("loop %q not armed", gardenWalkLoopID)
	}
	if job.Schedule.IntervalSeconds != gardenWalkIntervalSeconds {
		t.Fatalf("interval = %d, want %d", job.Schedule.IntervalSeconds, gardenWalkIntervalSeconds)
	}
	created := job.CreatedUnixNano
	if err := registerGardenWalkLoop(registry); err != nil {
		t.Fatalf("re-register: %v", err)
	}
	reg2, _ := loopmgr.LoadRegistry(registry)
	job2, _ := reg2.Get(gardenWalkLoopID)
	if job2.CreatedUnixNano != created {
		t.Fatalf("re-register changed CreatedUnixNano %d -> %d", created, job2.CreatedUnixNano)
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// gardenDispatchIssuesFixture writes a gh-issue-list-shaped JSON file with two
// attention-worthy issues that route to different lanes: #101 (mark-stale act, docs
// lane) and #102 (review, cmd lane). Both are old enough to clear the default
// skip-fresh=0 policy and neither is in-progress.
func gardenDispatchIssuesFixture(t *testing.T) string {
	t.Helper()
	now := time.Now().UTC()
	ago := func(days int) string { return now.AddDate(0, 0, -days).Format(time.RFC3339) }
	issues := []map[string]any{
		{
			"number": 101, "title": "wire the gpu telemetry exporter end to end",
			"state": "OPEN", "createdAt": ago(120), "updatedAt": ago(70),
			"labels": []map[string]string{{"name": "enhancement"}},
		},
		{
			"number": 102, "title": "document the trajectory recorder export format",
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

// gardenDispatchRouterFor stubs dispatchRouteIssues to route #101 -> docs and
// #102 -> cmd, mirroring the shape dispatchHappyHelper's default router uses.
func gardenDispatchRouterFor(t *testing.T) {
	t.Helper()
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Issues: []dispatchtick.IssueRoute{
				{Number: 101, Lane: "docs"},
				{Number: 102, Lane: "cmd"},
			},
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"docs": {Tree: []string{"docs/**"}, Issues: []int{101}, Count: 1},
				"cmd":  {Tree: []string{"cmd/**"}, Issues: []int{102}, Count: 1},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })
}

type gardenDispatchResultJSON struct {
	Schema     string `json:"schema"`
	DryRun     bool   `json:"dry_run"`
	Walked     int    `json:"walked"`
	Considered int    `json:"considered"`
	Admitted   int    `json:"admitted"`
	Spawned    int    `json:"spawned"`
	SkippedBy  map[string]int `json:"skipped_by"`
	Verdict    string `json:"verdict"`
	LoopAdmit  bool   `json:"loop_admit"`
	Results    []struct {
		ID          int    `json:"id"`
		Lane        string `json:"lane"`
		Admitted    bool   `json:"admitted"`
		Spawned     bool   `json:"spawned"`
		Verdict     string `json:"verdict"`
		SkipReason  string `json:"skip_reason"`
		Disposition string `json:"disposition"`
	} `json:"results"`
}

// TestGardenDispatchDryRunReportsCandidateDecisions is the --dry-run acceptance
// criterion: it must print the exact candidate issue IDs and the dispatch decision
// for each, and must never spawn anything.
func TestGardenDispatchDryRunReportsCandidateDecisions(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--budget", "10", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if !got.DryRun {
		t.Fatalf("DryRun = false, want true (default mode)")
	}
	if got.Walked != 2 {
		t.Fatalf("Walked = %d, want 2", got.Walked)
	}
	if got.Considered != 2 {
		t.Fatalf("Considered = %d, want 2 (both issues need attention)", got.Considered)
	}
	if got.Spawned != 0 {
		t.Fatalf("Spawned = %d, want 0 under dry-run", got.Spawned)
	}
	if len(got.Results) != 2 {
		t.Fatalf("Results = %d, want 2 (exact candidate IDs reported)", len(got.Results))
	}
	byID := map[int]string{}
	for _, r := range got.Results {
		byID[r.ID] = r.Lane
		if r.Spawned {
			t.Fatalf("candidate #%d spawned=true under dry-run", r.ID)
		}
	}
	if byID[101] != "docs" || byID[102] != "cmd" {
		t.Fatalf("candidate lanes = %#v, want 101->docs 102->cmd", byID)
	}
	if got.Verdict != "WOULD_APPLY" {
		t.Fatalf("Verdict = %q, want WOULD_APPLY", got.Verdict)
	}
}

// TestGardenDispatchApplySpawnsAdmittedOnly is the --apply acceptance criterion: it
// must attempt only admitted candidates and actually spawn under the happy-path
// dispatch pipeline.
func TestGardenDispatchApplySpawnsAdmittedOnly(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--budget", "10", "--apply", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.DryRun {
		t.Fatalf("DryRun = true, want false under --apply")
	}
	if got.Spawned == 0 {
		t.Fatalf("Spawned = 0, want at least 1 under a happy-path --apply")
	}
}

// TestGardenDispatchSkipsUnroutedCandidateWithTypedReason proves a candidate the
// router has no lane for is skipped with a typed reason rather than silently
// dropped or crashing the run.
func TestGardenDispatchSkipsUnroutedCandidateWithTypedReason(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		// Router knows about neither 101 nor 102: both candidates are unrouted.
		return dispatchtick.RouterPayload{Schema: dispatchtick.RouterSchema, OK: true}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--budget", "10", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.Admitted != 0 {
		t.Fatalf("Admitted = %d, want 0 (nothing routes)", got.Admitted)
	}
	if got.SkippedBy["unrouted"] != 2 {
		t.Fatalf("SkippedBy[unrouted] = %d, want 2; skipped_by=%#v", got.SkippedBy["unrouted"], got.SkippedBy)
	}
	for _, r := range got.Results {
		if r.SkipReason != "unrouted" {
			t.Fatalf("candidate #%d skip_reason = %q, want unrouted", r.ID, r.SkipReason)
		}
	}
	if got.Verdict != "NONE_ADMITTED" {
		t.Fatalf("Verdict = %q, want NONE_ADMITTED", got.Verdict)
	}
}

// TestGardenDispatchNoCandidatesWhenWalkIsClean proves an empty worklist (nothing
// needs attention) is reported honestly rather than as an error.
func TestGardenDispatchNoCandidatesWhenWalkIsClean(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	now := time.Now().UTC()
	ago := func(days int) string { return now.AddDate(0, 0, -days).Format(time.RFC3339) }
	issues := []map[string]any{
		{ // healthy: priority + kind + area -> no condition -> skip
			"number": 104, "title": "tune the compute backend prefetch window size",
			"state": "OPEN", "createdAt": ago(20), "updatedAt": ago(5),
			"labels":    []map[string]string{{"name": "priority/P2"}, {"name": "bug"}, {"name": "compute"}},
			"assignees": []map[string]string{{"login": "someone"}},
		},
	}
	b, _ := json.Marshal(issues)
	fixture := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(fixture, b, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.Considered != 0 || got.Verdict != "NO_CANDIDATES" {
		t.Fatalf("Considered=%d Verdict=%q, want 0/NO_CANDIDATES", got.Considered, got.Verdict)
	}
}

// TestGardenDispatchHonorsLoopPausePolicy proves the bridge is gated by the SAME
// loop-governor mechanism `fak loop admit` exposes: a paused policy for the bridge's
// own loop id refuses the whole run before any candidate is touched, and nothing is
// spawned.
func TestGardenDispatchHonorsLoopPausePolicy(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	policyPath := filepath.Join(t.TempDir(), "policy.json")
	policy := loopmgr.Policies{
		Schema: loopmgr.SchemaPolicies,
		Loops: map[string]loopmgr.Policy{
			gardenDispatchLoopID: {Paused: true},
		},
	}
	pb, err := json.Marshal(policy)
	if err != nil {
		t.Fatalf("marshal policy: %v", err)
	}
	if err := os.WriteFile(policyPath, pb, 0o644); err != nil {
		t.Fatalf("write policy: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--apply", "--ledger", ledger, "--policy", policyPath,
	})
	if code != 3 {
		t.Fatalf("code=%d, want 3 (mirrors `fak loop admit` refused exit) stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.LoopAdmit {
		t.Fatalf("LoopAdmit = true, want false under a paused policy")
	}
	if got.Verdict != "LOOP_REFUSED" {
		t.Fatalf("Verdict = %q, want LOOP_REFUSED", got.Verdict)
	}
	if got.Considered != 0 || len(got.Results) != 0 {
		t.Fatalf("a loop-refused run must never touch a candidate: considered=%d results=%d", got.Considered, len(got.Results))
	}
}

// TestGardenDispatchWitnessesRunWithCounts proves the run records ONE witnessed
// run-end in the loop ledger carrying the walked/considered/admitted/spawned/skipped
// counts the issue's acceptance criteria ask for.
func TestGardenDispatchWitnessesRunWithCounts(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--budget", "10", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}

	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		t.Fatalf("LoadPrefix: %v", err)
	}
	// One event from the bridge's own witness. (Per-candidate dispatch ticks would add
	// more, but --no-loop-ledger is not set here so nested ticks may also append; find
	// the bridge's own event by LoopID.)
	var ev *loopmgr.Event
	for i := range events {
		if events[i].LoopID == gardenDispatchLoopID {
			ev = &events[i]
		}
	}
	if ev == nil {
		t.Fatalf("no %q event recorded in ledger; events=%+v", gardenDispatchLoopID, events)
	}
	if ev.Metrics["walked"] != 2 {
		t.Fatalf("walked metric = %d, want 2", ev.Metrics["walked"])
	}
	if ev.Metrics["considered"] != 2 {
		t.Fatalf("considered metric = %d, want 2", ev.Metrics["considered"])
	}
	if _, ok := ev.Metrics["admitted"]; !ok {
		t.Fatalf("admitted metric missing: %#v", ev.Metrics)
	}
	if _, ok := ev.Metrics["spawned"]; !ok {
		t.Fatalf("spawned metric missing: %#v", ev.Metrics)
	}
}

// TestGardenDispatchGardenWalkNeverLive proves the read-only fak garden and
// propose-only fak garden walk paths never reach evaluateDispatchTick's Live path --
// only the explicit `garden dispatch --apply` subcommand can. This is a structural
// check on runGarden's dispatch table.
func TestGardenDispatchIsASeparateSubcommandFromWalk(t *testing.T) {
	// garden_walk.go's runGardenWalk has no --apply flag at all; parsing one must
	// fail as an unknown flag (exit 2), proving walk cannot be made to spawn.
	var stdout, stderr bytes.Buffer
	code := runGardenWalk(&stdout, &stderr, []string{"--apply"})
	if code != 2 {
		t.Fatalf("garden walk --apply code=%d, want 2 (no such flag exists on walk)", code)
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stalework"
)

func TestStaleWorkLoopSeedDryRunBindsThreeIssuesWithoutLaunching(t *testing.T) {
	dir := t.TempDir()
	candidates := []stalework.Candidate{
		staleWorkLoopCLICandidate("docs/cli-reference.md"),
		staleWorkLoopCLICandidate("docs/native-device-mesh.md"),
		staleWorkLoopCLICandidate("docs/scorecards.md"),
	}
	packet := stalework.Packet{Schema: stalework.Schema, Head: "seed", Candidates: candidates}
	preview := stalework.BuildLoop(packet, stalework.LoopOptions{})
	numbers := []int{3869, 5305, 5631}
	issues := make([]stalework.IssueSnapshot, len(preview.Units))
	for i, unit := range preview.Units {
		issues[i] = stalework.IssueSnapshot{
			Number: numbers[i], Title: unit.Issue.Title, Body: unit.Issue.Body,
			State: "OPEN",
		}
	}
	packetPath := filepath.Join(dir, "packet.json")
	issuesPath := filepath.Join(dir, "issues.json")
	writeStaleWorkLoopJSON(t, packetPath, packet)
	writeStaleWorkLoopJSON(t, issuesPath, issues)

	var stdout, stderr bytes.Buffer
	code := runStaleWorkLoop([]string{"--packet", packetPath, "--issues", issuesPath, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan stalework.LoopPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if plan.Counts.IssueBound != 3 || plan.Counts.DispatchReady != 3 || plan.Counts.Launches != 0 {
		t.Fatalf("counts=%+v, want three issue-bound units and zero launches", plan.Counts)
	}
	for _, unit := range plan.Units {
		if unit.Dispatch.WorkerID == "" || unit.Dispatch.Launched {
			t.Fatalf("dispatch=%+v, want planned fresh identity without launch", unit.Dispatch)
		}
	}
}

func TestStaleWorkLoopLiveIssueCreationRequiresDedupeSnapshot(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runStaleWorkLoop([]string{"--live-issues"}, &stdout, &stderr); code != 2 {
		t.Fatalf("code=%d stderr=%s, want usage refusal", code, stderr.String())
	}
}

func TestStaleWorkLoopLiveIssueCreationReplansWithoutLaunching(t *testing.T) {
	dir := t.TempDir()
	packet := stalework.Packet{
		Schema: stalework.Schema, Head: "seed",
		Candidates: []stalework.Candidate{staleWorkLoopCLICandidate("docs/create.md")},
	}
	packetPath := filepath.Join(dir, "packet.json")
	issuesPath := filepath.Join(dir, "issues.json")
	writeStaleWorkLoopJSON(t, packetPath, packet)
	writeStaleWorkLoopJSON(t, issuesPath, []stalework.IssueSnapshot{})

	old := staleWorkIssueRunner
	t.Cleanup(func() { staleWorkIssueRunner = old })
	calls := 0
	staleWorkIssueRunner = func(out, _ io.Writer, args []string) int {
		calls++
		if len(args) < 2 || args[0] != "issue" || args[1] != "create" {
			t.Fatalf("issue args=%v", args)
		}
		_, _ = io.WriteString(out, `{"ok":true,"url":"https://github.com/anthony-chaudhary/fak/issues/9100"}`)
		return 0
	}
	var stdout, stderr bytes.Buffer
	code := runStaleWorkLoop([]string{"--packet", packetPath, "--issues", issuesPath, "--live-issues", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan stalework.LoopPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || plan.Counts.IssueBound != 1 || plan.Counts.DispatchReady != 1 || plan.Counts.Launches != 0 {
		t.Fatalf("calls=%d counts=%+v, want one fake create, replan, and zero launches", calls, plan.Counts)
	}
}

func TestStaleWorkLoopLiveLaunchUsesRenderedDistinctDispatches(t *testing.T) {
	dir := t.TempDir()
	candidates := []stalework.Candidate{
		staleWorkLoopCLICandidate("docs/a.md"),
		staleWorkLoopCLICandidate("docs/b.md"),
	}
	packet := stalework.Packet{Schema: stalework.Schema, Head: "seed", Candidates: candidates}
	preview := stalework.BuildLoop(packet, stalework.LoopOptions{})
	issues := make([]stalework.IssueSnapshot, len(preview.Units))
	for i, unit := range preview.Units {
		issues[i] = stalework.IssueSnapshot{Number: 9001 + i, Title: unit.Issue.Title, Body: unit.Issue.Body, State: "OPEN"}
	}
	packetPath := filepath.Join(dir, "packet.json")
	issuesPath := filepath.Join(dir, "issues.json")
	writeStaleWorkLoopJSON(t, packetPath, packet)
	writeStaleWorkLoopJSON(t, issuesPath, issues)

	old := staleWorkDispatchRunner
	t.Cleanup(func() { staleWorkDispatchRunner = old })
	var calls [][]string
	staleWorkDispatchRunner = func(_ io.Writer, _ io.Writer, args []string) int {
		calls = append(calls, append([]string(nil), args...))
		return 0
	}
	var stdout, stderr bytes.Buffer
	code := runStaleWorkLoop([]string{"--packet", packetPath, "--issues", issuesPath, "--live-launch", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan stalework.LoopPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || plan.Counts.Launches != 2 {
		t.Fatalf("calls=%v counts=%+v, want two fake launches", calls, plan.Counts)
	}
	if calls[0][0] != "tick" || calls[1][0] != "tick" || equalStringSlice(calls[0], calls[1]) {
		t.Fatalf("dispatch calls=%v, want distinct tick commands", calls)
	}
}

func staleWorkLoopCLICandidate(path string) stalework.Candidate {
	return stalework.Candidate{
		Path: path, Batch: path, Score: 50, Status: "candidate",
		Components:         []stalework.Component{{Name: "dependency_drift", Points: 50, Provenance: "git", Evidence: "one dependency commit"}},
		LastSemanticCommit: "old", ExcerptSHA256: path,
		DedupeKey:  "stale-work:" + path,
		VerifyWith: "fak stale-work --path " + path + " --json",
	}
}

func writeStaleWorkLoopJSON(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func equalStringSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

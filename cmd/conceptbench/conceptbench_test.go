package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captureRun runs the command with argv, capturing stdout, and returns (exit, stdout).
func captureRun(t *testing.T, argv []string) (int, string) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run(argv)
	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = buf.ReadFrom(r)
	return code, buf.String()
}

func replayDir() string { return filepath.Join("testdata", "replay") }

// TestReplaySmokeWritesReport is the DoD smoke test: the runner grades the replay
// fixtures end-to-end (>=1 concept x >=2 models), exits 0, writes a non-empty
// fak.conceptbench.report.v1, and the fixture exercises both a pass and a fail.
func TestReplaySmokeWritesReport(t *testing.T) {
	code, out := captureRun(t, []string{"--replay", replayDir()})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("empty report")
	}
	var rep struct {
		Schema             string   `json:"schema"`
		Mode               string   `json:"mode"`
		Grader             string   `json:"grader"`
		ResultClaimAllowed bool     `json:"result_claim_allowed"`
		Models             []string `json:"models"`
		Concepts           []string `json:"concepts"`
		Cells              []struct {
			Model   string `json:"model"`
			Concept string `json:"concept"`
			TaskID  string `json:"task_id"`
			Pass    bool   `json:"pass"`
		} `json:"cells"`
		Leaderboard []struct {
			Model   string  `json:"model"`
			PassAt1 float64 `json:"pass_at_1"`
			Passed  int     `json:"passed"`
			Total   int     `json:"total"`
		} `json:"leaderboard"`
		Totals struct {
			Passed   int `json:"passed"`
			Failed   int `json:"failed"`
			Attempts int `json:"attempts"`
		} `json:"totals"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("report not valid JSON: %v\n%s", err, out)
	}
	if rep.Schema != reportSchema {
		t.Fatalf("schema=%q, want %q", rep.Schema, reportSchema)
	}
	if rep.Mode != "replay" || rep.Grader != graderID {
		t.Fatalf("mode=%q grader=%q", rep.Mode, rep.Grader)
	}
	if rep.ResultClaimAllowed {
		t.Fatal("replay report must pin result_claim_allowed=false")
	}
	if len(rep.Models) < 2 {
		t.Fatalf("want >=2 models, got %v", rep.Models)
	}
	if len(rep.Concepts) < 1 {
		t.Fatalf("want >=1 concept, got %v", rep.Concepts)
	}
	if len(rep.Cells) == 0 || len(rep.Leaderboard) == 0 {
		t.Fatalf("empty cells/leaderboard: cells=%d leaderboard=%d", len(rep.Cells), len(rep.Leaderboard))
	}
	var pass, fail int
	for _, c := range rep.Cells {
		if c.Pass {
			pass++
		} else {
			fail++
		}
	}
	if pass == 0 || fail == 0 {
		t.Fatalf("fixture must exercise both a pass and a fail; pass=%d fail=%d", pass, fail)
	}
	if rep.Totals.Attempts != len(rep.Cells) || rep.Totals.Passed != pass || rep.Totals.Failed != fail {
		t.Fatalf("totals disagree with cells: totals=%+v pass=%d fail=%d", rep.Totals, pass, fail)
	}
	// The raw substring the honesty gate downstream greps for (matches livecodebench).
	if !strings.Contains(out, `"result_claim_allowed": false`) {
		t.Fatalf("report did not pin result_claim_allowed=false:\n%s", out)
	}
}

// TestContractHasNoScoresAndClaimDisallowed pins the --contract discipline (#868):
// a pre-run contract lists models/concepts/task ids + budget, holds the claim gate
// shut, and carries NO scores (no cells/leaderboard/totals).
func TestContractHasNoScoresAndClaimDisallowed(t *testing.T) {
	code, out := captureRun(t, []string{"--contract", "--replay", replayDir(), "--models", "opus-4.8,smollm2-135m"})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}
	if !strings.Contains(out, `"result_claim_allowed": false`) {
		t.Fatalf("contract must pin result_claim_allowed=false:\n%s", out)
	}
	var c map[string]any
	if err := json.Unmarshal([]byte(out), &c); err != nil {
		t.Fatalf("contract not valid JSON: %v\n%s", err, out)
	}
	if c["schema"] != contractSchema {
		t.Fatalf("schema=%v, want %q", c["schema"], contractSchema)
	}
	for _, k := range []string{"leaderboard", "cells", "totals"} {
		if _, ok := c[k]; ok {
			t.Fatalf("contract must not carry %q (no scores before a graded run)", k)
		}
	}
	ids, _ := c["task_ids"].([]any)
	if len(ids) == 0 {
		t.Fatal("contract must list task ids")
	}
	models, _ := c["models"].([]any)
	if len(models) != 2 {
		t.Fatalf("contract models=%v, want the two requested", c["models"])
	}
}

// TestContractRequiresModels guards the operator-declared model set.
func TestContractRequiresModels(t *testing.T) {
	code, _ := captureRun(t, []string{"--contract", "--replay", replayDir()})
	if code == 0 {
		t.Fatal("--contract without --models must refuse")
	}
}

// TestLiveModeRefusesWithoutDriverRegistry keeps the runner honest: with neither
// --replay nor --contract there is no wired live path, so it must refuse (not fake a
// pass) until the model-driver registry (#2731) lands.
func TestLiveModeRefusesWithoutDriverRegistry(t *testing.T) {
	code, _ := captureRun(t, []string{"--models", "opus-4.8,smollm2-135m", "--concepts", "structured-refusal"})
	if code == 0 {
		t.Fatal("a live run without --replay/--contract must refuse until #2731/#2732 are wired")
	}
}

// TestReplayWritesToFile proves the --out file path emits a lineage-stamped artifact.
func TestReplayWritesToFile(t *testing.T) {
	out := filepath.Join(t.TempDir(), "report.json")
	code, _ := captureRun(t, []string{"--replay", replayDir(), "--out", out})
	if code != 0 {
		t.Fatalf("exit=%d, want 0", code)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read out: %v", err)
	}
	if !strings.Contains(string(b), reportSchema) {
		t.Fatalf("report file missing schema:\n%s", b)
	}
	// benchcli stamps the four lineage axes onto the artifact.
	if !strings.Contains(string(b), "git_commit") && !strings.Contains(string(b), "lineage") {
		t.Fatalf("report file missing lineage stamp:\n%s", b)
	}
}

// TestConceptFilterSelectsOneConcept proves --concepts narrows the graded cells.
func TestConceptFilterSelectsOneConcept(t *testing.T) {
	code, out := captureRun(t, []string{"--replay", replayDir(), "--concepts", "structured-refusal"})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}
	var rep struct {
		Concepts []string `json:"concepts"`
		Cells    []struct {
			Concept string `json:"concept"`
		} `json:"cells"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if len(rep.Concepts) != 1 || rep.Concepts[0] != "structured-refusal" {
		t.Fatalf("concepts=%v, want [structured-refusal]", rep.Concepts)
	}
	for _, c := range rep.Cells {
		if c.Concept != "structured-refusal" {
			t.Fatalf("cell concept=%q leaked past --concepts filter", c.Concept)
		}
	}
}

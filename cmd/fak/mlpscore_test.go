package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mlpscore"
)

func TestMLPScoreCLIReadsCommittedHeadOnly(t *testing.T) {
	root := initMLPTestRepo(t)
	score := runMLPScoreJSON(t, root)
	if score.Witnessed != 0 || score.MLPVerdict != "not-yet" {
		t.Fatalf("empty snapshot score = %+v", score)
	}

	proof := "docs/mlp/proofs/init-agent.md"
	manifest := mlpscore.WitnessManifest{
		Schema:    mlpscore.WitnessSchema,
		Criterion: "init_agent_emits_governed_agent",
		Claims: []mlpscore.WitnessClaim{{
			Key: "scaffolded_agent_completed", Kind: "captured-run", Path: proof, Command: "generated-agent --selfcheck",
		}},
	}
	writeMLPJSON(t, filepath.Join(root, mlpscore.WitnessDir, "init_agent_emits_governed_agent.json"), manifest)
	writeMLPFile(t, filepath.Join(root, filepath.FromSlash(proof)), "captured run\n")

	untracked := runMLPScoreJSON(t, root)
	if untracked.Witnessed != 0 {
		t.Fatalf("untracked witness moved score to %d", untracked.Witnessed)
	}

	runMLPTestGit(t, root, "add", "docs/mlp")
	runMLPTestGit(t, root, "commit", "-q", "-m", "ship one witness")
	committed := runMLPScoreJSON(t, root)
	if committed.Witnessed != 1 || committed.MLPVerdict != "not-yet" {
		t.Fatalf("committed witness score = %+v", committed)
	}
	var found bool
	for _, row := range committed.Criteria {
		if row.Key == "init_agent_emits_governed_agent" {
			found = true
			if row.Grade != mlpscore.GradeWitnessed || len(row.Evidence) != 1 {
				t.Fatalf("D2 row = %+v", row)
			}
		}
	}
	if !found {
		t.Fatal("D2 criterion missing")
	}
}

func TestMLPScoreCLIOutputAndGate(t *testing.T) {
	root := initMLPTestRepo(t)
	var out, errb bytes.Buffer
	if code := runMLPScore(&out, &errb, []string{"--workspace", root, "--markdown"}); code != 0 {
		t.Fatalf("markdown code = %d, stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "MLP scorecard - first lovable cut") ||
		!strings.Contains(out.String(), "](docs/mlp/witnesses/") {
		t.Fatalf("markdown output:\n%s", out.String())
	}

	out.Reset()
	errBReset(&errb)
	if code := runMLPScore(&out, &errb, []string{"--workspace", root, "--check"}); code != 1 {
		t.Fatalf("not-yet --check code = %d, want 1; stderr=%s", code, errb.String())
	}

	out.Reset()
	errBReset(&errb)
	if code := runMLPScore(&out, &errb, []string{"--workspace", root, "--json", "--markdown"}); code != 2 {
		t.Fatalf("conflicting output code = %d, want 2", code)
	}
}

func TestMLPMilestoneProjectionPreservesCriterionRows(t *testing.T) {
	score := mlpscore.Grade(nil, mlpscore.FoldOpts{Commit: "abc", Date: "2026-07-10"})
	card := mlpMilestoneScorecard(score)
	if card.Milestone != 17 || card.Key != "mlp" || card.Verdict != "not-yet" {
		t.Fatalf("card head = %+v", card)
	}
	if len(card.Criteria) != score.Total || card.Total != score.Total {
		t.Fatalf("card rows = %d/%d, score total %d", len(card.Criteria), card.Total, score.Total)
	}
	for i, row := range card.Criteria {
		if row.Workstream != score.Criteria[i].Workstream || row.WitnessRef != score.Criteria[i].WitnessRef {
			t.Fatalf("card row %d drifted: %+v vs %+v", i, row, score.Criteria[i])
		}
	}
}

func runMLPScoreJSON(t *testing.T, root string) mlpscore.Score {
	t.Helper()
	var out, errb bytes.Buffer
	if code := runMLPScore(&out, &errb, []string{"--workspace", root, "--json"}); code != 0 {
		t.Fatalf("mlp-score --json code = %d, stderr=%s", code, errb.String())
	}
	var score mlpscore.Score
	if err := json.Unmarshal(out.Bytes(), &score); err != nil {
		t.Fatalf("decode mlp score: %v\n%s", err, out.String())
	}
	return score
}

func initMLPTestRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	runMLPTestGit(t, root, "init", "-q")
	runMLPTestGit(t, root, "config", "user.email", "mlp-cli@example.invalid")
	runMLPTestGit(t, root, "config", "user.name", "MLP CLI Test")
	writeMLPFile(t, filepath.Join(root, "README.md"), "fixture\n")
	runMLPTestGit(t, root, "add", "README.md")
	runMLPTestGit(t, root, "commit", "-q", "-m", "seed")
	return root
}

func runMLPTestGit(t *testing.T, root string, argv ...string) {
	t.Helper()
	cmd := exec.Command("git", argv...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", argv, err, out)
	}
}

func writeMLPFile(t *testing.T, name, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeMLPJSON(t *testing.T, name string, value any) {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeMLPFile(t, name, string(raw)+"\n")
}

func errBReset(buf *bytes.Buffer) { buf.Reset() }

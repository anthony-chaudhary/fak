package main

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func spineFixturePath() string { return filepath.Join("testdata", "spine", "spine.json") }

// TestSpineDeterministicPassAndFail is the #2729 acceptance gate: the spine runs
// one concept task against two named models from a single command, exits 0, and
// each arm's PRODUCED commit is graded by a real `dos commit-audit` call — the
// stamped path-scoped arm deterministically grades OK / diff-witnessed (pass)
// and the subject-only arm deterministically grades CLAIM_UNWITNESSED /
// subject-only (fail), with witness_source recorded honestly.
func TestSpineDeterministicPassAndFail(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos CLI not on PATH — the spine grade is a real dos commit-audit call, never a recording")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH — the spine replays arms into a scratch git repo")
	}

	code, out := captureRun(t, []string{"--spine", spineFixturePath()})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}

	var rep struct {
		Schema             string `json:"schema"`
		Mode               string `json:"mode"`
		Concept            string `json:"concept"`
		Grader             string `json:"grader"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		ResultClaimReason  string `json:"result_claim_reason"`
		Rows               []struct {
			Model         string `json:"model"`
			Concept       string `json:"concept"`
			Pass          bool   `json:"pass"`
			Source        string `json:"source"`
			Verdict       string `json:"verdict"`
			Witness       string `json:"witness"`
			WitnessSource string `json:"witness_source"`
			StampLeaf     string `json:"stamp_leaf"`
			Branch        string `json:"branch"`
			CommitSHA     string `json:"commit_sha"`
			Evidence      string `json:"evidence"`
		} `json:"rows"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("parse report: %v\n%s", err, out)
	}

	if rep.Schema != "fak.conceptbench.v1" {
		t.Errorf("schema = %q, want fak.conceptbench.v1", rep.Schema)
	}
	if rep.Concept != "commit_stamp" {
		t.Errorf("concept = %q, want commit_stamp", rep.Concept)
	}
	if rep.ResultClaimAllowed {
		t.Error("result_claim_allowed = true; a replay spine run must never claim a result (#868 discipline)")
	}
	if len(rep.Rows) != 2 {
		t.Fatalf("rows = %d, want one row per model (2)", len(rep.Rows))
	}

	byModel := map[string]int{}
	for i, r := range rep.Rows {
		byModel[r.Model] = i
		if r.WitnessSource != "dos_commit_audit" {
			t.Errorf("row %s witness_source = %q, want dos_commit_audit", r.Model, r.WitnessSource)
		}
		if r.Source != "replay" {
			t.Errorf("row %s source = %q, want replay (no live arm ran)", r.Model, r.Source)
		}
		if r.CommitSHA == "" {
			t.Errorf("row %s has no produced commit sha", r.Model)
		}
		if r.Branch != "main" {
			t.Errorf("row %s branch = %q, want main (trunk fidelity)", r.Model, r.Branch)
		}
	}

	passIdx, ok := byModel["claude-opus-4-8"]
	if !ok {
		t.Fatalf("no row for claude-opus-4-8: %s", out)
	}
	pass := rep.Rows[passIdx]
	if !pass.Pass || !strings.EqualFold(pass.Verdict, "OK") || pass.Witness != "diff-witnessed" || pass.StampLeaf != "gateway" {
		t.Errorf("deterministic-pass arm graded {pass=%v verdict=%s witness=%s leaf=%s}, want {true OK diff-witnessed gateway}\nevidence: %s",
			pass.Pass, pass.Verdict, pass.Witness, pass.StampLeaf, pass.Evidence)
	}

	failIdx, ok := byModel["claude-3-5-haiku"]
	if !ok {
		t.Fatalf("no row for claude-3-5-haiku: %s", out)
	}
	fail := rep.Rows[failIdx]
	if fail.Pass || fail.Verdict != "CLAIM_UNWITNESSED" || fail.Witness != "subject-only" {
		t.Errorf("deterministic-fail arm graded {pass=%v verdict=%s witness=%s}, want {false CLAIM_UNWITNESSED subject-only}\nevidence: %s",
			fail.Pass, fail.Verdict, fail.Witness, fail.Evidence)
	}
}

// TestSpineRefusesASingleArm pins the two-model contract without needing the dos
// CLI: filtering the fixture to one model must refuse (exit 2), never grade a
// one-arm "spine".
func TestSpineRefusesASingleArm(t *testing.T) {
	code, _ := captureRun(t, []string{"--spine", spineFixturePath(), "--models", "claude-opus-4-8"})
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (the spine is a two-model contrast)", code)
	}
}

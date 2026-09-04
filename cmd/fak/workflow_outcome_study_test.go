package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRunWorkflowOutcomeStudyReportsIncompleteWithoutOverclaim(t *testing.T) {
	path := filepath.Join(t.TempDir(), "study.json")
	raw := []byte(`{"schema":"fak-workflow-outcome-study/1","study_id":"frozen","tasks":[{"id":"T1","class":"serial","prompt":"inspect","rubric":["correct"]}],"results":[],"grades":[]}`)
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runWorkflowOutcomeStudy(&out, &stderr, []string{"--input", path, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(out.Bytes(), []byte(`"gain_claim_ready": false`)) || !bytes.Contains(out.Bytes(), []byte(`"complete_pairs": 0`)) {
		t.Fatalf("out=%s", out.String())
	}
}

func TestRunWorkflowOutcomeStudyPRCohortCLI(t *testing.T) {
	var out, stderr bytes.Buffer
	if code := runWorkflowOutcomeStudy(&out, &stderr, []string{"pr-cohort", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(out.Bytes(), []byte(`"schema": "fak-wip-pr-cohort/1"`)) {
		t.Fatalf("out=%s", out.String())
	}
	if !bytes.Contains(out.Bytes(), []byte(`"decision": "REJECT_PR_LANE"`)) {
		t.Fatalf("out=%s", out.String())
	}

	out.Reset()
	stderr.Reset()
	if code := runWorkflowOutcomeStudy(&out, &stderr, []string{"pr-cohort"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(out.Bytes(), []byte("decision=REJECT_PR_LANE")) {
		t.Fatalf("out=%s", out.String())
	}

	// Test via runWip
	out.Reset()
	stderr.Reset()
	if code := runWip(&out, &stderr, []string{"pr-cohort", "--json"}); code != 0 {
		t.Fatalf("runWip code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(out.Bytes(), []byte(`"decision": "REJECT_PR_LANE"`)) {
		t.Fatalf("out=%s", out.String())
	}
}

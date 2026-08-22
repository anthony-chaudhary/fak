package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelQwen38LadderSelfcheckRendersFastToExactTargetSpine(t *testing.T) {
	if testing.Short() {
		t.Skip("cmd/fak package init starts shared endpoints; covered by internal evaluator tests in short mode")
	}
	var stdout, stderr bytes.Buffer
	if exit := runModelQwen38Ladder(&stdout, &stderr, []string{"--selfcheck"}); exit != 0 {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
	for _, want := range []string{"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B", "does_not_prove"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q: %s", want, stdout.String())
		}
	}
}

func TestModelQwen38LadderEvidencePromotesToFirstStage(t *testing.T) {
	if testing.Short() {
		t.Skip("cmd/fak package init starts shared endpoints; covered by internal evaluator tests in short mode")
	}
	path := filepath.Join(t.TempDir(), "evidence.json")
	body := `{"schema":"fak.qwen38-ladder-evidence/1","concept":"loader","corpus_sha256":"corpus","baseline_runtime_sha":"base","candidate_runtime_sha":"candidate","metric":"p95_ms","direction":"lower","minimum_improvement_pct":5,"results":[]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if exit := runModelQwen38Ladder(&stdout, &stderr, []string{"--evidence", path}); exit != 0 {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"verdict": "PROMOTE"`) || !strings.Contains(stdout.String(), `"id": "smoke"`) {
		t.Fatalf("output=%s", stdout.String())
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/learningmesh"
)

func TestRunLearningMeshCompile(t *testing.T) {
	fixture := filepath.Join("..", "..", "docs", "_witnesses", "issue-9839", "mechanisms.json")
	var out, errout bytes.Buffer
	if code := runLearningMesh(&out, &errout, []string{"compile", "--file", fixture}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var result learningmesh.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Schema != learningmesh.OutputSchema || result.CandidateCount == 0 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestRunLearningMeshUsage(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runLearningMesh(&out, &errout, nil); code != 2 {
		t.Fatalf("code=%d want 2", code)
	}
}

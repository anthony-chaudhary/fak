package main

import (
	"bytes"
	"encoding/json"
	"os"
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

func TestRunLearningMeshFromReceipts(t *testing.T) {
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })
	root := filepath.Join("docs", "_witnesses", "issue-9886")
	args := []string{"from-receipts",
		"--receipt", filepath.Join(root, "amd-vulkan.json"),
		"--receipt", filepath.Join(root, "nvidia-cuda.json"),
		"--receipt", filepath.Join(root, "apple-metal.json"),
		"--targets", filepath.Join(root, "targets.json"),
	}
	var out, errout bytes.Buffer
	if code := runLearningMesh(&out, &errout, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var result learningmesh.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.CandidateCount != 9 {
		t.Fatalf("candidate_count=%d want 9", result.CandidateCount)
	}
}

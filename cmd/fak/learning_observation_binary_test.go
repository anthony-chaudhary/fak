package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestLearningObservationPublicBinaryPath is the dispatch witness for #5984: it
// builds the public entry point and invokes the real fak binary rather than
// calling runLearningObservation directly.
func TestLearningObservationPublicBinaryPath(t *testing.T) {
	if testing.Short() {
		t.Skip("public binary build is an integration witness")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(t.TempDir(), "fak")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	build := exec.Command("go", "build", "-o", exe, "./cmd/fak")
	build.Dir = root
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build public fak binary: %v\n%s", err, out)
	}

	store := filepath.Join(t.TempDir(), "lineage.json")
	cmd := exec.Command(exe, "learning-observation", "add", "--store", store,
		"--kind", "observation", "--source", "fixture://binary", "--content", "public route")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("public learning-observation route: %v\n%s", err, out)
	}
	var got struct {
		Record struct {
			Kind    string `json:"kind"`
			Source  string `json:"source"`
			Content string `json:"content"`
		} `json:"record"`
		Created bool `json:"created"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode public binary output: %v\n%s", err, out)
	}
	if !got.Created || got.Record.Kind != "observation" || got.Record.Source != "fixture://binary" || got.Record.Content != "public route" {
		t.Fatalf("unexpected public binary result: %+v", got)
	}
}

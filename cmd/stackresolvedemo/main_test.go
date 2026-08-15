package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelfcheckCommand(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "-selfcheck")
	cmd.Dir = "."
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("selfcheck: %v\nstderr:\n%s", err, stderr.String())
	}
	for _, want := range []string{"ALLOW stack", "REFUSE stack", "device.cuda.sm80", "SELFCHECK PASS"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestManifestRefusalUsesDistinctExit(t *testing.T) {
	manifest := filepath.Join("..", "..", "internal", "stackresolve", "testdata", "awq-sm75-unsat.json")
	if _, err := os.Stat(manifest); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "run", ".", "-manifest", manifest)
	cmd.Dir = "."
	output, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("refused manifest exited zero:\n%s", output)
	}
	if !strings.Contains(string(output), "UNSATISFIED_REQUIREMENT") || !strings.Contains(string(output), "exit status 3") {
		t.Fatalf("unexpected refusal:\n%s", output)
	}
}

package landregression

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestLandProspectiveBindsUncommittedCandidate(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	sourceDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	sourceRoot := filepath.Clean(filepath.Join(sourceDir, "..", "..", ".."))
	tmp := t.TempDir()
	generatedTest := filepath.Join(tmp, "worktree_candidate_regression_test.go")
	testSource := `package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestWorkerLandSymptomBindsUncommittedCandidate(t *testing.T) {
	repo, worktree, base := newSymptomWorkerFixture(t, true)
	if out, err := exec.Command("git", "-C", worktree, "reset", "--mixed", base).CombinedOutput(); err != nil {
		t.Fatalf("reset worker to base: %v: %s", err, out)
	}
	fakDir := filepath.Join(worktree, ".fak")
	if err := os.MkdirAll(fakDir, 0o755); err != nil {
		t.Fatal(err)
	}
	receipt := []byte("{\"schema\":\"fak-test-witness/1\",\"status\":\"passed\",\"verdict\":\"PASS\",\"witness\":\"test-witnessed\"}")
	if err := os.WriteFile(filepath.Join(fakDir, "test-witness.json"), receipt, 0o644); err != nil {
		t.Fatal(err)
	}
	msg := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(msg, []byte("fix(calc): fix negative calculation (fak calc)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	res, code := runWorktreeWorkerLand(&out, &errb, []string{
		"--root", repo,
		"--worktree", worktree,
		"--base-sha", base,
		"--paths", "pkg/calc.go",
		"--paths", "pkg/calc_test.go",
		"--msg-file", msg,
		"--require-test-witness",
	})
	if !res.OK || code != 0 || !res.Applied || !res.Committed {
		t.Fatalf("verified uncommitted candidate did not land: res=%+v code=%d err=%s", res, code, errb.String())
	}
}
`
	if err := os.WriteFile(generatedTest, []byte(testSource), 0o644); err != nil {
		t.Fatal(err)
	}
	overlay := struct {
		Replace map[string]string
	}{Replace: map[string]string{
		filepath.Join(sourceRoot, "cmd", "fak", "worktree_candidate_regression_test.go"): generatedTest,
	}}
	overlayJSON, err := json.Marshal(overlay)
	if err != nil {
		t.Fatal(err)
	}
	overlayFile := filepath.Join(tmp, "overlay.json")
	if err := os.WriteFile(overlayFile, overlayJSON, 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("go", "test", "-count=1", "-overlay", overlayFile, "./cmd/fak", "-run", "^TestWorkerLandSymptomBindsUncommittedCandidate$")
	cmd.Dir = sourceRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("focused CLI regression: %v\n%s", err, out)
	}
}

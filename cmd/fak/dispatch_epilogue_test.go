package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchEpilogueCLI(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	repo := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test-cli",
			"GIT_AUTHOR_EMAIL=cli@localhost",
			"GIT_COMMITTER_NAME=test-cli",
			"GIT_COMMITTER_EMAIL=cli@localhost",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	git("init", "-q")
	git("config", "user.name", "test-cli")
	git("config", "user.email", "cli@localhost")

	filePath := filepath.Join(repo, "cli_test.txt")
	if err := os.WriteFile(filePath, []byte("initial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "cli_test.txt")
	git("commit", "-m", "init: cli test (fak kernel)")

	// Create a patch
	if err := os.WriteFile(filePath, []byte("updated_by_cli\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmdDiff := exec.Command("git", "-C", repo, "diff", "cli_test.txt")
	patchBytes, err := cmdDiff.CombinedOutput()
	if err != nil {
		t.Fatalf("diff: %v", err)
	}
	git("checkout", "cli_test.txt")

	patchFile := filepath.Join(t.TempDir(), "patch.diff")
	if err := os.WriteFile(patchFile, patchBytes, 0o600); err != nil {
		t.Fatal(err)
	}

	runsDir := filepath.Join(repo, ".dispatch-runs")

	// 1. Submit epilogue
	var stdout, stderr bytes.Buffer
	rc := runDispatchEpilogue(&stdout, &stderr, []string{
		"submit",
		"--issue", "12084",
		"--lane", "dispatch",
		"--path", "cli_test.txt",
		"-m", "fix(#12084): epilogue landing queue (fak dispatch)",
		"--patch", patchFile,
		"--runs-dir", runsDir,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("submit rc=%d stderr=%s", rc, stderr.String())
	}

	var rec dispatchtick.EpilogueRecord
	if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
		t.Fatalf("unmarshal submit: %v; raw=%s", err, stdout.String())
	}
	if rec.Issue != 12084 || rec.Status != dispatchtick.EpilogueStatusPending || rec.ID == "" {
		t.Fatalf("unexpected submitted record: %+v", rec)
	}

	// 2. List epilogues
	stdout.Reset()
	stderr.Reset()
	rc = runDispatchEpilogue(&stdout, &stderr, []string{
		"list",
		"--runs-dir", runsDir,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("list rc=%d stderr=%s", rc, stderr.String())
	}
	var list []dispatchtick.EpilogueRecord
	if err := json.Unmarshal(stdout.Bytes(), &list); err != nil {
		t.Fatalf("unmarshal list: %v", err)
	}
	if len(list) != 1 || list[0].ID != rec.ID {
		t.Fatalf("unexpected list output: %+v", list)
	}

	// Also test table list
	stdout.Reset()
	stderr.Reset()
	rc = runDispatchEpilogue(&stdout, &stderr, []string{
		"list",
		"--runs-dir", runsDir,
	})
	if rc != 0 || !strings.Contains(stdout.String(), rec.ID) {
		t.Fatalf("table list output failed: rc=%d stdout=%s", rc, stdout.String())
	}

	// 3. Status
	stdout.Reset()
	stderr.Reset()
	rc = runDispatchEpilogue(&stdout, &stderr, []string{
		"status",
		"--runs-dir", runsDir,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, stderr.String())
	}
	var statusMap map[string]int
	if err := json.Unmarshal(stdout.Bytes(), &statusMap); err != nil {
		t.Fatalf("unmarshal status: %v", err)
	}
	if statusMap["pending"] != 1 || statusMap["total"] != 1 {
		t.Fatalf("unexpected status: %+v", statusMap)
	}

	// 4. Drain epilogue
	stdout.Reset()
	stderr.Reset()
	rc = runDispatchEpilogue(&stdout, &stderr, []string{
		"drain",
		"--workspace", repo,
		"--runs-dir", runsDir,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, stderr.String())
	}
	var drainRes dispatchtick.EpilogueDrainResult
	if err := json.Unmarshal(stdout.Bytes(), &drainRes); err != nil {
		t.Fatalf("unmarshal drain: %v", err)
	}
	if drainRes.Landed != 1 || drainRes.Total != 1 {
		t.Fatalf("unexpected drain result: %+v", drainRes)
	}

	// Verify file content landed
	content, _ := os.ReadFile(filePath)
	if string(content) != "updated_by_cli\n" {
		t.Fatalf("file content = %q, want 'updated_by_cli\\n'", string(content))
	}

	// Verify status after drain
	stdout.Reset()
	stderr.Reset()
	rc = runDispatchEpilogue(&stdout, &stderr, []string{
		"status",
		"--runs-dir", runsDir,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, stderr.String())
	}
	if err := json.Unmarshal(stdout.Bytes(), &statusMap); err != nil {
		t.Fatal(err)
	}
	if statusMap["landed"] != 1 || statusMap["pending"] != 0 {
		t.Fatalf("unexpected status after drain: %+v", statusMap)
	}
}

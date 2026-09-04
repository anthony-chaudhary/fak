package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func TestWIPLatencyUsage(t *testing.T) {
	var out bytes.Buffer
	wipUsage(&out)
	if !strings.Contains(out.String(), "fak wip latency [--repo .] [--json] [--budget 1h]") {
		t.Fatalf("usage missing latency subcommand: %s", out.String())
	}
}

func TestWIPLatencyCLIExecution(t *testing.T) {
	repo := t.TempDir()
	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.com", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.com")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	runGit("init", "-q")
	if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "main.go")
	runGit("commit", "-qm", "feat: initial commit (fak test)")

	// 1. Test JSON output
	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"latency", "--json", "--repo", repo, "--budget", "1h"})
	if code != 0 {
		t.Fatalf("fak wip latency returned %d, stderr=%s", code, stderr.String())
	}

	var rep wipinventory.ProtectionLatencyReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to decode JSON report: %v\n%s", err, stdout.String())
	}

	if rep.Schema != wipinventory.ProtectionLatencySchema {
		t.Errorf("rep.Schema = %q, want %q", rep.Schema, wipinventory.ProtectionLatencySchema)
	}
	if rep.TotalSourcePaths != 1 {
		t.Errorf("TotalSourcePaths = %d, want 1", rep.TotalSourcePaths)
	}
	if rep.Outcomes["landed"] != 1 {
		t.Errorf("Outcomes[landed] = %d, want 1", rep.Outcomes["landed"])
	}
	if rep.SLOVerdict != "PASS" {
		t.Errorf("SLOVerdict = %q, want PASS", rep.SLOVerdict)
	}

	// 2. Test human summary output
	stdout.Reset()
	stderr.Reset()
	code = runWip(&stdout, &stderr, []string{"latency", "--repo", repo})
	if code != 0 {
		t.Fatalf("fak wip latency (human) returned %d, stderr=%s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "PROTECTION LATENCY — SLO verdict: PASS") {
		t.Errorf("output missing SLO verdict line: %s", outStr)
	}
	if !strings.Contains(outStr, "p50=") || !strings.Contains(outStr, "p95=") || !strings.Contains(outStr, "max=") {
		t.Errorf("output missing p50/p95/max: %s", outStr)
	}

	// 3. Test invalid budget flag
	stdout.Reset()
	stderr.Reset()
	code = runWip(&stdout, &stderr, []string{"latency", "--budget", "not-a-duration"})
	if code != 2 {
		t.Errorf("expected exit code 2 for invalid budget, got %d", code)
	}
}

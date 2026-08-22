package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokencache"
)

func TestDupCacheMaintainJSONAndHumanReceipts(t *testing.T) {
	root := initDupCacheRepo(t)
	cacheDir := tokencache.TokenCacheDir(filepath.Join(root, ".git"))
	c := tokencache.New(cacheDir, "v1")
	for i, src := range []string{"one", "two", "three"} {
		c.Put(src, []string{strings.Repeat("k", 80)}, [][2]int{{i + 1, i + 1}})
	}

	var stdout, stderr bytes.Buffer
	code := runDupCacheMaintain(&stdout, &stderr, []string{"--repo", root, "--max-bytes", "1", "--max-entries", "1", "--json"})
	if code != 0 {
		t.Fatalf("json command exit=%d stderr=%q", code, stderr.String())
	}
	var receipt tokencache.MaintenanceReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("invalid JSON receipt: %v\n%s", err, stdout.String())
	}
	if receipt.BeforeEntries != 3 || receipt.RemovedEntries != 3 || receipt.AfterEntries != 0 || receipt.Verdict != tokencache.VerdictPruned {
		t.Fatalf("JSON receipt = %+v", receipt)
	}

	stdout.Reset()
	stderr.Reset()
	code = runDupCacheMaintain(&stdout, &stderr, []string{"--repo", root, "--max-bytes", "1024", "--max-entries", "10"})
	if code != 0 {
		t.Fatalf("human command exit=%d stderr=%q", code, stderr.String())
	}
	for _, want := range []string{"before: 0 bytes / 0 entries", "removed: 0 entries", "stale temps: 0", "verdict: within_limits"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human receipt missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestDupCacheMaintainHonorsDisable(t *testing.T) {
	root := initDupCacheRepo(t)
	t.Setenv(tokencache.FlagEnv, "off")
	var stdout, stderr bytes.Buffer
	if code := runDupCacheMaintain(&stdout, &stderr, []string{"--repo", root, "--json"}); code != 0 {
		t.Fatalf("disabled command exit=%d stderr=%q", code, stderr.String())
	}
	var receipt tokencache.MaintenanceReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != tokencache.VerdictDisabled {
		t.Fatalf("disabled receipt = %+v", receipt)
	}
}

func initDupCacheRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	cmd := exec.Command("git", "init", "-q")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(tokencache.TokenCacheDir(filepath.Join(real, ".git")), 0o755); err != nil {
		t.Fatal(err)
	}
	return real
}

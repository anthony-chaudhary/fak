package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipreadiness"
)

func TestWIPReadinessJSONBindsCanonicalReceipt(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "README.md")
	runGit("commit", "-qm", "fixture")

	var stdout, stderr bytes.Buffer
	if code := runWip(&stdout, &stderr, []string{"readiness", "--json", "--root", repo, "--remote", ""}); code != 0 {
		t.Fatalf("runWIP readiness code=%d stderr=%s", code, stderr.String())
	}
	var receipt wipreadiness.Receipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode readiness JSON: %v\n%s", err, stdout.String())
	}
	if receipt.Schema != wipreadiness.Schema {
		t.Fatalf("schema=%q want %q", receipt.Schema, wipreadiness.Schema)
	}
	if receipt.Queue.ExpectedSchema != wipQueueSchema || receipt.Inventory.ExpectedSchema == "" || receipt.Lifecycle.ExpectedSchema == "" || receipt.Capacity.ExpectedSchema == "" {
		t.Fatalf("canonical source schemas not bound: %#v", receipt)
	}
	if receipt.ObservedAt.IsZero() || receipt.ExpiresAt.Sub(receipt.ObservedAt) != defaultWIPReadinessMaxAge {
		t.Fatalf("freshness window observed=%s expires=%s", receipt.ObservedAt, receipt.ExpiresAt)
	}
	if !receipt.EvidenceOnly {
		t.Fatal("readiness receipt must remain evidence-only")
	}
	if len(receipt.Hosts.Expected) != 1 || len(receipt.Hosts.Observed) != 1 || receipt.Hosts.Expected[0] != receipt.Hosts.Observed[0] {
		t.Fatalf("local host coverage=%#v", receipt.Hosts)
	}
	if receipt.Verdict != wipreadiness.VerdictCurrent {
		t.Fatalf("verdict=%s reasons=%v diagnostics=%v", receipt.Verdict, receipt.Reasons, receipt.Diagnostics)
	}
}

func TestWIPReadinessRejectsNegativeFreshness(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runWip(&stdout, &stderr, []string{"readiness", "--json", "--max-age", (-time.Second).String()}); code != 2 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

func TestRunWipAdmitRealGitUntrackedRefusalAndCleanAdmission(t *testing.T) {
	repo := initWipAdmitRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "new.txt"), []byte("work\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{"admit", "-C", repo, "--session", "session-a", "--path", "new.txt", "--json"})
	if code != 3 {
		t.Fatalf("untracked code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var refused wipattr.AdmitReport
	if err := json.Unmarshal(out.Bytes(), &refused); err != nil {
		t.Fatal(err)
	}
	if refused.Verdict != wipattr.AdmitHold || !admitHasReason(refused, wipattr.ReasonPathUntrackedWIP) {
		t.Fatalf("report=%+v", refused)
	}

	if err := os.Remove(filepath.Join(repo, "new.txt")); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	code = runWip(&out, &errOut, []string{"admit", "-C", repo, "--session", "session-a", "--json"})
	if code != 0 {
		t.Fatalf("clean code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var clean wipattr.AdmitReport
	if err := json.Unmarshal(out.Bytes(), &clean); err != nil {
		t.Fatal(err)
	}
	if clean.Verdict != wipattr.AdmitOK {
		t.Fatalf("clean report=%+v", clean)
	}
}

func TestRunWipAdmitRequiresSession(t *testing.T) {
	t.Setenv("CLAUDE_CODE_SESSION_ID", "")
	t.Setenv("FAK_SESSION_ID", "")
	var out, errOut bytes.Buffer
	if code := runWip(&out, &errOut, []string{"admit", "-C", initWipAdmitRepo(t)}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func admitHasReason(rep wipattr.AdmitReport, reason wipattr.AdmitReason) bool {
	for _, finding := range rep.Findings {
		if finding.Reason == reason {
			return true
		}
	}
	return false
}

func initWipAdmitRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"config", "user.email", "test@example.com"}, {"config", "user.name", "Test"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"add", "base.txt"}, {"commit", "-qm", "base"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	return repo
}

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncCheckPublicLeakRemediationSpine(t *testing.T) {
	oldWorktree := syncWorktree
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, true }
	t.Cleanup(func() { syncWorktree = oldWorktree })

	t.Run("eight deterministic findings, targeted repair, and bound resume", func(t *testing.T) {
		repo := t.TempDir()
		syncGit(t, repo, "init", "-b", "main")
		syncGit(t, repo, "config", "user.name", "test")
		syncGit(t, repo, "config", "user.email", "test@example.com")

		syncPublicLeakWriteFile(t, filepath.Join(repo, "docs", "inherited.txt"), syncPublicLeakFixtureLines(1, 2))
		syncPublicLeakWriteFile(t, filepath.Join(repo, "docs", "worktree.txt"), "clean\n")
		syncGit(t, repo, "add", ".")
		syncGit(t, repo, "commit", "-m", "baseline")
		syncGit(t, repo, "update-ref", "refs/remotes/origin/main", "HEAD")

		syncPublicLeakWriteFile(t, filepath.Join(repo, "cmd", "fak", "introduced.txt"), syncPublicLeakFixtureLines(3, 4))
		syncPublicLeakWriteFile(t, filepath.Join(repo, "internal", "hooks", "introduced.txt"), syncPublicLeakFixtureLines(5, 6))
		syncGit(t, repo, "add", ".")
		syncGit(t, repo, "commit", "-m", "candidate")
		syncPublicLeakWriteFile(t, filepath.Join(repo, "docs", "worktree.txt"), syncPublicLeakFixtureLines(7, 8))

		firstCode, firstOut, firstErr := runSyncPublicLeakTest(t, repo, "--json")
		secondCode, secondOut, secondErr := runSyncPublicLeakTest(t, repo, "--json")
		if firstCode != syncExitRefused || secondCode != syncExitRefused {
			t.Fatalf("exit = %d/%d, want refused; stderr=%q/%q", firstCode, secondCode, firstErr, secondErr)
		}
		if firstOut != secondOut {
			t.Fatalf("PUBLIC_LEAK JSON is not deterministic:\n--- first ---\n%s\n--- second ---\n%s", firstOut, secondOut)
		}

		var report syncCheckReport
		if err := json.Unmarshal([]byte(firstOut), &report); err != nil {
			t.Fatalf("decode sync report: %v\n%s", err, firstOut)
		}
		leak := report.PublicLeak
		if leak == nil {
			t.Fatalf("report omitted public_leak: %s", firstOut)
		}
		if leak.Count != 8 || leak.IntroducedCount != 4 || leak.InheritedCount != 2 || leak.UnknownCount != 2 || leak.BlockingCount != 6 || leak.OK {
			t.Fatalf("PUBLIC_LEAK counts = %+v, want 8 total / 4 introduced / 2 inherited / 2 unknown / 6 blocking", leak)
		}

		ids := map[string]bool{}
		lastSortKey := ""
		for _, finding := range leak.Findings {
			if finding.ID == "" || ids[finding.ID] {
				t.Fatalf("finding IDs are not stable and unique: %+v", leak.Findings)
			}
			ids[finding.ID] = true
			if finding.Gate != syncPublicLeakGate || finding.Path == "" {
				t.Fatalf("finding lacks gate/path contract: %+v", finding)
			}
			sortKey := fmt.Sprintf("%s\x00%09d\x00%s", finding.Path, finding.Line, finding.Detail)
			if sortKey < lastSortKey {
				t.Fatalf("findings are not stably sorted: %+v", leak.Findings)
			}
			lastSortKey = sortKey
			switch finding.Provenance {
			case "introduced":
				if !finding.Blocking || !finding.Attributive {
					t.Fatalf("introduced finding policy = %+v, want blocking and attributive", finding)
				}
			case "inherited":
				if finding.Blocking || finding.Attributive {
					t.Fatalf("inherited finding policy = %+v, want visible but non-blocking/non-attributive", finding)
				}
			case "unknown":
				if !finding.Blocking || finding.Attributive {
					t.Fatalf("unknown finding policy = %+v, want fail-closed without attribution", finding)
				}
			default:
				t.Fatalf("unexpected provenance %q", finding.Provenance)
			}
		}

		wantActionable := []string{"cmd/fak/introduced.txt", "docs/worktree.txt", "internal/hooks/introduced.txt"}
		var repairPaths []string
		seenRepair := map[string]bool{}
		for _, repair := range leak.RepairSlices {
			if repair.ID == "" || repair.LaneResolution == "" {
				t.Fatalf("repair slice lacks stable ID/lane resolution: %+v", repair)
			}
			for _, path := range repair.Paths {
				if seenRepair[path] {
					t.Fatalf("repair slices collide on %q: %+v", path, leak.RepairSlices)
				}
				seenRepair[path] = true
				repairPaths = append(repairPaths, path)
			}
		}
		sort.Strings(repairPaths)
		if strings.Join(repairPaths, ",") != strings.Join(wantActionable, ",") {
			t.Fatalf("repair paths = %v, want %v", repairPaths, wantActionable)
		}
		for _, want := range wantActionable {
			if !strings.Contains(leak.TargetedRecheck, "--recheck-path "+want) {
				t.Fatalf("targeted recheck omitted %q: %s", want, leak.TargetedRecheck)
			}
		}
		if strings.Contains(leak.TargetedRecheck, "docs/inherited.txt") {
			t.Fatalf("targeted recheck assigned inherited debt to the candidate: %s", leak.TargetedRecheck)
		}
		if leak.ResumeToken == "" || !strings.Contains(leak.ResumeCommand, "--resume-token "+leak.ResumeToken) {
			t.Fatalf("resume action is not token-bound: token=%q command=%q", leak.ResumeToken, leak.ResumeCommand)
		}

		badToken := leak.ResumeToken[:len(leak.ResumeToken)-1] + "x"
		badCode, badOut, badErr := runSyncPublicLeakTest(t, repo, "--resume-token", badToken, "--json")
		if badCode != syncExitRefused || badOut != "" || !strings.Contains(badErr, "does not match this repo, branch, HEAD, and remote target") {
			t.Fatalf("bad resume token: exit=%d stdout=%q stderr=%q", badCode, badOut, badErr)
		}

		validCode, validOut, validErr := runSyncPublicLeakTest(t, repo, "--resume-token", leak.ResumeToken, "--json")
		if validCode != syncExitRefused || validErr != "" {
			t.Fatalf("valid resume with live findings: exit=%d stderr=%q", validCode, validErr)
		}
		var validReport syncCheckReport
		if err := json.Unmarshal([]byte(validOut), &validReport); err != nil || validReport.PublicLeak == nil || validReport.PublicLeak.BlockingCount != 6 {
			t.Fatalf("resume bypassed the gate: err=%v report=%+v", err, validReport.PublicLeak)
		}

		for _, path := range wantActionable {
			syncPublicLeakWriteFile(t, filepath.Join(repo, filepath.FromSlash(path)), "repaired\n")
		}
		recheckArgs := []string{"--json"}
		for _, path := range wantActionable {
			recheckArgs = append(recheckArgs, "--recheck-path", path)
		}
		recheckCode, recheckOut, recheckErr := runSyncPublicLeakTest(t, repo, recheckArgs...)
		if recheckCode != syncExitOK || recheckErr != "" {
			t.Fatalf("targeted recheck: exit=%d stderr=%q stdout=%s", recheckCode, recheckErr, recheckOut)
		}
		var recheckReport syncCheckReport
		if err := json.Unmarshal([]byte(recheckOut), &recheckReport); err != nil || recheckReport.PublicLeak == nil || recheckReport.PublicLeak.Count != 0 || !recheckReport.PublicLeak.OK {
			t.Fatalf("targeted recheck did not witness clean paths: err=%v report=%+v", err, recheckReport.PublicLeak)
		}

		resumeCode, resumeOut, resumeErr := runSyncPublicLeakTest(t, repo, "--resume-token", leak.ResumeToken, "--json")
		if resumeCode != syncExitRefused || resumeErr != "" { // The original operation is still an ahead check.
			t.Fatalf("resumed operation: exit=%d stderr=%q stdout=%s", resumeCode, resumeErr, resumeOut)
		}
		var resumed syncCheckReport
		if err := json.Unmarshal([]byte(resumeOut), &resumed); err != nil || resumed.PublicLeak == nil {
			t.Fatalf("decode resumed operation: err=%v report=%s", err, resumeOut)
		}
		if resumed.PublicLeak.BlockingCount != 0 || resumed.PublicLeak.InheritedCount != 2 || !resumed.PublicLeak.OK {
			t.Fatalf("resumed operation did not rerun the whole gate: %+v", resumed.PublicLeak)
		}
		if !resumed.PublicLeak.ResumeValidated {
			t.Fatalf("resumed operation did not expose token validation: %+v", resumed.PublicLeak)
		}
	})

	t.Run("clean candidate preserves quiet output", func(t *testing.T) {
		repo := t.TempDir()
		syncGit(t, repo, "init", "-b", "main")
		syncGit(t, repo, "config", "user.name", "test")
		syncGit(t, repo, "config", "user.email", "test@example.com")
		syncWriteFile(t, filepath.Join(repo, "clean.txt"), "clean\n")
		syncGit(t, repo, "add", ".")
		syncGit(t, repo, "commit", "-m", "clean")
		syncGit(t, repo, "update-ref", "refs/remotes/origin/main", "HEAD")

		code, out, errOut := runSyncPublicLeakTest(t, repo)
		if code != syncExitOK || errOut != "" {
			t.Fatalf("clean check: exit=%d stderr=%q stdout=%q", code, errOut, out)
		}
		if out != "in sync: local branch already matches the remote; nothing to do\n" {
			t.Fatalf("clean candidate became noisy: %q", out)
		}

		code, out, errOut = runSyncPublicLeakTest(t, repo, "--json")
		if code != syncExitOK || errOut != "" {
			t.Fatalf("clean JSON check: exit=%d stderr=%q stdout=%q", code, errOut, out)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(out), &raw); err != nil {
			t.Fatalf("decode clean JSON: %v", err)
		}
		if _, noisy := raw["public_leak"]; noisy {
			t.Fatalf("clean JSON contains noisy public_leak envelope: %s", out)
		}
	})
}

func syncPublicLeakFixtureLines(numbers ...int) string {
	var b strings.Builder
	for _, number := range numbers {
		fmt.Fprintf(&b, "host=lab-%s%d\n", "dgx", number)
	}
	return b.String()
}

func syncPublicLeakWriteFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	syncWriteFile(t, path, text)
}

func runSyncPublicLeakTest(t *testing.T, repo string, extra ...string) (int, string, string) {
	t.Helper()
	args := []string{"check", "--repo", repo, "--remote", "origin", "--branch", "main"}
	args = append(args, extra...)
	var stdout, stderr bytes.Buffer
	code := runSync(&stdout, &stderr, args)
	return code, stdout.String(), stderr.String()
}

func TestSyncPublicLeakCRLFBaselineInherited(t *testing.T) {
	repo := t.TempDir()
	syncGit(t, repo, "init", "-b", "main")
	syncGit(t, repo, "config", "user.name", "test")
	syncGit(t, repo, "config", "user.email", "test@example.com")

	filePath := filepath.Join("docs", "inherited.txt")
	syncPublicLeakWriteFile(t, filepath.Join(repo, filePath), syncPublicLeakFixtureLines(1))
	syncGit(t, repo, "add", ".")
	syncGit(t, repo, "commit", "-m", "baseline")
	syncGit(t, repo, "update-ref", "refs/remotes/origin/main", "HEAD")

	finding := hooks.Finding{
		Gate:   syncPublicLeakGate,
		File:   filePath,
		Line:   1,
		Detail: "host=lab-dgx1",
	}

	// Working tree has CRLF line endings matching baseline LF logical content.
	crlfContent := strings.ReplaceAll(syncPublicLeakFixtureLines(1), "\n", "\r\n")
	syncPublicLeakWriteFile(t, filepath.Join(repo, filePath), crlfContent)

	if !syncPublicLeakExistsAtBaseline(repo, "HEAD", finding) {
		t.Fatalf("syncPublicLeakExistsAtBaseline() = false, want true for CRLF working tree matching LF baseline")
	}

	info := safesync.Assessment{
		Branch: "main",
		Head:   syncRev(t, repo, "HEAD"),
		Target: syncRev(t, repo, "HEAD"),
	}
	report, err := assessSyncPublicLeak(repo, "origin", info, nil)
	if err != nil {
		t.Fatalf("assessSyncPublicLeak failed: %v", err)
	}
	if report.InheritedCount != 1 || report.BlockingCount != 0 || !report.OK {
		t.Fatalf("assessSyncPublicLeak report = %+v, want 1 inherited, 0 blocking, ok=true", report)
	}
	if len(report.Findings) != 1 || report.Findings[0].Provenance != "inherited" || report.Findings[0].Blocking {
		t.Fatalf("finding = %+v, want provenance=inherited blocking=false", report.Findings)
	}
}

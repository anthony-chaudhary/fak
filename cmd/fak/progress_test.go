package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRunProgressReportsDeliveredWIPAndIssues(t *testing.T) {
	oldCmd, oldNow := progressCommand, progressNow
	t.Cleanup(func() { progressCommand = oldCmd; progressNow = oldNow })
	progressNow = func() time.Time { return time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC) }
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(key, "git rev-list") && strings.Contains(key, "--until="):
			return []byte("4\n"), nil
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("2\n"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M tracked\x00?? new\x00A  staged\x00"), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return []byte("4\t2\ttracked\x001\t0\tstaged\x00"), nil
		case strings.Contains(key, "--state closed") && strings.Contains(key, ".."):
			return []byte(`[{"number":8},{"number":9}]`), nil
		case strings.Contains(key, "--state closed"):
			return []byte(`[{"number":1}]`), nil
		case strings.Contains(key, "--state open"):
			return []byte(`[{"number":2},{"number":3}]`), nil
		}
		t.Fatalf("unexpected %s", key)
		return nil, nil
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var got progressReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PROGRESS" || got.Commits != 2 || got.CommitsPer10M != 2 || got.WIP.Files != 3 || got.WIP.Untracked != 1 || got.WIP.Staged != 1 || got.WIP.Unstaged != 1 || !got.GitHub.Available || got.GitHub.RecentlyClosed != 1 || got.GitHub.OpenTotal != 2 || got.Baseline.Commits != 4 || got.Baseline.IssuesClosed != 2 || got.Baseline.CommitsPer10M != 2 || got.Baseline.IssueClosesPer10M != 1 || got.IssueClosesPer10M != 1 {
		t.Fatalf("unexpected report: %+v", got)
	}
}

func TestRunProgressDoesNotCallWIPProgressAndFailsSoftWithoutGitHub(t *testing.T) {
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		if strings.HasPrefix(key, "git rev-list") {
			return []byte("0"), nil
		}
		if strings.HasPrefix(key, "git status") {
			return []byte(" M tracked\x00"), nil
		}
		if strings.HasPrefix(key, "git diff --numstat") {
			return nil, nil
		}
		return nil, errors.New("offline")
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "10m"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	s := out.String()
	for _, want := range []string{"WIP_ONLY", "commits=0", "files=1", "GitHub: unavailable"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in %s", want, s)
		}
	}
}

func TestParseProgressWIPCountsRenameOnce(t *testing.T) {
	var r progressReport
	paths := parseProgressWIP([]byte("R  new\x00old\x00UU conflict\x00"), &r)
	if r.WIP.Files != 2 || r.WIP.Staged != 2 || r.WIP.Conflicts != 1 || len(paths) != 2 {
		t.Fatalf("unexpected WIP: %+v", r.WIP)
	}
}

func TestRunProgressOneQuietWindowDoesNotOverAlert(t *testing.T) {
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(key, "git rev-list") && strings.Contains(key, "--until="):
			return []byte("2"), nil
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("0"), nil
		case strings.HasPrefix(key, "git status"):
			return nil, nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return nil, nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, errors.New("unexpected " + key)
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "20m"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "QUIET") || strings.Contains(out.String(), "STALLED") {
		t.Fatalf("one quiet window over-alerted: %s", out.String())
	}
}

func TestRunProgressStalledAfterDeclaredConsecutiveWindows(t *testing.T) {
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("0"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M unfinished\x00"), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return nil, nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, errors.New("unexpected " + key)
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "20m", "--stall-after", "3", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var got progressReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "STALLED" || got.StallAfterWindows != 3 || got.WIP.Files != 1 {
		t.Fatalf("unexpected report: %+v", got)
	}
}

func TestCollectProgressWIPDetailsMeasuresMagnitudeAgeAndUntrackedBytes(t *testing.T) {
	dir := t.TempDir()
	tracked := filepath.Join(dir, "tracked.txt")
	untracked := filepath.Join(dir, "new.bin")
	if err := os.WriteFile(tracked, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(untracked, []byte{1, 2, 3, 4, 5}, 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Date(2026, 8, 17, 10, 0, 0, 0, time.UTC)
	if err := os.Chtimes(tracked, old, old); err != nil {
		t.Fatal(err)
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		if name == "git" && len(args) > 1 && args[0] == "diff" {
			return []byte("7\t3\ttracked.txt\x00-\t-\timage.png\x00"), nil
		}
		return nil, errors.New("unexpected command")
	}
	var r progressReport
	err := collectProgressWIPDetails(dir, []progressWIPPath{{path: "tracked.txt"}, {path: "new.bin", untracked: true}, {path: "missing.txt", untracked: true}}, old.Add(95*time.Minute), &r)
	if err != nil {
		t.Fatal(err)
	}
	if r.WIP.Additions != 7 || r.WIP.Deletions != 3 || r.WIP.BinaryFiles != 1 || r.WIP.UntrackedBytes != 5 || r.WIP.OldestExistingMinutes != 95 || r.WIP.AgeFilesObserved != 2 || r.WIP.AgeFilesUnavailable != 1 {
		t.Fatalf("unexpected details: %+v", r.WIP)
	}
}

func TestCollectProgressWIPDetailsDoesNotReadOutsideRepository(t *testing.T) {
	dir := t.TempDir()
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, _ string, _ ...string) ([]byte, error) { return nil, nil }
	var r progressReport
	if err := collectProgressWIPDetails(dir, []progressWIPPath{{path: "../outside"}}, time.Now(), &r); err != nil {
		t.Fatal(err)
	}
	if r.WIP.AgeFilesUnavailable != 1 || r.WIP.AgeFilesObserved != 0 {
		t.Fatalf("outside path was observed: %+v", r.WIP)
	}
}

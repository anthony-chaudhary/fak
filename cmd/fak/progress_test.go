package main

import (
	"bytes"
	"encoding/json"
	"errors"
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
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("2\n"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M tracked\x00?? new\x00A  staged\x00"), nil
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
	if got.Verdict != "PROGRESS" || got.Commits != 2 || got.CommitsPer10M != 2 || got.WIP.Files != 3 || got.WIP.Untracked != 1 || got.WIP.Staged != 1 || got.WIP.Unstaged != 1 || !got.GitHub.Available || got.GitHub.RecentlyClosed != 1 || got.GitHub.OpenTotal != 2 {
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
		return nil, errors.New("offline")
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, nil); code != 0 {
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
	parseProgressWIP([]byte("R  new\x00old\x00UU conflict\x00"), &r)
	if r.WIP.Files != 2 || r.WIP.Staged != 2 || r.WIP.Conflicts != 1 {
		t.Fatalf("unexpected WIP: %+v", r.WIP)
	}
}

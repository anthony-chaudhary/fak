package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunGHSpamCommentsDryRunFromFixture(t *testing.T) {
	path := writeGHSpamFixture(t, `[
		{
			"id": 4885499913,
			"node_id": "IC_patch",
			"html_url": "https://github.com/anthony-chaudhary/fak/issues/2752#issuecomment-4885499913",
			"user": {"login": "cosuwacu"},
			"author_association": "NONE",
			"created_at": "2026-07-05T09:11:50Z",
			"body": "(https://github.com/bareneguboko/patch_fix/releases/download/release/patch_fix.rar)"
		},
		{
			"id": 2,
			"node_id": "IC_owner",
			"html_url": "https://github.com/anthony-chaudhary/fak/issues/1#issuecomment-2",
			"user": {"login": "anthony-chaudhary"},
			"author_association": "OWNER",
			"created_at": "2026-07-05T09:12:00Z",
			"body": "https://github.com/owner/repo/releases/download/v1/tool.zip"
		}
	]`)

	var stdout, stderr bytes.Buffer
	code := runGHSpamCommentsWith(&stdout, &stderr, []string{"--comments-json", path, "--json"}, func(args []string) ([]byte, error) {
		t.Fatalf("runner should not be called in fixture dry-run, got %v", args)
		return nil, nil
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var got struct {
		Counts struct {
			Scanned        int `json:"scanned"`
			TrustedSkipped int `json:"trusted_skipped"`
			Matched        int `json:"matched"`
		} `json:"counts"`
		Findings []struct {
			User       string `json:"user"`
			ArchiveURL string `json:"archive_url"`
		} `json:"findings"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if got.Counts.Scanned != 2 || got.Counts.TrustedSkipped != 1 || got.Counts.Matched != 1 {
		t.Fatalf("counts = %+v", got.Counts)
	}
	if len(got.Findings) != 1 || got.Findings[0].User != "cosuwacu" || !strings.HasSuffix(got.Findings[0].ArchiveURL, "patch_fix.rar") {
		t.Fatalf("findings = %+v", got.Findings)
	}
}

func TestRunGHSpamCommentsApplyMinimizesMatches(t *testing.T) {
	path := writeGHSpamFixture(t, `[{
		"id": 1,
		"node_id": "IC_spam",
		"html_url": "https://github.com/o/r/issues/1#issuecomment-1",
		"user": {"login": "outsider"},
		"author_association": "NONE",
		"created_at": "2026-07-05T09:11:50Z",
		"body": "https://github.com/x/y/releases/download/release/critical_patch_2026.zip"
	}]`)
	var calls [][]string
	runner := func(args []string) ([]byte, error) {
		calls = append(calls, append([]string(nil), args...))
		if len(args) >= 4 && args[0] == "api" && args[1] == "graphql" && strings.Contains(strings.Join(args, " "), "minimizeComment") {
			return []byte(`{"data":{"minimizeComment":{"minimizedComment":{"isMinimized":true,"minimizedReason":"spam"}}}}`), nil
		}
		t.Fatalf("unexpected runner call: %v", args)
		return nil, nil
	}

	var stdout, stderr bytes.Buffer
	code := runGHSpamCommentsWith(&stdout, &stderr, []string{"--comments-json", path, "--apply", "--json"}, runner)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if len(calls) != 1 {
		t.Fatalf("calls = %v, want one graphql call", calls)
	}
	joined := strings.Join(calls[0], " ")
	if !strings.Contains(joined, "id=IC_spam") || !strings.Contains(joined, "classifier:SPAM") {
		t.Fatalf("graphql args = %v", calls[0])
	}
	if !strings.Contains(stdout.String(), `"applied": 1`) {
		t.Fatalf("stdout missing applied count:\n%s", stdout.String())
	}
}

func TestRepoFromRemoteURL(t *testing.T) {
	cases := map[string]string{
		"https://github.com/anthony-chaudhary/fak.git": "anthony-chaudhary/fak",
		"git@github.com:anthony-chaudhary/fak.git":     "anthony-chaudhary/fak",
		"ssh://git@github.com/owner/repo.git":          "owner/repo",
		"https://example.com/owner/repo.git":           "",
	}
	for in, want := range cases {
		if got := repoFromRemoteURL(in); got != want {
			t.Fatalf("repoFromRemoteURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func writeGHSpamFixture(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "comments.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

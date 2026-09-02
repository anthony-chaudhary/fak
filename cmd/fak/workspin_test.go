package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestRunWorkspinJSONExposesDeterministicBucketsAndReasons(t *testing.T) {
	dir := t.TempDir()
	runGit := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.com", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.com")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	runGit("init", "-q")
	for i, date := range []string{"2026-08-20T12:00:00Z", "2026-08-21T12:00:00Z", "2026-08-22T12:00:00Z", "2026-08-27T12:00:00Z", "2026-08-28T12:00:00Z", "2026-08-29T12:00:00Z"} {
		p := filepath.Join(dir, "f")
		if err := os.WriteFile(p, bytes.Repeat([]byte("x"), i+1), 0o644); err != nil {
			t.Fatal(err)
		}
		runGit("add", "f")
		cmd := exec.Command("git", "-C", dir, "commit", "-q", "-m", "chore: small cleanup")
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.com", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.com", "GIT_AUTHOR_DATE="+date, "GIT_COMMITTER_DATE="+date)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("commit: %v: %s", err, out)
		}
	}
	var stdout, stderr bytes.Buffer
	code := runWorkspin(&stdout, &stderr, []string{"-repo", dir, "-now", "2026-09-01T00:00:00Z", "-json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Schema, Verdict string
		Spinning        bool
		Reasons         []string
		Buckets         []struct{ Activity, Delivery int }
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-workspin/1" || got.Verdict != "spinning" || !got.Spinning {
		t.Fatalf("payload=%s", stdout.String())
	}
	if len(got.Buckets) != 4 || len(got.Reasons) == 0 {
		t.Fatalf("missing buckets/reasons: %s", stdout.String())
	}
	if got.Buckets[2].Activity != 3 || got.Buckets[3].Activity != 3 {
		t.Fatalf("unexpected buckets: %s", stdout.String())
	}
}

package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunWipEvictCLI(t *testing.T) {
	repo := t.TempDir()
	runGit := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test",
			"GIT_AUTHOR_EMAIL=test@fak.local",
			"GIT_COMMITTER_NAME=test",
			"GIT_COMMITTER_EMAIL=test@fak.local",
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	runGit("init", "-q", "-b", "main")
	runGit("config", "user.name", "test")
	runGit("config", "user.email", "test@fak.local")

	base := filepath.Join(repo, "base.txt")
	if err := os.WriteFile(base, []byte("base"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "base.txt")
	runGit("commit", "-q", "-m", "init")

	orphan := filepath.Join(repo, "orphan.txt")
	if err := os.WriteFile(orphan, []byte("orphan file content"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runWipEvict(&stdout, &stderr, []string{"-C", repo, "--session", "test-cli-sess", "--json"})
	if code != 0 {
		t.Fatalf("runWipEvict exited %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"ref": "refs/fak/quarantine/test-cli-sess/`) {
		t.Fatalf("json missing quarantine ref: %s", out)
	}
	if !strings.Contains(out, `"count": 1`) {
		t.Fatalf("json missing count: %s", out)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("orphan file should have been removed: %v", err)
	}
}

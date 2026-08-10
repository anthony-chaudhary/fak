package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunWhatsChangedTextAndJSON(t *testing.T) {
	repo, base := whatsChangedFixture(t)
	var out, errOut bytes.Buffer
	if code := RunWhatsChanged(&out, &errOut, []string{"--dir", repo, "--since", base, "--paths", "cmd/fak/*.go"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"fak-dev whats-changed:", "peer change", "cmd/fak/main.go"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("text output missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errOut.Reset()
	if code := RunWhatsChanged(&out, &errOut, []string{"--dir", repo, "--since", base, "--paths", "cmd/fak/*.go", "--json"}); code != 0 {
		t.Fatalf("json code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Commits      []any    `json:"commits"`
		ChangedFiles []string `json:"changed_files"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Commits) != 1 || len(got.ChangedFiles) != 1 || got.ChangedFiles[0] != "cmd/fak/main.go" {
		t.Fatalf("unexpected JSON report: %+v", got)
	}
}

func TestRunWhatsChangedRequiresPaths(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunWhatsChanged(&out, &errOut, []string{"--dir", t.TempDir(), "--since", "HEAD"}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "--paths is required") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func whatsChangedFixture(t *testing.T) (string, string) {
	t.Helper()
	repo := t.TempDir()
	runGitWhatsChanged(t, repo, "init")
	runGitWhatsChanged(t, repo, "config", "user.email", "test@example.com")
	runGitWhatsChanged(t, repo, "config", "user.name", "Test")
	writeWhatsChangedFile(t, repo, "cmd/fak/main.go", "package main\n")
	runGitWhatsChanged(t, repo, "add", ".")
	runGitWhatsChanged(t, repo, "commit", "-m", "base")
	base := strings.TrimSpace(runGitWhatsChanged(t, repo, "rev-parse", "HEAD"))
	writeWhatsChangedFile(t, repo, "cmd/fak/main.go", "package main\n// peer\n")
	runGitWhatsChanged(t, repo, "add", ".")
	runGitWhatsChanged(t, repo, "commit", "-m", "peer change")
	return repo, base
}

func runGitWhatsChanged(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, b)
	}
	return string(b)
}

func writeWhatsChangedFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

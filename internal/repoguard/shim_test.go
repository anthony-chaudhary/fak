package repoguard_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestRepoGuardHookFallsBackFromStaleBinary(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", ".claude", "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Hooks struct {
			PreToolUse []struct {
				Hooks []struct {
					Args []string `json:"args"`
				} `json:"hooks"`
			} `json:"PreToolUse"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	script := cfg.Hooks.PreToolUse[0].Hooks[0].Args[1]

	root := t.TempDir()
	for _, dir := range []string{"tools/.bin", "cmd/repoguard", "internal/repoguard"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(dir)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	marker := filepath.Join(root, "fallback-ran")
	fallback := "import pathlib; pathlib.Path(" + pythonQuote(marker) + ").write_text('yes')\n"
	if err := os.WriteFile(filepath.Join(root, "tools", "repo_guard.py"), []byte(fallback), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "repoguard", "severity.go"), []byte("package repoguard\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(root, "tools", ".bin", "repoguard.exe")
	if runtime.GOOS != "windows" {
		binary = filepath.Join(root, "tools", ".bin", "repoguard.exe")
	}
	if err := os.WriteFile(binary, []byte("stale-not-executable"), 0o755); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(binary, old, old); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(pythonExe(t), "-c", script)
	cmd.Env = append(os.Environ(), "CLAUDE_PROJECT_DIR="+root)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("shim: %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stderr.String(), "repoguard binary is stale -- rebuild") {
		t.Fatalf("missing stale warning: %q", stderr.String())
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("source fallback did not run: %v", err)
	}
}

func pythonExe(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python", "python3"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	t.Skip("python unavailable")
	return ""
}
func pythonQuote(s string) string { return "r'" + strings.ReplaceAll(s, "'", "\\'") + "'" }

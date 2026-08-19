package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchAttachOpencodePromptFileKeepsPromptOutOfArgv(t *testing.T) {
	cmd := []string{"opencode", "run", "--print-logs", "-m", "glm", dispatchtick.OpencodePromptNotice}
	got := dispatchAttachOpencodePromptFile(cmd, `C:\work\fak\.dispatch-runs\resolve-2506.prompt.txt`)
	want := []string{
		"opencode", "run", "--print-logs", "-m", "glm",
		"--file", `C:\work\fak\.dispatch-runs\resolve-2506.prompt.txt`,
		"--",
		dispatchtick.OpencodePromptNotice,
	}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("attached command = %#v, want %#v", got, want)
	}
	if strings.Contains(strings.Join(got, " "), "your goal: resolve GitHub issue") {
		t.Fatalf("attached command leaked full dispatch prompt into argv: %#v", got)
	}
}

func TestUnwrapOpencodeNpmShimTargetsRealExecutable(t *testing.T) {
	npm := t.TempDir()
	real := filepath.Join(npm, "node_modules", "opencode-ai", "bin", "opencode.exe")
	if err := os.MkdirAll(filepath.Dir(real), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(real, []byte("fake exe"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := unwrapOpencodeNpmShim(filepath.Join(npm, "opencode.cmd"))
	if got != real {
		t.Fatalf("unwrap target = %q, want %q", got, real)
	}
	if got := unwrapOpencodeNpmShim(filepath.Join(npm, "claude.cmd")); got != "" {
		t.Fatalf("non-opencode shim unwrapped to %q", got)
	}
}

func TestUnwrapCodexNpmShim(t *testing.T) {
	npm := t.TempDir()
	target := filepath.Join(npm, "node_modules", "@openai", "codex", "node_modules", "@openai", "codex-win32-x64", "vendor", "x86_64-pc-windows-msvc", "bin", "codex.exe")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("native"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := unwrapCodexNpmShim(filepath.Join(npm, "codex.cmd"))
	if got != target {
		t.Fatalf("unwrap codex = %q, want %q", got, target)
	}
	if got := unwrapCodexNpmShim(filepath.Join(npm, "opencode.cmd")); got != "" {
		t.Fatalf("unrelated shim resolved to %q", got)
	}
}

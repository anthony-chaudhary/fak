package main

import (
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

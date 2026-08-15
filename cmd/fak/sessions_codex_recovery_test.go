package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCodexProjectHookBreakGlassWorksWithoutFak(t *testing.T) {
	h := loadCodexProjectHook(t)
	cmd := codexProjectHookCommand(t, h)
	cmd.Env = append(os.Environ(),
		codexRawRecoveryEnv+"="+codexRawRecoveryValue,
		"PATH="+t.TempDir(),
	)
	cmd.Stdin = strings.NewReader("{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"repair fak\"}\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("break-glass hook without fak: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("break-glass stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), codexRawRecoveryWarning) {
		t.Fatalf("break-glass stderr = %q, want warning %q", stderr.String(), codexRawRecoveryWarning)
	}
}

func TestCodexProjectHookRequiresExactBreakGlassValue(t *testing.T) {
	h := loadCodexProjectHook(t)
	fakeDir := t.TempDir()
	writeCodexHookFakeFak(t, fakeDir, false)
	cmd := codexProjectHookCommand(t, h)
	cmd.Env = append(os.Environ(),
		codexRawRecoveryEnv+"=1",
		"PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"),
	)
	cmd.Stdin = strings.NewReader("{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"ordinary raw turn\"}\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("non-exact recovery value: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"decision":"block"`) {
		t.Fatalf("non-exact recovery stdout = %q, want fak block", stdout.String())
	}
	if strings.Contains(stderr.String(), codexRawRecoveryWarning) {
		t.Fatalf("non-exact recovery value activated warning: %q", stderr.String())
	}
}

func TestCodexHookInstallerEmitsBreakGlassBeforeFak(t *testing.T) {
	home := t.TempDir()
	if code := sessionsCodexHookInstall(&bytes.Buffer{}, &bytes.Buffer{}, []string{"--codex-home", home}); code != 0 {
		t.Fatalf("install exit = %d", code)
	}
	raw, err := os.ReadFile(filepath.Join(home, "hooks.json"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{codexRawRecoveryEnv, codexRawRecoveryValue, codexRawRecoveryWarning} {
		if !strings.Contains(text, want) {
			t.Fatalf("installed hook missing %q:\n%s", want, text)
		}
	}
	if recovery, fak := strings.Index(text, codexRawRecoveryEnv), strings.Index(text, "fak sessions codex-loop-hook"); recovery < 0 || fak < 0 || recovery > fak {
		t.Fatalf("recovery must be evaluated before fak: recovery=%d fak=%d\n%s", recovery, fak, text)
	}
}

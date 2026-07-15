package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGuardSeedPromptUsesClaudeFileFlagBeyondWindowsArgvLimit(t *testing.T) {
	seed := strings.Repeat("continuity ", 5000) // >32 KiB source; bounded before publication.
	got, handback, injected := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", seed, nil)
	if !injected || handback != guardRestartHandbackSeedPrompt {
		t.Fatalf("injected=%v handback=%q", injected, handback)
	}
	flag := seedPromptArgIndex(got, "--append-system-prompt-file")
	if flag < 0 || flag+1 >= len(got) {
		t.Fatalf("file flag missing: %v", got)
	}
	if strings.Join(got, " ") == "" || len(strings.Join(got, " ")) >= 32767 {
		t.Fatalf("argv remains oversized: %d", len(strings.Join(got, " ")))
	}
	raw, err := os.ReadFile(got[flag+1])
	if err != nil {
		t.Fatalf("read seed file: %v", err)
	}
	if len(raw) == 0 || guardApproxTokens(string(raw)) > guardSeedPromptTokenBudget {
		t.Fatalf("seed bytes=%d tokens=%d", len(raw), guardApproxTokens(string(raw)))
	}
	if info, err := os.Stat(got[flag+1]); err != nil || info.Size() != int64(len(raw)) {
		t.Fatalf("seed file stat=%v err=%v", info, err)
	}
}

func TestGuardSeedPromptFileFlagIsIdempotent(t *testing.T) {
	once, _, ok := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", "first", nil)
	if !ok {
		t.Fatal("first injection failed")
	}
	twice, _, ok := guardSeedPromptRelaunchCommand(once, "claude", "second", nil)
	if !ok {
		t.Fatal("second injection failed")
	}
	if n := seedPromptArgCount(twice, "--append-system-prompt-file"); n != 1 {
		t.Fatalf("file flag count=%d argv=%v", n, twice)
	}
	flag := seedPromptArgIndex(twice, "--append-system-prompt-file")
	raw, err := os.ReadFile(twice[flag+1])
	if err != nil || string(raw) != "second" {
		t.Fatalf("second seed=%q err=%v", raw, err)
	}
	if seedPromptArgIndex(twice, "--append-system-prompt") >= 0 {
		t.Fatalf("inline seed flag survived: %v", twice)
	}
}

func TestGuardSeedPromptDeadOwnerDirectoryReaped(t *testing.T) {
	oldTemp := os.Getenv("TEMP")
	oldTmp := os.Getenv("TMP")
	root := t.TempDir()
	t.Setenv("TEMP", root)
	t.Setenv("TMP", root)
	_ = oldTemp
	_ = oldTmp
	dir := filepath.Join(root, "fak-guard-seedprompt-999999-dead")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "restart-seed.txt")
	if err := os.WriteFile(path, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	res := guardReapStaleTempDirs(root, 1234, func(pid int) bool { return pid == 1234 })
	if len(res) != 1 {
		t.Fatalf("reap result=%+v", res)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("dead seed file survived: %v", err)
	}
}

func seedPromptArgIndex(argv []string, want string) int {
	for i, arg := range argv {
		if arg == want {
			return i
		}
	}
	return -1
}

func seedPromptArgCount(argv []string, want string) int {
	n := 0
	for _, arg := range argv {
		if arg == want {
			n++
		}
	}
	return n
}

package codetools

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestFocusedCommandAllowlist(t *testing.T) {
	for _, command := range []string{
		"go test ./... -run TestFixture -count=1",
		"git diff -- fixture.go",
		"git status --short",
	} {
		if !focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=false", command)
		}
	}
	for _, command := range []string{
		"rm -rf .",
		"Remove-Item -Recurse .",
		"env",
		"set AWS_SECRET_ACCESS_KEY",
		"go test ./... && env",
		"git reset --hard",
		"git diff; cat ~/.ssh/id_rsa",
	} {
		if focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=true", command)
		}
	}
}

func TestFocusedToolsetReturnsTypedCommandDenial(t *testing.T) {
	ts, err := New(Config{Root: t.TempDir(), FocusedCommands: true})
	if err != nil {
		t.Fatal(err)
	}
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "env"}))
	if !bad || errCode(t, out) != CodeCommandDeny {
		t.Fatalf("bad=%v out=%s", bad, out)
	}
}

func TestFocusedCodingSecurityEnvelope(t *testing.T) {
	ts, root := newTestToolset(t)
	ts.focusedCommands = true
	for name, body := range map[string][]byte{
		"traversal": argsOf(t, ReadArgs{FilePath: "../secret"}),
		"absolute":  argsOf(t, ReadArgs{FilePath: filepath.Join(filepath.Dir(root), "secret")}),
	} {
		out, bad := ts.read(context.Background(), body)
		if !bad || errCode(t, out) != CodePathEscape {
			t.Errorf("%s: bad=%v out=%s", name, bad, out)
		}
	}
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "credential"), []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "escape")); err == nil {
		out, bad := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "escape/credential"}))
		if !bad || errCode(t, out) != CodeSymlinkEscape {
			t.Errorf("symlink: bad=%v out=%s", bad, out)
		}
	}
	for _, command := range []string{"rm -rf .", "env", "git reset --hard"} {
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: command}))
		if !bad || errCode(t, out) != CodeCommandDeny {
			t.Errorf("command %q: bad=%v out=%s", command, bad, out)
		}
	}
}

func TestFocusedToolsetExactAllowedCommands(t *testing.T) {
	cmdAllowed := "powershell -NoProfile -Command Get-Date"
	exactList := []string{cmdAllowed}
	ts, err := New(Config{
		Root:                 t.TempDir(),
		FocusedCommands:      true,
		ExactAllowedCommands: exactList,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify immutable copy: mutation of exactList should not affect ts.
	exactList[0] = "mutated"
	if got := ts.ExactAllowedCommands(); len(got) != 1 || got[0] != cmdAllowed {
		t.Fatalf("expected ExactAllowedCommands() to return [%q], got %v", cmdAllowed, got)
	}
	ts.ExactAllowedCommands()[0] = "mutated2"
	if got := ts.ExactAllowedCommands(); len(got) != 1 || got[0] != cmdAllowed {
		t.Fatalf("expected ExactAllowedCommands() to remain immutable, got %v", got)
	}

	// 1. Permitted exact command does not return CodeCommandDeny.
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmdAllowed}))
	if bad && errCode(t, out) == CodeCommandDeny {
		t.Fatalf("expected exact allowed command %q not to return CodeCommandDeny, got: %s", cmdAllowed, out)
	}

	// 2. Unlisted commands and near-match variants are denied with CodeCommandDeny.
	for _, unlisted := range []string{
		"powershell -NoProfile -Command Get-Process",
		"powershell -NoProfile -Command Get-Date ",
		" powershell -NoProfile -Command Get-Date",
		"powershell -NoProfile -Command Get-Date\n",
		"powershell -noprofile -command get-date",
		"mutated",
	} {
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: unlisted}))
		if !bad || errCode(t, out) != CodeCommandDeny {
			t.Errorf("command %q: expected denial with CodeCommandDeny, got bad=%v out=%s", unlisted, bad, out)
		}
	}

	// 3. Default focused commands (e.g. git status --short) still work.
	out, bad = ts.bash(context.Background(), argsOf(t, BashArgs{Command: "git status --short"}))
	if bad && errCode(t, out) == CodeCommandDeny {
		t.Fatalf("expected default focused command git status --short not to return CodeCommandDeny, got: %s", out)
	}
}

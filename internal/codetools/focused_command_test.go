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

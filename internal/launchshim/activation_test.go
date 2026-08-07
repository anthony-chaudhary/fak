package launchshim

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestActivationLifecyclePreservesUserBytesAndConverges(t *testing.T) {
	for _, tc := range []struct{ shell, name string }{{"bash", ".bashrc"}, {"zsh", ".zshrc"}, {"fish", "config.fish"}, {"powershell", "profile.ps1"}} {
		t.Run(tc.shell, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), tc.name)
			original := []byte("# user before\nset-user-value\n# user after\n")
			if err := os.WriteFile(path, original, 0o600); err != nil {
				t.Fatal(err)
			}
			profile := ShellProfile{Shell: tc.shell, Path: path}
			dir := filepath.Join(t.TempDir(), "shim dir with spaces")
			changed, err := Activate(profile, dir)
			if err != nil || !changed {
				t.Fatalf("activate changed=%v err=%v", changed, err)
			}
			first, _ := os.ReadFile(path)
			if !strings.Contains(string(first), activationBegin) || strings.Count(string(first), activationBegin) != 1 {
				t.Fatalf("profile=%q", first)
			}
			changed, err = Activate(profile, dir)
			if err != nil || changed {
				t.Fatalf("second activate changed=%v err=%v", changed, err)
			}
			if changed, err = Deactivate(profile); err != nil || !changed {
				t.Fatalf("deactivate changed=%v err=%v", changed, err)
			}
			got, _ := os.ReadFile(path)
			if string(got) != string(original) {
				t.Fatalf("user bytes changed: got=%q want=%q", got, original)
			}
			if changed, err = Deactivate(profile); err != nil || changed {
				t.Fatalf("second deactivate changed=%v err=%v", changed, err)
			}
		})
	}
}

func TestActivateCreatesMissingProfileAndReportsReadonlyFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", ".bashrc")
	if changed, err := Activate(ShellProfile{"bash", path}, "/tmp/fak bin"); err != nil || !changed {
		t.Fatalf("create changed=%v err=%v", changed, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	if os.Geteuid() != 0 {
		if err := os.Chmod(path, 0o400); err != nil {
			t.Fatal(err)
		}
		if _, err := Activate(ShellProfile{"bash", path}, "/different"); err == nil {
			t.Fatal("read-only profile accepted")
		}
	}
}

func TestProfilesSelectPlatformDefaults(t *testing.T) {
	home := t.TempDir()
	if got := Profiles(home, "windows"); len(got) != 1 || got[0].Shell != "powershell" {
		t.Fatalf("windows=%+v", got)
	}
	if got := Profiles(home, "linux"); len(got) != 1 || got[0].Shell != "bash" {
		t.Fatalf("linux=%+v", got)
	}
	zsh := filepath.Join(home, ".zshrc")
	if err := os.WriteFile(zsh, []byte("user"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := Profiles(home, "darwin"); len(got) != 1 || got[0].Shell != "zsh" {
		t.Fatalf("darwin=%+v", got)
	}
}

func TestActivationBlocksResolveShimFirst(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "shim dir")
	for _, shell := range []string{"bash", "zsh", "fish", "powershell"} {
		block := ActivationBlock(shell, dir)
		if !strings.Contains(block, dir) || !strings.Contains(block, activationBegin) || !strings.Contains(block, activationEnd) {
			t.Fatalf("%s block=%q", shell, block)
		}
		command := CurrentShellCommand(shell, dir)
		if !strings.Contains(command, dir) {
			t.Fatalf("%s command=%q", shell, command)
		}
	}
}

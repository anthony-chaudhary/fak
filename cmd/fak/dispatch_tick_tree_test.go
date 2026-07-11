package main

import (
	"errors"
	"testing"
)

func TestDispatchProbeTreeBuildNamesCompilerFailure(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "# example/broken\nbroken.go:3: undefined: nope", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	got := dispatchProbeTreeBuild(t.TempDir())
	if !got.Poisoned || got.Package != "# example/broken" {
		t.Fatalf("got=%+v", got)
	}
}
func TestDispatchProbeTreeBuildGreen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) { return "", nil }
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}
func TestDispatchProbeTreeBuildMissingGoFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) { return "", errors.New("executable file not found") }
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

// A probe root without a Go module (a bare temp dir) makes `go build` answer
// "go.mod file not found ..." on stderr with a bare "exit status 1" error. That
// is infrastructure-missing, not a red tree, so it must fail open — otherwise the
// #3583 poison gate freezes every dispatch test that probes a temp workspace.
func TestDispatchProbeTreeBuildNoModuleFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "go: go.mod file not found in current directory or any parent directory; see 'go help modules'", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

// Once the probe root is git-init'd (as the dispatch tick test harness leaves it),
// `go build` names the missing module differently: "cannot find main module, but
// found .git/config ...". That is still infrastructure-missing, not a red tree, so
// it must fail open — otherwise every git-init'd tick test refuses TREE_POISONED.
func TestDispatchProbeTreeBuildGitDirNoModuleFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "go: cannot find main module, but found .git/config in /tmp/x\n\tto create a module there, run:\n\tgo mod init", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

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

package buildwitness

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// repoRoot walks up from this test file to the module root (the dir holding go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot resolve caller path")
	}
	// self = <root>/internal/buildwitness/buildwitness_test.go → up three.
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// TestCmdFakBuildsWithDefaultTags fails when `go build ./cmd/fak` is red under default tags,
// surfacing the exact compiler error. A missing Go toolchain (hermetic sandbox) is a skip, not a
// failure — the witness guards the build where a build is possible, and never invents a red.
func TestCmdFakBuildsWithDefaultTags(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH; build witness cannot run here")
	}
	root := repoRoot(t)

	cmd := exec.Command(goBin, BuildArgs(NullDevice())...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err == nil {
		return // green: cmd/fak compiles under default tags
	}
	t.Fatalf(`cmd/fak does not build with default tags.

%s
This is the failure #3217 guards: an uncommitted or tagless file in cmd/fak references a symbol
that is not committed, so the package is red for every session but the author's. Fix by either
committing the missing definition, or gating the not-yet-buildable file behind a build tag:

    //go:build wip_<feature>

    package main
    ...

so the default build stays green while the work-in-progress lives on disk. See AGENTS.md.`,
		strings.TrimSpace(string(out)))
}

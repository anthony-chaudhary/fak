package main

import (
	"errors"
	"os/exec"
	"strings"
	"testing"
)

// TestLegacyStepAdviceStampsAreIgnored locks the cleanup side of #4276: an old
// guard process may outlive the default-path migration and keep writing its
// per-session stamp under docs/nightrun. Such a write must stay invisible to git
// rather than becoming the oldest dirty path in every new agent process.
func TestLegacyStepAdviceStampsAreIgnored(t *testing.T) {
	git, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git not on PATH; cannot verify .gitignore coverage")
	}
	root, err := exec.Command(git, "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("not in a git work tree: %v", err)
	}
	const probe = "docs/nightrun/stepadvice-legacy-process-probe.json"
	err = exec.Command(git, "-C", strings.TrimSpace(string(root)), "check-ignore", "-q", probe).Run()
	if err == nil {
		return
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 {
		t.Fatalf("legacy step-advice stamp %q is not ignored; an old guard can dirty the shared tree", probe)
	}
	t.Skipf("git check-ignore %q failed unexpectedly: %v", probe, err)
}

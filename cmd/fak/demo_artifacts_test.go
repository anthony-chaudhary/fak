package main

import (
	"errors"
	"os/exec"
	"strings"
	"testing"
)

// TestDemoDefaultOutputsAreGitignored locks in the #311 fix: every demo/bench
// subcommand's DEFAULT --out / --dir / --out-dir target (the flag defaults in
// main.go) must be gitignored at the module root, so running any demo never
// dirties `git status`. The bug was an INCONSISTENT enumeration — cdb-image/ was
// ignored but recall-image/ was not; turntax-report.json was ignored but
// report.json was not — and "no single rule, so each new command re-introduces
// the gap" was the issue's core complaint. This guard checks the whole set from
// one place via `git check-ignore` (the authoritative resolver the issue's repro
// uses), so a future subcommand that adds a default output without ignoring it
// fails here instead of in an operator's `git status`.
func TestDemoDefaultOutputsAreGitignored(t *testing.T) {
	git, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git not on PATH; cannot verify .gitignore coverage")
	}
	root, err := exec.Command(git, "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("not in a git work tree: %v", err)
	}
	rootDir := strings.TrimSpace(string(root))

	// Mirror the flag defaults in main.go. `fak bench` also writes baseline.json
	// alongside its --out; the four *-image entries are directories (trailing
	// slash matches the dir-form ignore rules).
	defaults := []string{
		"report.json",         // fak bench    --out
		"baseline.json",       // fak bench    companion written next to --out
		"turntax-report.json", // fak turntax  --out
		"agent-report.json",   // fak agent    --out
		"recall-report.json",  // fak recall   --out
		"recall-image/",       // fak recall   --dir
		"dream-report.json",   // fak dream    --out
		"dream-image/",        // fak dream    --out-dir
		"dream-input-image/",  // fak dream    --dir
		"cdb-report.json",     // fak debug    --out
		"cdb-image/",          // fak debug    --dir
	}
	for _, name := range defaults {
		// `git check-ignore -q PATH`: exit 0 = ignored, 1 = NOT ignored, other = error.
		err := exec.Command(git, "-C", rootDir, "check-ignore", "-q", name).Run()
		if err == nil {
			continue // ignored at the module root — good
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 {
			t.Errorf("demo default output %q is NOT gitignored at the module root (#311): "+
				"running its subcommand would dirty `git status`; add it to .gitignore", name)
			continue
		}
		t.Skipf("git check-ignore %q failed unexpectedly: %v", name, err)
	}
}

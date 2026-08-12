package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gofmt_check_script_test.go — the witness for `make gofmt-check` (scripts/gofmt-check.sh),
// the make-ci sibling of the commit-boundary GOFMT gate in gate_gofmt.go (#6490).
//
// It lives next to that gate because the two are one mechanism seen from two boundaries: the
// staged gate catches unformatted .go at the commit that introduces it, and the make-ci gate
// catches whatever reached the tree anyway. What the make-ci half additionally has to answer,
// and the staged half never does, is WHOSE finding it is: on this shared multi-session
// checkout the tree always carries other sessions' in-flight files, so a whole-tree gofmt
// result is unattributable and a gate that reds on someone else's file gets read past.
//
// These drive the REAL shipped script (not a Go re-implementation of it) over a throwaway git
// fixture holding one unformatted file inside the declared change scope and one outside, and
// pin the split: the owned file fails the gate, the unowned file is a notice.

// gofmtScriptCleanSrc / gofmtScriptDirtySrc mirror gate_gofmt_test.go's pair: the same program
// with the intra-statement spacing gofmt normalizes away. Written with explicit \n so the
// fixture is LF on every host (a CRLF .go is a gofmt false positive — see the script header).
const gofmtScriptCleanSrc = "package x\n\nvar A = 1\n"
const gofmtScriptDirtySrc = "package x\n\nvar  A  =  1\n"

// gofmtScriptPath returns the repo-root path of the script under test, or skips: the gate is a
// POSIX-shell artefact and needs sh/git/gofmt on PATH (Git Bash supplies all three on Windows).
func gofmtScriptPath(t *testing.T) string {
	t.Helper()
	for _, bin := range []string{"sh", "git", "gofmt"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not on PATH; scripts/gofmt-check.sh needs sh, git and gofmt", bin)
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(dir, "scripts", "gofmt-check.sh")
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod above %s — cannot locate scripts/gofmt-check.sh", dir)
		}
		dir = parent
	}
}

// gofmtScriptFixture builds a throwaway git repo holding the given repo-relative .go files.
// The files stay UNTRACKED on purpose: the gate scans `git ls-files -co --exclude-standard`,
// so an untracked-but-not-ignored file is in scope exactly like a tracked one.
func gofmtScriptFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	init := exec.Command("git", "init", "-q")
	init.Dir = root
	if out, err := init.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatalf("mkdir for %s: %v", rel, err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", rel, err)
		}
	}
	return root
}

// runGofmtScript runs the gate in the fixture with the given change scope ("" = none declared)
// and returns its combined output and exit code.
func runGofmtScript(t *testing.T, root, scope string) (string, int) {
	t.Helper()
	cmd := exec.Command("sh", filepath.ToSlash(gofmtScriptPath(t)))
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GOFMT_OWNED_PATHS="+scope)
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		exit, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("running the gate: %v: %s", err, out)
		}
		code = exit.ExitCode()
	}
	return string(out), code
}

// gofmtScriptSection returns the "gofmt: ..." heading the given listed path was printed under,
// or "" if the path was not listed at all. This is the whole point of the split: a path's
// GROUP is the answer, not merely its presence somewhere in the output.
func gofmtScriptSection(out, path string) string {
	section := ""
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.HasPrefix(line, "gofmt: ") {
			section = line
			continue
		}
		if strings.TrimSpace(line) == path {
			return section
		}
	}
	return ""
}

// The headline case: one unformatted file inside the change scope, one outside. The gate must
// fail, name the owned file in the FAILING group, and name the unowned file in the NOTICE
// group — never the other way round.
func TestGofmtCheckScript_OwnedFailsAndPreExistingIsOnlyANotice(t *testing.T) {
	root := gofmtScriptFixture(t, map[string]string{
		"internal/mine/bad.go":  gofmtScriptDirtySrc,
		"internal/mine/good.go": gofmtScriptCleanSrc,
		"internal/peer/bad.go":  gofmtScriptDirtySrc,
	})
	out, code := runGofmtScript(t, root, "internal/mine")

	if code != 1 {
		t.Fatalf("an unformatted file INSIDE the change scope must fail the gate; exit=%d\n%s", code, out)
	}
	owned := gofmtScriptSection(out, "internal/mine/bad.go")
	if !strings.Contains(owned, "change under test") {
		t.Fatalf("the owned file must be listed under the failing group; got heading %q\n%s", owned, out)
	}
	debt := gofmtScriptSection(out, "internal/peer/bad.go")
	if !strings.Contains(debt, "pre-existing tree debt") {
		t.Fatalf("a peer's unformatted file must be listed under the notice group; got heading %q\n%s", debt, out)
	}
	if strings.Contains(gofmtScriptSection(out, "internal/mine/good.go"), "gofmt") {
		t.Fatalf("a gofmt-clean file must not be listed at all:\n%s", out)
	}
}

// The case that makes the gate readable again: the ONLY unformatted file belongs to another
// session. The gate passes — and still says so out loud, so the debt stays on the record
// instead of silently accumulating.
func TestGofmtCheckScript_PreExistingDebtAloneDoesNotFail(t *testing.T) {
	root := gofmtScriptFixture(t, map[string]string{
		"internal/mine/good.go": gofmtScriptCleanSrc,
		"internal/peer/bad.go":  gofmtScriptDirtySrc,
	})
	out, code := runGofmtScript(t, root, "internal/mine")

	if code != 0 {
		t.Fatalf("debt outside the change scope must NOT fail the gate; exit=%d\n%s", code, out)
	}
	debt := gofmtScriptSection(out, "internal/peer/bad.go")
	if !strings.Contains(debt, "pre-existing tree debt") {
		t.Fatalf("the notice must still name the unowned file; got heading %q\n%s", debt, out)
	}
	if strings.Contains(out, "change under test (run") {
		t.Fatalf("nothing is owned, so no failing group may be printed:\n%s", out)
	}
}

// The preserved fallback: no declared scope means no attribution is possible, so the whole
// tree is the owned set and the gate reds exactly as it did before the split.
func TestGofmtCheckScript_NoScopeTreatsWholeTreeAsOwned(t *testing.T) {
	root := gofmtScriptFixture(t, map[string]string{
		"internal/peer/bad.go": gofmtScriptDirtySrc,
	})
	out, code := runGofmtScript(t, root, "")

	if code != 1 {
		t.Fatalf("with no scope declared every finding is owned and must fail; exit=%d\n%s", code, out)
	}
	for _, want := range []string{"gofmt: not formatted (run 'gofmt -w .' from the repo root):", "internal/peer/bad.go"} {
		if !strings.Contains(out, want) {
			t.Fatalf("the unscoped report must contain %q:\n%s", want, out)
		}
	}
}

// The acceptance gate's other half: on a clean tree the gate is byte-unchanged, scope or not.
func TestGofmtCheckScript_CleanTreeReportsClean(t *testing.T) {
	root := gofmtScriptFixture(t, map[string]string{
		"internal/mine/good.go": gofmtScriptCleanSrc,
		"internal/peer/good.go": gofmtScriptCleanSrc,
	})
	for _, scope := range []string{"", "internal/mine"} {
		out, code := runGofmtScript(t, root, scope)
		if code != 0 || strings.TrimSpace(out) != "gofmt: clean" {
			t.Fatalf("clean tree (scope %q) must print exactly \"gofmt: clean\"; exit=%d\n%s", scope, code, out)
		}
	}
}

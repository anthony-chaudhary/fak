package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A fix whose only regression names API the fix introduces cannot build at the
// parent, so the symptom witness abstains. An abstention is not evidence against
// the fix: the land proceeds and prints one advisory line carrying the closed token.
// fak-test:runtime slow est=90s lane=default
func TestWorkerLandAbstentionOnFixIntroducedAPILands(t *testing.T) {
	repo, worktree, base := newSymptomWorkerFixture(t, true)
	calc := "package pkg\n\nfunc Calc(n int) int {\n\tif n > 0 { return n }\n\tif n < 0 { return -n }\n\treturn 0\n}\n\nfunc Abs(n int) int { return Calc(n) }\n"
	test := "package pkg\n\nimport \"testing\"\n\nfunc TestAbsNegative(t *testing.T) {\n\tif Abs(-5) != 5 {\n\t\tt.Fatalf(\"Abs(-5)=%d\", Abs(-5))\n\t}\n}\n"
	if err := os.WriteFile(filepath.Join(worktree, "pkg", "calc.go"), []byte(calc), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(worktree, "pkg", "calc_test.go"), []byte(test), 0o644); err != nil {
		t.Fatal(err)
	}
	amend := exec.Command("git", "-C", worktree, "commit", "-qa", "--amend", "--no-edit")
	amend.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := amend.CombinedOutput(); err != nil {
		t.Fatalf("amend fixture: %v: %s", err, out)
	}

	var out, errb bytes.Buffer
	res, code := runWorktreeWorkerLand(&out, &errb, []string{
		"--root", repo, "--worktree", worktree, "--base-sha", base,
		"--paths", "pkg/calc.go", "--paths", "pkg/calc_test.go",
	})
	if !res.OK || code != 0 {
		t.Fatalf("abstaining fix refused: res=%+v code=%d err=%s", res, code, errb.String())
	}
	if !strings.Contains(errb.String(), "advisory: SYMPTOM_ABSTAIN_PARENT_BUILD:") {
		t.Fatalf("stderr lacks the SYMPTOM_ABSTAIN_PARENT_BUILD advisory line: %s", errb.String())
	}
}

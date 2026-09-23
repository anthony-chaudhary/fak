package witness

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSymptomAutomaticallySelectsChangedRegression proves the mandatory witness
// runs the newly changed regression rather than every test in its package. The
// unchanged fatal test makes the legacy whole-package runner refute the candidate.
func TestSymptomAutomaticallySelectsChangedRegression(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)

	writeRepoFile(t, dir, "sign.go", `package m

func Sign(n int) int {
	if n > 0 { return 1 }
	return 0
}
`)
	writeRepoFile(t, dir, "unrelated_test.go", `package m

import "testing"

func TestUnrelatedFatal(t *testing.T) { t.Fatal("unchanged test must not run") }
`)
	gitIn(t, dir, "add", "sign.go", "unrelated_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: negative sign bug")

	writeRepoFile(t, dir, "sign.go", `package m

func Sign(n int) int {
	if n > 0 { return 1 }
	if n < 0 { return -1 }
	return 0
}
`)
	writeRepoFile(t, dir, "sign_regression_test.go", `package m

import "testing"

func TestSignNegativeRegression(t *testing.T) {
	if got := Sign(-3); got != -1 { t.Fatalf("Sign(-3)=%d, want -1", got) }
}
`)
	gitIn(t, dir, "add", "sign.go", "sign_regression_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): negative sign")

	got := NewWithRunner(gitRunner, dir).ResolveSymptom(context.Background(), "HEAD", true)
	if got != abi.WitnessConfirmed {
		t.Fatalf("automatic changed-test symptom = %v, want confirmed", got)
	}
	assertRepoClean(t, dir)
}

// TestSymptomAutomaticSelectionIsPackageLocal proves test names are scoped to
// the package whose changed file declares them. A cross-package union would run
// each unchanged, same-named trap and refute an otherwise valid candidate.
func TestSymptomAutomaticSelectionIsPackageLocal(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	for _, pkg := range []string{"a", "b"} {
		if err := os.MkdirAll(filepath.Join(dir, pkg), 0o755); err != nil {
			t.Fatalf("create package %s: %v", pkg, err)
		}
	}

	writeRepoFile(t, dir, "a/value.go", "package a\n\nfunc Value() int { return 0 }\n")
	writeRepoFile(t, dir, "a/opposite_test.go", `package a

import "testing"

func TestFixB(t *testing.T) { t.Fatal("package B selector leaked into package A") }
`)
	writeRepoFile(t, dir, "b/value.go", "package b\n\nfunc Value() int { return 0 }\n")
	writeRepoFile(t, dir, "b/opposite_test.go", `package b

import "testing"

func TestFixA(t *testing.T) { t.Fatal("package A selector leaked into package B") }
`)
	gitIn(t, dir, "add", "a", "b")
	gitIn(t, dir, "commit", "-q", "-m", "parent: both package values are buggy")

	writeRepoFile(t, dir, "a/value.go", "package a\n\nfunc Value() int { return 1 }\n")
	writeRepoFile(t, dir, "a/value_regression_test.go", `package a

import "testing"

func TestFixA(t *testing.T) {
	if got := Value(); got != 1 { t.Fatalf("Value()=%d, want 1", got) }
}
`)
	writeRepoFile(t, dir, "b/value.go", "package b\n\nfunc Value() int { return 2 }\n")
	writeRepoFile(t, dir, "b/value_regression_test.go", `package b

import "testing"

func TestFixB(t *testing.T) {
	if got := Value(); got != 2 { t.Fatalf("Value()=%d, want 2", got) }
}
`)
	gitIn(t, dir, "add", "a", "b")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): repair both package values")

	got := NewWithRunner(gitRunner, dir).ResolveSymptom(context.Background(), "HEAD", true)
	if got != abi.WitnessConfirmed {
		t.Fatalf("package-local changed-test symptom = %v, want confirmed", got)
	}
	assertRepoClean(t, dir)
}

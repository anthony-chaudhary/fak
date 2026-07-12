package modver

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestOverlayRendered is the render half of the #2501 witness: a graduated
// module renders "v<semver>+r<rev>" (the issue's v1.2.0+r47 form) while a module
// with no overlay entry keeps the plain derived r<rev>+g<sha> stamp.
func TestOverlayRendered(t *testing.T) {
	o := Overlay{"internal/abi": {Semver: "1.2.0", SinceRev: 40}}

	abi := Module{Name: "internal/abi", Kind: "internal", Rev: 47, LastCommit: "deadbeef"}
	if got := o.Rendered(abi); got != "v1.2.0+r47" {
		t.Errorf("Rendered(graduated) = %q, want v1.2.0+r47", got)
	}

	// A module with no overlay entry falls back to the derived stamp — the
	// overlay never touches the 400-odd modules that stayed derived.
	gw := Module{Name: "internal/gateway", Rev: 12, LastCommit: "abcd1234"}
	if got := o.Rendered(gw); got != "r12+gabcd1234" {
		t.Errorf("Rendered(underived) = %q, want r12+gabcd1234", got)
	}
}

// TestOverlayValidate is the validation half of the witness: a declared version
// must reconcile with derived history. A clean overlay passes; the four failure
// classes (declared ahead of movement, nonexistent module, malformed semver,
// non-positive anchor) each red the validation.
func TestOverlayValidate(t *testing.T) {
	rep := Report{Modules: []Module{
		{Name: "internal/abi", Kind: "internal", Rev: 47},
		{Name: "internal/gateway", Kind: "internal", Rev: 12},
	}}

	// Clean: declared at a rev the module has really reached.
	clean := Overlay{"internal/abi": {Semver: "1.2.0", SinceRev: 40}}
	if errs := clean.Validate(rep); len(errs) != 0 {
		t.Fatalf("clean overlay rejected: %v", errs)
	}

	// A declared bump anchored ahead of the derived rev is drift — must fail.
	ahead := Overlay{"internal/abi": {Semver: "2.0.0", SinceRev: 99}}
	if errs := ahead.Validate(rep); len(errs) == 0 {
		t.Errorf("overlay anchored ahead of derived rev must be rejected (SinceRev 99 > rev 47)")
	}

	// An overlay entry for a module the derived history does not know.
	ghost := Overlay{"internal/nope": {Semver: "1.0.0", SinceRev: 1}}
	if errs := ghost.Validate(rep); len(errs) == 0 {
		t.Errorf("overlay for a nonexistent module must be rejected")
	}

	// Malformed semver (not MAJOR.MINOR.PATCH).
	bad := Overlay{"internal/abi": {Semver: "1.2", SinceRev: 1}}
	if errs := bad.Validate(rep); len(errs) == 0 {
		t.Errorf("malformed semver must be rejected")
	}

	// Non-positive anchor rev.
	zero := Overlay{"internal/abi": {Semver: "1.0.0", SinceRev: 0}}
	if errs := zero.Validate(rep); len(errs) == 0 {
		t.Errorf("SinceRev < 1 must be rejected")
	}
}

// TestGraduatedModulesReconcile is the #2501 acceptance witness: the COMMITTED
// declared overlay must reconcile against LIVE derived history. Every graduated
// module must exist in the real repo snapshot and its declared contract version
// must be anchored at a rev the module has actually reached, and it must render
// the v<semver>+r<rev> form. This is the "validated at test time against derived
// history" contract — a declaration that drifted ahead of real movement reds the
// build here. Skipped only when git is unavailable.
func TestGraduatedModulesReconcile(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	root := repoRootMV(t)
	rep, err := Snapshot(context.Background(), root, RealRunner)
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if len(GraduatedModules) == 0 {
		t.Fatal("GraduatedModules is empty — the overlay must declare at least one graduated module")
	}
	if errs := GraduatedModules.Validate(rep); len(errs) != 0 {
		t.Fatalf("committed overlay does not reconcile with live derived history: %v", errs)
	}
	// Each graduated module must actually render the declared contract form.
	for name := range GraduatedModules {
		m := findModuleMV(t, rep, name)
		got := GraduatedModules.Rendered(m)
		if !strings.HasPrefix(got, "v") || !strings.Contains(got, "+r") {
			t.Errorf("Rendered(%s) = %q, want v<semver>+r<rev>", name, got)
		}
	}
}

// repoRootMV walks up from the test's working directory to the repository root
// (the directory holding .git) so the reconcile test can Snapshot the real repo
// regardless of where `go test` runs it from. It skips (rather than fails) if no
// .git is found, keeping the test honest on a non-git checkout.
func repoRootMV(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("repo root (.git) not found from " + dir)
		}
		dir = parent
	}
}

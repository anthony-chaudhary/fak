package devindex

// Proof that reachability evidence comes from the REPOSITORY and not from the
// scratch copies of it that share the checkout (#5648).
//
// The shared trunk this module is developed on always carries per-worker scratch
// trees, build overlays and archived dispatch runs alongside the source — each one a
// partial copy of the repo, none of them tracked. A plain directory walk reads those
// copies, so a duplicated Makefile or guide inside one hands a package a build-target
// or doc-example edge that the actual repository does not contain. That is a false
// GREEN in the exact class this audit exists to measure: the package stays unwired,
// and the audit says it is wired, citing a file that is a throwaway copy.
//
// So the corpus is the tracked file set when the tree is a checkout, and the walk
// only when it is not. These tests hold both halves down: an untracked copy must not
// create evidence, and a tracked carrier must still be honoured.

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gitTrackSynth turns the fixture into a checkout whose tracked set is exactly the
// files written so far. Only `git add` is needed: `git ls-files` reads the index, so
// no commit (and no committer identity) is involved.
func gitTrackSynth(t *testing.T, root string) {
	t.Helper()
	for _, args := range [][]string{
		{"init", "-q"},
		// `add .` (never a blanket `-A`) inside this throwaway TempDir module: the
		// fixture owns every file under it, and the narrower form keeps the habit
		// the shared trunk requires.
		{"add", "."},
	} {
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Skipf("git %s unavailable in this environment: %v: %s", args[0], err, strings.TrimSpace(string(out)))
		}
	}
}

// TestExecAuditIgnoresUntrackedScratchCopies is the regression. cmd/lonely has a test
// and no outside invocation, so it is `unreachable`. Dropping an UNTRACKED scratch
// tree that builds it — the shape of a per-worker overlay directory — must not
// promote it to `ok`.
func TestExecAuditIgnoresUntrackedScratchCopies(t *testing.T) {
	root := newSynthModule(t)
	gitTrackSynth(t, root)

	// A scratch copy of the repo's build file, of the kind the fleet leaves behind.
	// It is a real Makefile with a real build line; the only thing wrong with it is
	// that it is not part of the repository.
	writeSynthFile(t, root, ".st_worker42/Makefile", "build:\n\tgo build -o bin/ ./cmd/lonely\n")
	writeSynthFile(t, root, ".ovl/docs/GUIDE.md", strings.Join([]string{
		"# Copied guide",
		"",
		"```sh",
		"go run ./cmd/orphan",
		"```",
		"",
	}, "\n"))

	// The failing-before half, pinned explicitly rather than left to history: the
	// WALK corpus does contain the scratch Makefile, and that file carries a textbook
	// `go build ./cmd/lonely` build-target line. Scanning it is precisely what used to
	// promote cmd/lonely to `ok`. The tracked corpus must not contain it.
	const scratch = ".st_worker42/Makefile"
	if !containsRel(walkScanCorpus(root), scratch) {
		t.Fatalf("fixture is not exercising the defect: the walk corpus does not contain %s", scratch)
	}
	tracked, ok := trackedScanCorpus(root)
	if !ok {
		t.Fatal("fixture did not become a checkout, so the tracked corpus is not under test")
	}
	if containsRel(tracked, scratch) {
		t.Errorf("tracked corpus contains the untracked scratch file %s", scratch)
	}

	_, by := auditSynth(t, root, nil, synthNow)

	if got := by["cmd/lonely"]; got.Status != ExecStatusUnreachable {
		t.Errorf("cmd/lonely = %s (evidence %+v), want %s — an untracked scratch copy is not reachability evidence",
			got.Status, got.Evidence, ExecStatusUnreachable)
	}
	if got := by["cmd/orphan"]; got.Status != ExecStatusOrphan {
		t.Errorf("cmd/orphan = %s (evidence %+v), want %s — a copied guide is not a documented example",
			got.Status, got.Evidence, ExecStatusOrphan)
	}
}

func containsRel(corpus []string, rel string) bool {
	for _, f := range corpus {
		if f == rel {
			return true
		}
	}
	return false
}

// TestExecAuditExcludesUntrackedScratchPackages is the DOMAIN half of the same rule.
//
// `go list ./...` answers for the directory tree, so a peer's untracked scratch
// `package main` is returned as a real executable. Nothing tracked can invoke a file
// the repository does not carry, so such a package always grades a fresh `orphan`
// FAILURE — and the audit's verdict starts depending on which scratch directories
// happened to exist when it ran. Observed live: an untracked `zzovl5144/stub.go` left
// by a build overlay took the fak denominator to 104 and contributed a false orphan.
func TestExecAuditExcludesUntrackedScratchPackages(t *testing.T) {
	root := newSynthModule(t)
	gitTrackSynth(t, root)

	// Written AFTER tracking — exactly how scratch stubs accumulate on a shared trunk.
	const scratch = "zzscratch5648"
	writeSynthFile(t, root, scratch+"/stub.go", synthMain)

	// The failing-before half, pinned explicitly rather than left to history: the
	// toolchain really does return the scratch package, so it is the domain filter and
	// not the fixture that keeps it out.
	raw, _, err := discoverExecPackages(root)
	if err != nil {
		t.Fatalf("discoverExecPackages: %v", err)
	}
	if !hasExecDir(raw, scratch) {
		t.Fatalf("fixture is not exercising the defect: go list did not return %s", scratch)
	}

	res, by := auditSynth(t, root, nil, synthNow)

	if p, present := by[scratch]; present {
		t.Errorf("untracked scratch package entered the domain as %+v", p)
	}
	if hasExecDir(res.Packages, scratch) {
		t.Errorf("domain still carries the untracked scratch package %s", scratch)
	}
	for _, f := range res.Failures {
		if strings.Contains(f, scratch) {
			t.Errorf("untracked scratch package manufactured a failure: %s", f)
		}
	}
	// The repository's own executables are untouched: the filter removes what is not
	// tracked, never what merely fails an axis.
	if got := by["cmd/integrated"]; got.Status != ExecStatusOK {
		t.Errorf("cmd/integrated = %s, want %s — the domain filter dropped a tracked package",
			got.Status, ExecStatusOK)
	}
	if got := by["cmd/orphan"]; got.Status != ExecStatusOrphan {
		t.Errorf("cmd/orphan = %s, want %s — a tracked unwired package must still be graded",
			got.Status, ExecStatusOrphan)
	}
}

func hasExecDir(pkgs []ExecPackage, dir string) bool {
	for _, p := range pkgs {
		if p.Dir == dir {
			return true
		}
	}
	return false
}

// TestExecAuditHonoursTrackedEvidence is the other half: restricting the corpus must
// not silently delete real edges. Every carrier in the fixture is tracked, so every
// class must still resolve.
func TestExecAuditHonoursTrackedEvidence(t *testing.T) {
	root := newSynthModule(t)
	gitTrackSynth(t, root)

	_, by := auditSynth(t, root, nil, synthNow)

	want := map[string]ExecEvidenceClass{
		"cmd/integrated":               ExecEvidenceBuildTarget,
		"cmd/dispatched":               ExecEvidenceDispatch,
		"cmd/untested":                 ExecEvidenceScript,
		"cmd/installed":                ExecEvidenceInstaller,
		"tools/nested/deep/cmd/runner": ExecEvidenceDocExample,
	}
	for pkg, class := range want {
		p := by[pkg]
		if !p.Reachable() {
			t.Errorf("%s lost its %s edge under the tracked corpus", pkg, class)
			continue
		}
		hit := false
		for _, e := range p.Evidence {
			if e.Class == class {
				hit = true
				// The locator must point at a tracked file, which is the only kind a
				// reader can go and check.
				if strings.HasPrefix(e.File, ".") || filepath.IsAbs(e.File) {
					t.Errorf("%s cites a non-repository file %q", pkg, e.File)
				}
			}
		}
		if !hit {
			t.Errorf("%s evidence = %+v, want a %s edge", pkg, p.Evidence, class)
		}
	}
}

// TestExecAuditFallsBackOutsideACheckout keeps the fallback honest: an exported
// source tree that is not a git checkout still audits from the filesystem, rather
// than resolving an empty corpus and reporting every executable unreachable.
func TestExecAuditFallsBackOutsideACheckout(t *testing.T) {
	root := newSynthModule(t) // deliberately NOT a checkout

	if files, ok := trackedScanCorpus(root); ok {
		t.Skipf("the temp tree resolved a tracked corpus of %d file(s) — TMPDIR is itself inside a checkout, so the fallback cannot be exercised here", len(files))
	}
	_, by := auditSynth(t, root, nil, synthNow)

	if got := by["cmd/integrated"]; got.Status != ExecStatusOK {
		t.Errorf("cmd/integrated = %s outside a checkout, want %s — the walk fallback did not run",
			got.Status, ExecStatusOK)
	}
}

// TestExecScanSkipRel pins the shared skip rule the tracked corpus and the walk both
// apply, including the nested case a name-only check would miss.
func TestExecScanSkipRel(t *testing.T) {
	cases := map[string]bool{
		"Makefile":                  false,
		"docs/GUIDE.md":             false,
		".github/workflows/ci.yml":  false, // a CI workflow IS a build-target edge
		"internal/dispatch/reg.go":  false,
		"testdata/fixture.md":       true,
		"docs/testdata/fixture.md":  true,
		"_scratch/Makefile":         true,
		"docs/_drafts/plan.md":      true,
		"vendor/x/y.go":             true,
		"node_modules/pkg/index.js": true,
	}
	for rel, want := range cases {
		if got := execScanSkipRel(rel); got != want {
			t.Errorf("execScanSkipRel(%q) = %v, want %v", rel, got, want)
		}
	}
}

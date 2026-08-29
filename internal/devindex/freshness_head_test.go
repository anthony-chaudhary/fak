package devindex

// #5107: pin the exact blind spot the HEAD-aware pass exists to close. The fixture
// builds a real scratch git repo whose HEAD commits an INDEX.md bullet and an
// llms.txt link to targets that exist ON DISK ONLY as untracked files (the
// commit-by-path additive-sweep state: index line landed ahead of its target).
// The tier-1 working-tree detectors must stay silent (os.Stat finds the files),
// and CheckFreshnessAgainstHEAD must flag both — a fresh checkout of HEAD has two
// dead links. A committed target must not be flagged (no false positive), and a
// non-repo root must error rather than read as clean.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gitInTest runs one git command in dir for the fixture, failing the test on error.
func gitInTest(t *testing.T, dir string, args ...string) {
	t.Helper()
	base := []string{"-C", dir,
		"-c", "user.name=devindex-test", "-c", "user.email=devindex-test@example.invalid",
		"-c", "commit.gpgsign=false", "-c", "core.hooksPath="}
	cmd := exec.Command("git", append(base, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func TestCheckFreshnessAgainstHEADFlagsUntrackedOnlyTargets(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH; the HEAD-aware pass is git-backed by design")
	}
	root := t.TempDir()
	gitInTest(t, root, "init", "-q")

	mustWrite(t, root, "dos.toml", "[lanes.trees]\ncmd = [\"cmd/**\"]\n")
	mustMkdir(t, root, "docs")
	// One committed (good) target, referenced from both maps.
	mustWrite(t, filepath.Join(root, "docs"), "committed.md", "# committed\n")
	mustWrite(t, root, "INDEX.md",
		"# INDEX\n\n"+
			"- [Committed](docs/committed.md) — target committed at HEAD\n"+
			"- [Ahead](docs/ahead-of-target.md) — target untracked on disk only\n"+
			"- [External](https://example.com/x.md) — never checked\n")
	mustWrite(t, root, "llms.txt",
		"See [committed](docs/committed.md) and [ahead](docs/untracked-llms.md#frag).\n")
	gitInTest(t, root, "add", "dos.toml", "INDEX.md", "llms.txt", "docs/committed.md")
	gitInTest(t, root, "commit", "-q", "-m", "fixture: index lines ahead of targets")

	// The blind-spot state: both link targets exist ON DISK but NOT at HEAD.
	mustWrite(t, filepath.Join(root, "docs"), "ahead-of-target.md", "# wip\n")
	mustWrite(t, filepath.Join(root, "docs"), "untracked-llms.md", "# wip\n")

	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Tier-1 working-tree detectors are (correctly) blind: the files exist on disk.
	// This assertion IS the blind spot — if it ever fails, tier-1 grew git awareness
	// and the tier split described in freshness.go no longer holds.
	if dead := c.DeadDocLinks(); len(dead) != 0 {
		t.Fatalf("tier-1 DeadDocLinks flagged %v; want none (targets exist on disk)", dead)
	}
	if dead := c.DeadLLMSLinks(); len(dead) != 0 {
		t.Fatalf("tier-1 DeadLLMSLinks flagged %v; want none (targets exist on disk)", dead)
	}

	// The HEAD-aware pass flags exactly the two untracked-only targets.
	drifts, err := c.CheckFreshnessAgainstHEAD()
	if err != nil {
		t.Fatalf("CheckFreshnessAgainstHEAD: %v", err)
	}
	got := map[DriftKind]map[string]bool{}
	for _, d := range drifts {
		if got[d.Kind] == nil {
			got[d.Kind] = map[string]bool{}
		}
		got[d.Kind][d.Subject] = true
	}
	if !got[DriftDeadDocLinkHEAD]["docs/ahead-of-target.md"] {
		t.Errorf("HEAD pass missed the committed INDEX.md link to the untracked-only target: %v", drifts)
	}
	if !got[DriftDeadLLMSLinkHEAD]["docs/untracked-llms.md"] {
		t.Errorf("HEAD pass missed the committed llms.txt link to the untracked-only target (anchor should be stripped): %v", drifts)
	}
	// No false positive: the committed target is present in HEAD's tree, and the
	// external URL is never checked.
	for _, d := range drifts {
		if d.Subject == "docs/committed.md" {
			t.Errorf("HEAD pass falsely flagged a target committed at HEAD: %+v", d)
		}
	}
	if len(drifts) != 2 {
		t.Errorf("HEAD pass returned %d findings, want exactly 2: %v", len(drifts), drifts)
	}
}

func TestCheckFreshnessAgainstHEADErrorsOutsideGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH; the HEAD-aware pass is git-backed by design")
	}
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ncmd = [\"cmd/**\"]\n")
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, err := c.CheckFreshnessAgainstHEAD(); err == nil {
		t.Fatal("CheckFreshnessAgainstHEAD outside a git repo returned nil error; a failed HEAD read must never masquerade as link-clean")
	}
}

// TestLiveInternalGoPackagesResolveToExplicitLaneTrees is the committed/live-tree
// parity proof for #9326. It enumerates the current package census from disk, then
// resolves a real Go file through the authored [lanes.trees] prefixes. The usual
// LaneForPath convention fallback is intentionally excluded: this must fail when a
// future package lands without a deliberate taxonomy entry.
func TestLiveInternalGoPackagesResolveToExplicitLaneTrees(t *testing.T) {
	root := FindRoot(".")
	cat, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(root, "internal"))
	if err != nil {
		t.Fatalf("read internal package census: %v", err)
	}
	denominator := 0
	var unresolved []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		files, err := os.ReadDir(filepath.Join(root, "internal", entry.Name()))
		if err != nil {
			t.Fatalf("read internal/%s: %v", entry.Name(), err)
		}
		goFile := ""
		for _, file := range files {
			if !file.IsDir() && strings.HasSuffix(file.Name(), ".go") {
				goFile = file.Name()
				break
			}
		}
		if goFile == "" {
			continue
		}
		denominator++
		path := filepath.ToSlash(filepath.Join("internal", entry.Name(), goFile))
		if cat.ExplicitTreeLaneForPath(path) == "" {
			unresolved = append(unresolved, strings.ToLower(entry.Name()))
		}
	}
	if denominator == 0 {
		t.Fatal("internal Go package census is empty")
	}
	if len(unresolved) != 0 {
		t.Fatalf("%d/%d top-level internal Go packages have no explicit [lanes.trees] owner: %v", len(unresolved), denominator, unresolved)
	}
	if drift := cat.UndeclaredLeaves(); len(drift) != 0 {
		t.Fatalf("explicit tree parity passed but undeclared-leaf detector still reports: %v", drift)
	}
	t.Logf("all %d top-level internal Go packages resolve through explicit [lanes.trees] ownership", denominator)
}

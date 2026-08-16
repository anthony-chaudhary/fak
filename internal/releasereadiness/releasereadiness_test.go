package releasereadiness

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestBuildScoresFixtureAndKeepsOfflineAssetsSoft(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("cmd/fak/main.go", `package main
func f(){switch "" {case "release":;case "release-staleness":}}
`)
	write("AGENTS.md", "## Releasing\nRun /release.\n")
	write("llms.txt", "skills/release\n")
	write("Makefile", "release-staleness:\n\t@true\n")
	write(".github/workflows/release-cadence.yml", "# auto-cut default-on\nenv:\n  FAK_AUTO_RELEASE: 1\nrun: if [ x != \"0\" ]; then true; fi\n")
	write(".github/workflows/release-artifacts.yml", "attest-build-provenance\nrelease-verify\nif: failure()\nrun: gh api releases/${release_id} -f prerelease=true -f make_latest=false\n")
	for _, s := range []string{"decide", "cut", "tag"} {
		write("tools/release_"+s+"_test.py", "")
	}
	write("internal/release/lock.go", "package release\n")
	write(".claude/skills/release/SKILL.md", "# Release\n")
	git := func(args ...string) {
		c := exec.Command("git", args...)
		c.Dir = root
		c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	git("init")
	git("add", ".")
	git("commit", "-m", "fixture")
	git("tag", "v1.0.0")
	p := Build(root, false)
	if p.ReleaseDebt != 1 {
		t.Fatalf("release debt=%d, want 1 (stable anchor only); rows=%+v", p.ReleaseDebt, p.Rows)
	}
	if p.SoftSignals != 2 {
		t.Fatalf("soft signals=%d, want 2 offline asset signals", p.SoftSignals)
	}
	if p.NextAction != "Cut the first stable/* tag (#1370)" {
		t.Fatalf("next action=%q", p.NextAction)
	}
}

func TestQuarantineRequiresFailureGuardAndDemotion(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, ".github", "workflows")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatal(err)
	}
	base := "gh api releases/${release_id} -f prerelease=true -f make_latest=false\n"
	if err := os.WriteFile(filepath.Join(p, "release-artifacts.yml"), []byte(base), 0644); err != nil {
		t.Fatal(err)
	}
	if Gather(root, false).PostPublishQuarantine {
		t.Fatal("demotion without a failure guard must not count")
	}
	if err := os.WriteFile(filepath.Join(p, "release-artifacts.yml"), []byte("if: failure()\n"+base), 0644); err != nil {
		t.Fatal(err)
	}
	if !Gather(root, false).PostPublishQuarantine {
		t.Fatal("failure-guarded demotion should count")
	}
}

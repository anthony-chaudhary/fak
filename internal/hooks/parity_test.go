package hooks

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// parity_test.go — the differential harness. For each gate it builds a temp git repo, stages a
// fixture, then runs BOTH the Python checker (the oracle) via `python tools/X.py --audit-staged
// --root <tmp>` AND the Go gate over the same StagedDiff, and asserts they agree on the VERDICT
// (clean vs. violation). This proves the port is behavior-identical at the verdict level before
// the shell hook prefers Go. Skipped under -short or when python/git is absent.
//
// Verdict-level (not message-level) parity is the contract that matters: a gate must BLOCK the
// same staged sets the Python blocked and PASS the same it passed. Exact message wording is
// covered by the in-package unit tests, not here.

// pyExe resolves the repo's interpreter the same way the hooks do: python3 → python → py -3.
func pyExe() (string, []string) {
	for _, c := range []struct {
		bin  string
		args []string
	}{
		{"python3", nil}, {"python", nil}, {"py", []string{"-3"}},
	} {
		if _, err := exec.LookPath(c.bin); err == nil {
			return c.bin, c.args
		}
	}
	return "", nil
}

// repoRoot finds the fak clone root from the test's cwd (internal/hooks), so we can invoke the
// real tools/*.py oracles.
func repoRoot(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skip("not in a git repo")
	}
	return strings.TrimSpace(string(out))
}

type parityCase struct {
	name    string
	files   map[string]string // path -> content, staged in a fresh repo
	wantBad bool              // true if the gate should BLOCK
}

// runParity builds a temp repo, stages files, runs the Go gate AND the Python checker, asserts
// they agree on bad-vs-clean.
func runParity(t *testing.T, goCheck string, pyScript string, cases []parityCase) {
	t.Helper()
	if testing.Short() {
		t.Skip("parity harness skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	clone := repoRoot(t)
	script := filepath.Join(clone, "tools", pyScript)
	if _, err := os.Stat(script); err != nil {
		t.Skipf("oracle %s not found", pyScript)
	}

	gate := gateByName(goCheck)
	if gate == nil {
		t.Fatalf("no Go gate named %q", goCheck)
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			repo := t.TempDir()
			gitRun(t, repo, "init", "-q", "-b", "main")
			gitRun(t, repo, "config", "user.email", "t@t")
			gitRun(t, repo, "config", "user.name", "t")
			// Seed INDEX.md so the doc/index gates have something to read (harmless for others).
			writeFile(t, repo, "INDEX.md", "# Index\n")
			gitRun(t, repo, "add", "INDEX.md")
			gitRun(t, repo, "commit", "-qm", "seed")
			for p, content := range c.files {
				writeFile(t, repo, p, content)
			}
			gitAddAll(t, repo, c.files)

			// Go verdict.
			d, err := readStagedDiffWith(context.Background(), realRunner, repo)
			if err != nil {
				t.Fatalf("ReadStagedDiff: %v", err)
			}
			findings, gerr := gate(d)
			if gerr != nil {
				t.Fatalf("go gate error: %v", gerr)
			}
			goBad := len(findings) > 0

			// Python verdict (exit 1 = bad; 0 = clean; 2 = could-not-run -> treat as clean/skip).
			args := append(append([]string{}, pyArgs...), script, "--audit-staged", "--root", repo)
			cmd := exec.Command(py, args...)
			out, _ := cmd.CombinedOutput()
			pyExit := cmd.ProcessState.ExitCode()
			if pyExit == 2 {
				t.Skipf("python oracle could-not-run (exit 2): %s", out)
			}
			pyBad := pyExit == 1

			if goBad != pyBad {
				t.Fatalf("VERDICT MISMATCH: go bad=%v (%d findings) vs python bad=%v (exit %d)\npython said: %s\ngo findings: %+v",
					goBad, len(findings), pyBad, pyExit, out, findings)
			}
			if goBad != c.wantBad {
				t.Fatalf("both agree bad=%v but the fixture expected bad=%v", goBad, c.wantBad)
			}
		})
	}
}

func gateByName(name string) func(*StagedDiff) ([]Finding, error) {
	for _, g := range PreCommitGates() {
		if g.Name == name {
			return g.Check
		}
	}
	return nil
}

func gitRun(t *testing.T, repo string, args ...string) {
	t.Helper()
	c := exec.Command("git", append([]string{"-C", repo}, args...)...)
	c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := c.CombinedOutput(); err != nil {
		t.Skipf("git %v failed: %s", args, out)
	}
}

func gitAddAll(t *testing.T, repo string, files map[string]string) {
	t.Helper()
	for p := range files {
		gitRun(t, repo, "add", "--", p)
	}
}

func writeFile(t *testing.T, repo, rel, content string) {
	t.Helper()
	full := filepath.Join(repo, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func parityLeakIP() string { return "100" + ".64.0.10" }
func parityGCPEmail() string {
	return "svc@proj." + "iam." + "gserviceaccount.com"
}
func parityInternalHost() string { return "ms" + "l-bui" + "ld-7" }

func TestParity_PublicLeak(t *testing.T) {
	runParity(t, "PUBLIC_LEAK", "scrub_public_copy.py", []parityCase{
		{"clean", map[string]string{"docs/a.md": "a perfectly normal doc line\n"}, false},
		{"ip-needle", map[string]string{"docs/a.md": "the host is " + parityLeakIP() + " internally\n"}, true},
		{"gcp-regex", map[string]string{"docs/a.md": "key " + parityGCPEmail() + "\n"}, true},
	})
}

func TestParity_SecretShape(t *testing.T) {
	runParity(t, "SECRET_SHAPE", "check_secret_shapes.py", []parityCase{
		{"clean", map[string]string{"a.md": "nothing operator-shaped here\n"}, false},
		{"placeholder-ok", map[string]string{"a.md": "see C:/Users/runner/go path\n"}, false},
		{"internal-host", map[string]string{"a.md": "connect to " + parityInternalHost() + " now\n"}, true},
		{"example-lab-ok", map[string]string{"a.md": "host gpu.example.lab is fake\n"}, false},
	})
}

func TestParity_FileAdmission(t *testing.T) {
	runParity(t, "FILE_ADMISSION", "check_committed_files.py", []parityCase{
		{"clean", map[string]string{"src/x.go": "package x\n"}, false},
		{"secrets-dir", map[string]string{"secrets/db.txt": "pw\n"}, true},
		{"pycache", map[string]string{"a/__pycache__/x.pyc": "junk\n"}, true},
	})
}

func TestParity_DocPlacement(t *testing.T) {
	runParity(t, "DOC_PLACEMENT", "check_doc_placement.py", []parityCase{
		{"clean-allowlisted", map[string]string{"ROADMAP.md": "# roadmap\n"}, false},
		{"clean-subdir", map[string]string{"docs/note.md": "# note\n"}, false},
		{"bad-root", map[string]string{"MY-RANDOM-DOC.md": "# stray\n"}, true},
	})
}

func TestParity_Provenance(t *testing.T) {
	runParity(t, "PROVENANCE_LABEL", "check_provenance_labels.py", []parityCase{
		{"clean", map[string]string{"README.md": "a normal sentence\n"}, false},
		{"modeled-ok", map[string]string{"README.md": "WebVoyager 643-task modeled 9.7x\n"}, false},
		{"measured-bad", map[string]string{"README.md": "WebVoyager 643-task measured 9.7x speedup\n"}, true},
	})
}

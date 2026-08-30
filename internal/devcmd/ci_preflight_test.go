package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/armbench"
	"github.com/anthony-chaudhary/fak/internal/committedbuildwitness"
	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestCIPreflightGoArgsArePathPortable(t *testing.T) {
	tests := []struct {
		name string
		got  []string
		want []string
	}{
		{"generated check", ciPreflightDisambiguationArgs(), []string{"run", "-trimpath", "./cmd/fak", "disambiguation", "generate", "--check", "--json"}},
		{"committed build", ciPreflightBuildArgs(), []string{"build", "-trimpath", "./..."}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if !reflect.DeepEqual(tc.got, tc.want) {
				t.Fatalf("args = %v, want exact %v", tc.got, tc.want)
			}
		})
	}
}

// ci_preflight_test.go — proves `fak ci-preflight` reads the COMMITTED tip (not the working
// tree): it seeds a real temp git repo, commits a clean/dirty tip, and asserts the verdict.
// The whole point of the verb is immunity to the peer-dirty tree, so every case commits the
// bytes under test and then dirties the working tree to show the verdict does not move.

// seedCIPreflightRepo returns a temp repo with an initial commit and a git() helper.
func seedCIPreflightRepo(t *testing.T) (repo string, git func(args ...string) (string, error)) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo = t.TempDir()
	emptyHooks := t.TempDir() // no hooks => the commit-gate hooks don't fire
	git = func(args ...string) (string, error) {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t",
		)
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if _, err := git("init", "-q", "-b", "main"); err != nil {
		if _, e2 := git("init", "-q"); e2 != nil {
			t.Skipf("git init failed: %v", e2)
		}
		_, _ = git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	if _, err := git("config", "core.hooksPath", emptyHooks); err != nil {
		t.Skipf("git config failed: %v", err)
	}
	return repo, git
}

// commitFiles writes files (path->content, relative to repo) and commits them by add-all.
func commitFiles(t *testing.T, repo string, git func(args ...string) (string, error), msg string, files map[string]string) {
	t.Helper()
	for rel, content := range files {
		full := filepath.Join(repo, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if out, err := git("add", "-A"); err != nil {
		t.Fatalf("git add: %s", out)
	}
	if out, err := git("commit", "-qm", msg); err != nil {
		t.Skipf("commit failed (likely no git identity): %s", out)
	}
}

func runPreflightJSON(t *testing.T, argv []string) (ciPreflightResult, int) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := RunCIPreflight(&stdout, &stderr, argv)
	var res ciPreflightResult
	if stdout.Len() > 0 {
		if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
			t.Fatalf("ci-preflight --json did not emit valid JSON: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
		}
	}
	return res, code
}

// A clean, gofmt-formatted, self-contained module — its committed tip must verify OK.
const cleanGoMod = "module cipreflight.test\n\ngo 1.26\n"
const cleanGoFile = "package p\n\n// Add returns a + b.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"

func TestCIPreflight_cleanCommittedTip_OK(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod":   cleanGoMod,
		"p/p.go":   cleanGoFile,
		"seed.txt": "seed\n",
	})
	// Dirty the working tree with an UNCOMMITTED gofmt violation — the verdict must ignore it,
	// because ci-preflight reads the committed tip only.
	if err := os.WriteFile(filepath.Join(repo, "p", "dirty.go"), []byte("package p\nfunc  Bad(){}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code := runPreflightJSON(t, []string{"--root", repo, "--json"})
	if !res.OK || code != 0 {
		t.Fatalf("clean committed tip should verify OK; got OK=%v code=%d failures=%+v", res.OK, code, res.Failures)
	}
	if !committedbuildwitness.Fresh(repo, res.Tip, time.Now()) {
		t.Fatal("successful ci-preflight did not publish committed build witness")
	}
}

func TestCIPreflight_committedGofmtViolation_fails(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not on PATH")
	}
	repo, git := seedCIPreflightRepo(t)
	// A committed, badly-formatted file: double space after func, no gofmt.
	commitFiles(t, repo, git, "unformatted", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": "package p\nfunc  Bad( ) int {return  1}\n",
	})
	res, code := runPreflightJSON(t, []string{"--root", repo, "--json", "--skip-build"})
	if res.OK || code != 1 {
		t.Fatalf("committed gofmt violation should fail; got OK=%v code=%d", res.OK, code)
	}
	var sawGofmt bool
	for _, f := range res.Failures {
		if f.Step == "gofmt" && len(f.Files) > 0 {
			sawGofmt = true
		}
	}
	if !sawGofmt {
		t.Fatalf("expected a gofmt failure listing files; got %+v", res.Failures)
	}
}

func TestCIPreflight_committedBuildBreak_fails(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	repo, git := seedCIPreflightRepo(t)
	// A committed file referencing an undefined symbol -> `go build ./...` fails.
	commitFiles(t, repo, git, "buildbreak", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": "package p\n\nfunc Use() int {\n\treturn undefinedSymbol()\n}\n",
	})
	res, code := runPreflightJSON(t, []string{"--root", repo, "--json"})
	if res.OK || code != 1 {
		t.Fatalf("committed build break should fail; got OK=%v code=%d", res.OK, code)
	}
	var sawBuild bool
	for _, f := range res.Failures {
		if f.Step == "build" && f.Detail != "" {
			sawBuild = true
		}
	}
	if !sawBuild {
		t.Fatalf("expected a build failure with detail; got %+v", res.Failures)
	}
}

func TestCIPreflightRejectsStaleCommittedDisambiguationArtifact(t *testing.T) {
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "disambiguation fixture", map[string]string{
		"go.mod":          "module example.test/ci\n\ngo 1.26\n",
		"cmd/fak/main.go": disambiguationFixtureMain,
		"docs/generated/disambiguation-index.json": "stale\n",
	})
	res, code := runPreflightJSON(t, []string{"--root", repo, "--skip-build", "--json"})
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	for _, failure := range res.Failures {
		if failure.Step == "disambiguation-generated" {
			return
		}
	}
	t.Fatalf("missing disambiguation-generated failure: %+v", res.Failures)
}

func TestCIPreflightAcceptsFreshCommittedDisambiguationArtifact(t *testing.T) {
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "disambiguation fixture", map[string]string{
		"go.mod":          "module example.test/ci\n\ngo 1.26\n",
		"cmd/fak/main.go": disambiguationFixtureMain,
		"docs/generated/disambiguation-index.json": "fresh\n",
	})
	res, code := runPreflightJSON(t, []string{"--root", repo, "--skip-build", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
}

func TestCIPreflightRejectsStaleCommittedStudySelfInventory(t *testing.T) {
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "seed", map[string]string{"README.md": "one\n"})
	manifest, err := studymonitor.BuildSelfInventory(repo, "anthony-chaudhary/fak", studymonitor.DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	var data bytes.Buffer
	if err := studymonitor.WriteSelfInventory(&data, manifest); err != nil {
		t.Fatal(err)
	}
	commitFiles(t, repo, git, "manifest", map[string]string{studymonitor.DefaultSelfInventoryPath: data.String()})
	commitFiles(t, repo, git, "tracked mutation", map[string]string{"README.md": "two\n"})

	res, code := runPreflightJSON(t, []string{"--root", repo, "--skip-build", "--json"})
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	for _, failure := range res.Failures {
		if failure.Step == "study-self-inventory" && strings.Contains(failure.Detail, "[content_changed] README.md") {
			return
		}
	}
	t.Fatalf("missing typed study-self-inventory failure: %+v", res.Failures)
}

func TestCIPreflightAcceptsFreshCommittedStudySelfInventory(t *testing.T) {
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "seed", map[string]string{"README.md": "one\n"})
	manifest, err := studymonitor.BuildSelfInventory(repo, "anthony-chaudhary/fak", studymonitor.DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	var data bytes.Buffer
	if err := studymonitor.WriteSelfInventory(&data, manifest); err != nil {
		t.Fatal(err)
	}
	commitFiles(t, repo, git, "manifest", map[string]string{studymonitor.DefaultSelfInventoryPath: data.String()})

	res, code := runPreflightJSON(t, []string{"--root", repo, "--skip-build", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
}

func TestCIPreflightStudySelfInventoryMutationDiagnostics(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   []string
	}{
		{
			name: "add", mutate: func(t *testing.T, root string) {
				writeStudySelfFile(t, root, "b.go", "package b\n")
			},
			want: []string{"[path_added] b.go"},
		},
		{
			name: "delete", mutate: func(t *testing.T, root string) {
				if err := os.Remove(filepath.Join(root, "a.md")); err != nil {
					t.Fatal(err)
				}
			},
			want: []string{"[path_removed] a.md"},
		},
		{
			name: "rename", mutate: func(t *testing.T, root string) {
				if err := os.Rename(filepath.Join(root, "a.md"), filepath.Join(root, "z.md")); err != nil {
					t.Fatal(err)
				}
			},
			want: []string{"[path_removed] a.md", "[path_added] z.md"},
		},
		{
			name: "reclassification", mutate: func(t *testing.T, root string) {
				mutateStudySelfManifestClass(t, root, "runtime_source")
			},
			want: []string{"[classification_changed] a.md"},
		},
		{
			name: "unknown class", mutate: func(t *testing.T, root string) {
				mutateStudySelfManifestClass(t, root, "unknown_future_class")
			},
			want: []string{"[classification_changed] a.md"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeStudySelfFile(t, root, "a.md", "original\n")
			manifest, err := studymonitor.BuildSelfInventory(root, "anthony-chaudhary/fak", studymonitor.DefaultSelfInventoryPath)
			if err != nil {
				t.Fatal(err)
			}
			var data bytes.Buffer
			if err := studymonitor.WriteSelfInventory(&data, manifest); err != nil {
				t.Fatal(err)
			}
			writeStudySelfFile(t, root, studymonitor.DefaultSelfInventoryPath, data.String())
			tt.mutate(t, root)
			detail, checked, ok := checkStudySelfInventory(root)
			if !checked || ok {
				t.Fatalf("checked=%v ok=%v detail=%q", checked, ok, detail)
			}
			for _, want := range tt.want {
				if !strings.Contains(detail, want) {
					t.Fatalf("detail=%q missing %q", detail, want)
				}
			}
		})
	}
}

const disambiguationFixtureMain = `package main

import (
	"bytes"
	"flag"
	"os"
)

func main() {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	check := fs.Bool("check", false, "")
	_ = fs.Bool("json", false, "")
	_ = fs.Parse(os.Args[3:])
	got, _ := os.ReadFile("docs/generated/disambiguation-index.json")
	if *check && !bytes.Equal(got, []byte("fresh\n")) {
		os.Exit(1)
	}
}
`

func TestArmbenchWitnessDriftRejectsStaleArtifact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "docs", "_witnesses", "armbench-selfcheck-2026-08-13.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	checked, detail, err := armbenchWitnessDrift(dir)
	if err != nil || !checked || !strings.Contains(detail, "stale") {
		t.Fatalf("checked=%v detail=%q err=%v", checked, detail, err)
	}
}

func TestArmbenchWitnessDriftAcceptsFreshArtifact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "docs", "_witnesses", "armbench-selfcheck-2026-08-13.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	res, err := armbench.Selfcheck()
	if err != nil {
		t.Fatal(err)
	}
	data, err := armbench.MarshalSelfcheck(res)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	checked, detail, err := armbenchWitnessDrift(dir)
	if err != nil || !checked || detail != "" {
		t.Fatalf("checked=%v detail=%q err=%v", checked, detail, err)
	}
}

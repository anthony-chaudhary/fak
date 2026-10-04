package selfupdatecmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// One real repository and stamped binary exercise the public CLI contract. The
// existing TestSelfUpdateAttempt tests cover the lower-level preparation seam.
// fak-test:runtime medium est=3s lane=default
func TestSelfUpdateSource(t *testing.T) {
	const helperEnv = "FAK_TEST_SELFUPDATE_SOURCE_ARGS"
	if encoded := os.Getenv(helperEnv); encoded != "" {
		var args []string
		if err := json.Unmarshal([]byte(encoded), &args); err != nil {
			panic(err)
		}
		Run(args)
		os.Exit(0)
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is required for explicit-source acceptance")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	root := t.TempDir()
	remote, seed, repo := filepath.Join(root, "origin.git"), filepath.Join(root, "seed"), filepath.Join(root, "checkout")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(filepath.Join(home, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(home, "bin", "fak"+exeSuffix())
	git := func(dir string, args ...string) string {
		t.Helper()
		return mustSelfUpdateGit(t, ctx, dir, args...)
	}
	write := func(path, body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git(root, "init", "--bare", remote)
	git(root, "init", "-b", "main", seed)
	git(seed, "config", "user.name", "FAK Test")
	git(seed, "config", "user.email", "fak-test@example.invalid")
	write(filepath.Join(seed, "go.mod"), "module example.com/sourcefixture\n\ngo 1.26\n")
	write(filepath.Join(seed, "main.go"), "package main\nfunc main() {}\n")
	git(seed, "add", "go.mod", "main.go")
	git(seed, "commit", "-m", "selected A")
	selected := git(seed, "rev-parse", "HEAD")
	git(seed, "branch", "reviewed")
	git(seed, "tag", "-a", "release", "-m", "selected annotated tag")
	build := exec.CommandContext(ctx, "go", "build", "-buildvcs=true", "-o", target, ".")
	build.Dir = seed
	build.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-p=1", "GOMAXPROCS=2")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build stamped fixture: %v\n%s", err, out)
	}
	if stamp, ok := stampOfBinary(target); !ok || !stamp.HasVCS || stamp.Dirty || stamp.Revision != selected {
		t.Fatalf("fixture stamp=%+v ok=%v; want clean selected A=%s", stamp, ok, selected)
	}
	write(filepath.Join(seed, "README.md"), "main B\n")
	git(seed, "add", "README.md")
	git(seed, "commit", "-m", "main B")
	main := git(seed, "rev-parse", "HEAD")
	git(seed, "remote", "add", "origin", remote)
	git(seed, "push", "origin", "main", "reviewed", "refs/tags/release")
	git(root, "clone", "--branch", "main", remote, repo)
	// A same-named local branch must not override the selected origin branch.
	git(repo, "branch", "reviewed", main)
	git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
	git(repo, "symbolic-ref", "refs/tags/alias", "refs/remotes/origin/main")
	write(filepath.Join(repo, "main.go"), "package main\nfunc main() {}\n// ordinary peer WIP\n")
	write(filepath.Join(repo, "untracked.txt"), "ordinary peer WIP\n")
	fetchHead := filepath.Join(repo, ".git", "FETCH_HEAD")
	write(fetchHead, "check-only sentinel\n")
	targetBefore, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	refsBefore := git(repo, "show-ref")
	treeBefore := git(repo, "status", "--porcelain")
	worktreesBefore := git(repo, "worktree", "list", "--porcelain")

	run := func(t *testing.T, allowFetch bool, args ...string) (selfUpdateReceipt, string, error) {
		t.Helper()
		args = append([]string{"--root", repo, "--target", target, "--json"}, args...)
		encoded, err := json.Marshal(args)
		if err != nil {
			t.Fatal(err)
		}
		trace := filepath.Join(t.TempDir(), "git-trace.jsonl")
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestSelfUpdateSource$")
		cmd.Env = append(os.Environ(), helperEnv+"="+string(encoded), "HOME="+home, "USERPROFILE="+home,
			"FAK_SELF_UPDATE_INSTALLER=native", "GIT_TRACE2_EVENT="+filepath.ToSlash(trace))
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		runErr := cmd.Run()
		var receipt selfUpdateReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("CLI did not return one JSON receipt: %v (exit %v)\nstdout=%s\nstderr=%s", err, runErr, stdout.String(), stderr.String())
		}
		traceBytes, err := os.ReadFile(trace)
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		fetches := 0
		for _, line := range bytes.Split(traceBytes, []byte{'\n'}) {
			var event struct {
				Event string   `json:"event"`
				Argv  []string `json:"argv"`
			}
			if json.Unmarshal(line, &event) != nil || event.Event != "start" {
				continue
			}
			command := strings.Join(event.Argv, " ")
			if strings.Contains(command, " fetch ") {
				fetches++
				if !allowFetch || !strings.Contains(command, "origin +refs/heads/reviewed:refs/remotes/origin/reviewed") {
					t.Errorf("unexpected source fetch: %s", command)
				}
			}
			if strings.Contains(command, " worktree add ") {
				t.Errorf("selection/check/refusal invoked mutating Git: %s", command)
			}
		}
		if allowFetch && fetches != 1 {
			t.Errorf("normal update fetched %d times, want exactly one selected-ref fetch", fetches)
		}
		if got := git(repo, "show-ref"); got != refsBefore {
			t.Errorf("CLI changed refs: before=%s after=%s", refsBefore, got)
		}
		if got := git(repo, "status", "--porcelain"); got != treeBefore {
			t.Errorf("CLI changed ordinary dirty checkout: %q -> %q", treeBefore, got)
		}
		if got := git(repo, "worktree", "list", "--porcelain"); got != worktreesBefore {
			t.Errorf("CLI created a build worktree: %s", got)
		}
		if got, err := os.ReadFile(fetchHead); !allowFetch && (err != nil || string(got) != "check-only sentinel\n") {
			t.Errorf("CLI fetched or changed FETCH_HEAD: %q err=%v", got, err)
		}
		if got, err := os.ReadFile(target); err != nil || !bytes.Equal(got, targetBefore) {
			t.Errorf("CLI changed installed target: err=%v", err)
		}
		if receipt.Attempted != 0 || receipt.Changed != 0 {
			t.Errorf("check/refusal reported install attempts: %+v", receipt)
		}
		return receipt, stderr.String(), runErr
	}

	var captured string
	for _, tc := range []struct {
		name, selector, revision, posture string
		args                              []string
	}{
		{"default", "", main, "stale/SKEWED", nil},
		{"origin branch", "origin/reviewed", selected, "fresh/FRESH", []string{"--ref", "origin/reviewed"}},
		{"bare branch", "reviewed", selected, "fresh/FRESH", []string{"--ref", "reviewed"}},
		{"annotated tag", "refs/tags/release", selected, "fresh/FRESH", []string{"--ref", "refs/tags/release"}},
		{"full revision", selected, selected, "fresh/FRESH", []string{"--revision", selected}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt, stderr, err := run(t, false, append([]string{"--check"}, tc.args...)...)
			if err != nil {
				t.Fatalf("check refused valid selector: %v\n%s", err, stderr)
			}
			if receipt.OldRevision == nil || *receipt.OldRevision != selected || receipt.NewRevision == nil || *receipt.NewRevision != tc.revision {
				t.Fatalf("wrong installed/selected commit: %+v; want old=%s new=%s", receipt, selected, tc.revision)
			}
			if !strings.Contains(receipt.Detail, tc.posture) {
				t.Errorf("comparison or ancestry ignored selected commit: detail=%q want %s", receipt.Detail, tc.posture)
			}
			if tc.selector != "" && (!strings.Contains(receipt.Detail, "source="+tc.selector+" ") || !strings.Contains(receipt.Detail, "revision="+tc.revision)) {
				t.Errorf("check omitted selector/full resolved SHA: %q", receipt.Detail)
			}
			if tc.name == "origin branch" {
				captured = *receipt.NewRevision
			}
		})
	}
	t.Run("normal update fetches selected branch", func(t *testing.T) {
		// The local tracking ref is stale, while origin and the installed binary
		// both identify A. An origin/main fallback would incorrectly build B.
		git(repo, "update-ref", "refs/remotes/origin/reviewed", main)
		receipt, stderr, err := run(t, true, "--ref", "origin/reviewed")
		if err != nil || receipt.NewRevision == nil || *receipt.NewRevision != selected || !strings.Contains(stderr, "source=origin/reviewed revision="+selected) {
			t.Errorf("normal update did not select current origin A: exit=%v receipt=%+v stderr=%s", err, receipt, stderr)
		}
		write(fetchHead, "check-only sentinel\n")
	})

	for _, tc := range []struct {
		name, token string
		args        []string
	}{
		{"empty ref", "SOURCE_INVALID", []string{"--force", "--ref="}},
		{"empty revision", "SOURCE_INVALID", []string{"--force", "--revision="}},
		{"abbreviated revision", "SOURCE_INVALID", []string{"--force", "--revision", selected[:12]}},
		{"dirty revision", "SOURCE_INVALID", []string{"--force", "--revision", selected + "+uncommitted"}},
		{"missing ref", "SOURCE_UNRESOLVED", []string{"--check", "--ref", "origin/missing"}},
		{"missing object", "SOURCE_UNRESOLVED", []string{"--check", "--revision", strings.Repeat("0", 40)}},
		{"noncommit object", "SOURCE_UNRESOLVED", []string{"--check", "--revision", git(repo, "rev-parse", "HEAD:main.go")}},
		{"symbolic alias", "SOURCE_INVALID", []string{"--check", "--ref", "origin/HEAD"}},
		{"symbolic tag alias", "SOURCE_INVALID", []string{"--check", "--ref", "refs/tags/alias"}},
		{"conflicting selectors", "SOURCE_CONFLICT", []string{"--force", "--ref", "origin/reviewed", "--revision", selected}},
		{"manifest", "SOURCE_CONFLICT", []string{"--force", "--ref", "origin/reviewed", "--manifest-url", "https://updates.example.invalid/manifest.json"}},
		{"MSIX", "SOURCE_CONFLICT", []string{"--force", "--revision", selected, "--installer", "msix"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt, stderr, err := run(t, false, tc.args...)
			if err == nil || receipt.Status != "prepare_failed" || !strings.Contains(receipt.Detail+stderr, tc.token) {
				t.Errorf("selector refusal missing %s: exit=%v receipt=%+v stderr=%s", tc.token, err, receipt, stderr)
			}
			if receipt.NewRevision != nil {
				t.Errorf("refused selector fell back to revision %q", *receipt.NewRevision)
			}
		})
	}

	if captured == "" {
		t.Fatal("valid branch selection produced no captured commit")
	}
	// Carry the actual CLI selection into the existing attempt API after the
	// mutable branch advances. Preparation and build provenance must still bind A.
	git(remote, "update-ref", "refs/heads/reviewed", main)
	git(repo, "update-ref", "refs/remotes/origin/reviewed", main)
	buildDir := filepath.Join(root, "selected-build")
	dir, cleanup, err := prepareSelfUpdateAttempt(ctx, selfinstall.RealRunner, repo, captured, buildDir)
	if err != nil {
		t.Fatalf("prepare captured selector: %v", err)
	}
	defer cleanup()
	if got := git(dir, "rev-parse", "HEAD"); got != selected {
		t.Errorf("prepared HEAD followed moving selector: got %s want captured A %s", got, selected)
	}
	if opts := selfUpdateAttemptOptions(dir, target, captured); opts.ExpectedCommit != selected {
		t.Errorf("build provenance differs from captured selection: got %s want %s", opts.ExpectedCommit, selected)
	}
}

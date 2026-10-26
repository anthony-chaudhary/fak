package witness

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const symptomWorkspaceEnvHelper = "FAK_WITNESS_WORKSPACE_ENV_HELPER"

func init() {
	if os.Getenv(symptomWorkspaceEnvHelper) == "1" {
		fmt.Print(os.Getenv("GOWORK"))
		os.Exit(0)
	}
}

func nestedWorkspaceSymptomFixture(t *testing.T, foreignWorkspace bool) (abi.WitnessOutcome, string) {
	requireGoAndGit(t)
	base := t.TempDir()
	companion := filepath.Join(base, "companion")
	repository := filepath.Join(base, "kernel")
	managed := filepath.Join(repository, ".worktrees", "candidate")
	foreign := filepath.Join(base, "foreign")
	for _, dir := range []string{companion, repository, foreign} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	writeRepoFile(t, companion, "go.mod", "module example.invalid/companion\n\ngo 1.21\n")
	writeRepoFile(t, companion, "value.go", "package companion\n\nfunc Value() int { return 7 }\n")
	writeRepoFile(t, foreign, "go.mod", "module example.invalid/foreign\n\ngo 1.21\n")
	writeRepoFile(t, foreign, "go.work", "go 1.21\n\nuse .\n")

	gitIn(t, repository, "init", "-q")
	gitIn(t, repository, "config", "user.email", "witness@example.invalid")
	gitIn(t, repository, "config", "user.name", "witness")
	writeRepoFile(t, repository, ".gitignore", ".worktrees/\n")
	writeRepoFile(t, repository, "go.mod", "module example.invalid/kernel\n\ngo 1.21\n\nrequire example.invalid/companion v0.0.0\n\nreplace example.invalid/companion => ../companion\n")
	writeRepoFile(t, repository, "go.work", "go 1.21\n\nuse (\n\t.\n\t../companion\n)\n")
	writeRepoFile(t, repository, "value.go", "package kernel\n\nfunc Value() int { return 0 }\n")
	gitIn(t, repository, "add", ".gitignore", "go.mod", "go.work", "value.go")
	gitIn(t, repository, "commit", "-q", "-m", "parent: incorrect value")

	writeRepoFile(t, repository, "value.go", "package kernel\n\nfunc Value() int { return -1 }\n")
	writeRepoFile(t, repository, "value_test.go", `package kernel

import (
	"testing"

	"example.invalid/companion"
)

func TestNegativeValue(t *testing.T) {
	if companion.Value() != 7 {
		t.Fatal("companion workspace unavailable")
	}
	if Value() != -1 {
		t.Fatalf("Value()=%d, want -1", Value())
	}
}
`)
	gitIn(t, repository, "add", "value.go", "value_test.go")
	gitIn(t, repository, "commit", "-q", "-m", "fix: return negative value")
	gitIn(t, repository, "worktree", "add", "--detach", "--quiet", managed, "HEAD")
	t.Cleanup(func() {
		gitIn(t, repository, "worktree", "remove", "--force", managed)
	})

	if foreignWorkspace {
		t.Setenv("GOWORK", filepath.Join(foreign, "go.work"))
	} else {
		t.Setenv("GOWORK", "")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	var executionOutput string
	execRun := func(ctx context.Context, dir string, argv ...string) (string, int, error) {
		out, code, err := commandRunner(ctx, dir, argv...)
		if strings.TrimSpace(out) != "" {
			executionOutput += strings.Join(argv, " ") + ":\n" + out
			if len(executionOutput) > 4096 {
				executionOutput = executionOutput[len(executionOutput)-4096:]
			}
		}
		return out, code, err
	}
	got, detail := NewWithRunners(gitRunner, execRun, managed).ResolveSymptomWithDetail(ctx, "HEAD", true)
	assertRepoClean(t, managed)
	if got != abi.WitnessConfirmed && executionOutput != "" {
		detail += "\nactual command output:\n" + executionOutput
	}
	return got, detail
}

// fak-test:runtime slow est=15s lane=default
func TestSymptomNestedWorktreePreservesCommittedSiblingWorkspace(t *testing.T) {
	got, detail := nestedWorkspaceSymptomFixture(t, false)
	if got != abi.WitnessConfirmed {
		t.Fatalf("nested-worktree symptom=%v (%s), want confirmed candidate-pass/parent-fail", got, detail)
	}
}

// fak-test:runtime slow est=15s lane=default
func TestSymptomScratchIgnoresForeignAmbientWorkspace(t *testing.T) {
	got, detail := nestedWorkspaceSymptomFixture(t, true)
	if got != abi.WitnessConfirmed {
		t.Fatalf("foreign-workspace symptom=%v (%s), want confirmed candidate-pass/parent-fail", got, detail)
	}
}

func symptomWorkspaceWriteModule(t *testing.T, dir, module string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeRepoFile(t, dir, "go.mod", "module "+module+"\n\ngo 1.21\n")
	writeRepoFile(t, dir, "go.work", "go 1.21\n\nuse .\n")
	return filepath.Join(dir, "go.work")
}

// fak-test:runtime medium est=2s lane=default
func TestCommandRunnerPreservesForeignWorkspaceOutsideRegisteredScratch(t *testing.T) {
	requireGoAndGit(t)
	foreign := symptomWorkspaceWriteModule(t, filepath.Join(t.TempDir(), "foreign"), "example.invalid/foreign")
	t.Setenv("GOWORK", foreign)
	unregistered, err := os.MkdirTemp(t.TempDir(), execWitnessPattern())
	if err != nil {
		t.Fatal(err)
	}
	symptomWorkspaceWriteModule(t, unregistered, "example.invalid/unregistered")
	goExecutable, err := exec.LookPath("go")
	if err != nil {
		t.Fatal(err)
	}
	out, code, runErr := commandRunner(context.Background(), unregistered, goExecutable, "env", "GOWORK")
	if runErr != nil || code != 0 {
		t.Fatalf("go env GOWORK error=%v code=%d", runErr, code)
	}
	if got := filepath.Clean(strings.TrimSpace(out)); got != filepath.Clean(foreign) {
		t.Fatalf("unregistered scratch-shaped cwd GOWORK=%q, want caller workspace %q", got, foreign)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestCommandRunnerScopesWorkspaceOnlyForActiveScratchGo(t *testing.T) {
	requireGoAndGit(t)
	foreign := symptomWorkspaceWriteModule(t, filepath.Join(t.TempDir(), "foreign"), "example.invalid/foreign")
	t.Setenv("GOWORK", foreign)
	repository := newExecutionRepo(t)
	symptomWorkspaceWriteModule(t, repository, "example.invalid/active")
	gitIn(t, repository, "add", "go.mod", "go.work")
	gitIn(t, repository, "commit", "-q", "-m", "active workspace")
	scratch, cleanup, err := NewExecutionVerifier(repository).scratchWorktree(context.Background(), "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	goExecutable, err := exec.LookPath("go")
	if err != nil {
		t.Fatal(err)
	}
	executables := []struct {
		name string
		path string
	}{
		{name: "path lookup", path: "go"},
		{name: "absolute", path: goExecutable},
	}
	if runtime.GOOS == "windows" {
		executables = append(executables, struct {
			name string
			path string
		}{name: "windows name", path: "go.exe"})
	}
	for _, executable := range executables {
		t.Run(executable.name, func(t *testing.T) {
			out, code, runErr := commandRunner(context.Background(), scratch, executable.path, "env", "GOWORK")
			if runErr != nil || code != 0 {
				t.Fatalf("go env GOWORK error=%v code=%d", runErr, code)
			}
			want := filepath.Join(scratch, "go.work")
			if got := filepath.Clean(strings.TrimSpace(out)); got != filepath.Clean(want) {
				t.Fatalf("active scratch GOWORK=%q, want committed workspace %q", got, want)
			}
		})
	}

	helpExecutable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(symptomWorkspaceEnvHelper, "1")
	out, code, runErr := commandRunner(context.Background(), scratch, helpExecutable)
	if runErr != nil || code != 0 {
		t.Fatalf("non-Go helper error=%v code=%d", runErr, code)
	}
	if got := filepath.Clean(strings.TrimSpace(out)); got != filepath.Clean(foreign) {
		t.Fatalf("non-Go active-scratch GOWORK=%q, want caller workspace %q", got, foreign)
	}
}

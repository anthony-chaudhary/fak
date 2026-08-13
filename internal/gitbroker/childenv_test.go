package gitbroker

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// envProbeEnv turns the copy of this test binary that fakeGitOnPath installs as
// `git` into a program that prints the environment it was handed. It is checked
// in init, not in a test, because the broker execs `git` with git's own
// arguments — there is no place to pass `-test.run`, and the child must answer
// before the testing framework would run anything.
//
// The name is deliberately outside git's control namespace: childEnv drops
// GIT_*/SSH_* it does not allow, so a probe named GIT_ANYTHING would be scrubbed
// on its way in and the fake git would run the whole test suite instead.
const envProbeEnv = "FAK_GITBROKER_ENV_PROBE"

func init() {
	if os.Getenv(envProbeEnv) != "1" {
		return
	}
	// This process IS the fake git; its stdout is the answer under test.
	for _, kv := range os.Environ() {
		_, _ = os.Stdout.WriteString(kv + "\n")
	}
	os.Exit(0)
}

// fakeGitOnPath installs a copy of this test binary as `git` at the front of
// PATH and returns nothing but the guarantee that the next spawn of `git`
// reports its environment. Copying the test binary is what makes this work
// identically on Windows and Linux: a shell script needs a shebang the one does
// not have, and CreateProcess will not start a .bat directly.
func fakeGitOnPath(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	name := "git"
	if runtime.GOOS == "windows" {
		name = "git.exe"
	}
	self, err := os.ReadFile(os.Args[0])
	if err != nil {
		t.Skipf("cannot read this test binary to install it as git: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), self, 0o755); err != nil {
		t.Fatalf("install fake git: %v", err)
	}
	t.Setenv(envProbeEnv, "1")
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

// childGitEnv runs one brokered invocation against the fake git and returns the
// environment the child actually observed.
func childGitEnv(t *testing.T, inv Invocation) map[string]string {
	t.Helper()
	out, err := New().Run(context.Background(), inv)
	if err != nil {
		t.Fatalf("brokered probe invocation: %v (stderr %q)", err, out.Stderr)
	}
	got := map[string]string{}
	for _, line := range strings.Split(strings.ReplaceAll(string(out.Stdout), "\r\n", "\n"), "\n") {
		if name, val, ok := strings.Cut(line, "="); ok && name != "" {
			got[strings.ToUpper(name)] = val
		}
	}
	if len(got) == 0 {
		t.Fatalf("fake git reported no environment at all: %q", out.Stdout)
	}
	return got
}

// TestBrokeredGitRunsWithAScrubbedEnvironment is the witness for #6489: every
// variable that can silently re-aim, re-configure or wedge a brokered git call
// is set in THIS process, and the child is asked what it actually got.
func TestBrokeredGitRunsWithAScrubbedEnvironment(t *testing.T) {
	hostile := map[string]string{
		"GIT_DIR":                 filepath.Join(t.TempDir(), "elsewhere.git"),
		"GIT_WORK_TREE":           t.TempDir(),
		"GIT_CONFIG_GLOBAL":       "hostile-global",
		"GIT_CONFIG_SYSTEM":       "hostile-system",
		"GIT_CONFIG_COUNT":        "1",
		"GIT_CONFIG_KEY_0":        "core.hooksPath",
		"GIT_CONFIG_VALUE_0":      "hostile-hooks",
		"GIT_CONFIG_PARAMETERS":   "'core.hooksPath=hostile-hooks'",
		"GIT_ASKPASS":             "hostile-askpass",
		"SSH_ASKPASS":             "hostile-askpass",
		"GIT_SSH":                 "hostile-ssh",
		"GIT_SSH_COMMAND":         "hostile-ssh --steer",
		"GIT_TERMINAL_PROMPT":     "1",
		"GIT_INDEX_FILE":          "ambient-index",
		"FAK_GITBROKER_UNRELATED": "kept",
	}
	for k, v := range hostile {
		t.Setenv(k, v)
	}
	fakeGitOnPath(t)

	got := childGitEnv(t, Invocation{Dir: t.TempDir(), Args: []string{"status"}})

	for _, name := range []string{
		"GIT_DIR", "GIT_WORK_TREE",
		"GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT",
		"GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_CONFIG_PARAMETERS",
		"GIT_ASKPASS", "SSH_ASKPASS", "GIT_SSH", "GIT_SSH_COMMAND",
	} {
		if v, ok := got[name]; ok {
			t.Errorf("%s reached brokered git as %q; ambient steering must not survive", name, v)
		}
	}
	if got["GIT_TERMINAL_PROMPT"] != "0" {
		t.Errorf("GIT_TERMINAL_PROMPT = %q, want %q: a headless worker can never answer a prompt",
			got["GIT_TERMINAL_PROMPT"], "0")
	}
	// GIT_INDEX_FILE is the one intentional exception, in BOTH directions: fak
	// also runs as a git hook, where git itself puts the commit's index in the
	// environment, and a hook that read the repository's real index instead
	// would be answering about a different commit than the one being made.
	if got["GIT_INDEX_FILE"] != hostile["GIT_INDEX_FILE"] {
		t.Errorf("GIT_INDEX_FILE = %q, want %q: the deliberate steering exception was scrubbed",
			got["GIT_INDEX_FILE"], hostile["GIT_INDEX_FILE"])
	}
	// Preserve what git genuinely needs to run on either host.
	for _, name := range []string{"PATH", "FAK_GITBROKER_UNRELATED"} {
		if _, ok := got[name]; !ok {
			t.Errorf("%s did not reach brokered git; the scrub took more than git's steering surface", name)
		}
	}
	if runtime.GOOS == "windows" {
		for _, name := range []string{"SYSTEMROOT", "USERPROFILE"} {
			if _, ok := got[name]; !ok {
				t.Errorf("%s did not reach brokered git on Windows", name)
			}
		}
	} else if _, ok := got["HOME"]; !ok {
		t.Error("HOME did not reach brokered git")
	}
}

// TestCallerSetIndexFileStillReachesGit pins the deliberate exception: a
// throwaway index named at the call site is how internal/patchcommit stages
// without touching the shared repository's real index, so the scrub must not
// take it. It is set as Invocation.Env, which is exactly how that path sets it.
func TestCallerSetIndexFileStillReachesGit(t *testing.T) {
	t.Setenv("GIT_DIR", filepath.Join(t.TempDir(), "elsewhere.git"))
	t.Setenv("GIT_INDEX_FILE", "ambient-index")
	fakeGitOnPath(t)

	index := filepath.Join(t.TempDir(), "throwaway-index")
	got := childGitEnv(t, Invocation{
		Dir:  t.TempDir(),
		Args: []string{"add", "."},
		Env:  []string{"GIT_INDEX_FILE=" + index, "GIT_CONFIG_COUNT=1"},
	})
	// The caller's value must also WIN over the ambient one: Invocation.Env is
	// applied last precisely so the call site outranks whatever launched fak.
	if got["GIT_INDEX_FILE"] != index {
		t.Errorf("caller-set GIT_INDEX_FILE = %q, want %q: patchcommit's throwaway index would land on the real one",
			got["GIT_INDEX_FILE"], index)
	}
	// Invocation.Env is a decision the call site made; only the ambient
	// environment is scrubbed, so a deliberate config override still lands.
	if got["GIT_CONFIG_COUNT"] != "1" {
		t.Errorf("caller-set GIT_CONFIG_COUNT = %q, want %q", got["GIT_CONFIG_COUNT"], "1")
	}
	if got["GIT_TERMINAL_PROMPT"] != "0" {
		t.Errorf("GIT_TERMINAL_PROMPT = %q, want %q", got["GIT_TERMINAL_PROMPT"], "0")
	}
}

// TestAmbientGitDirCannotResteerRealGit is the same claim against real git
// rather than a probe: with GIT_DIR and an injected config in this process, a
// brokered call still answers about the repository the broker named.
func TestAmbientGitDirCannotResteerRealGit(t *testing.T) {
	repo := runnerTestRepo(t)
	other := t.TempDir()
	if out, err := exec.Command("git", "-C", other, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init %s: %v: %s", other, err, out)
	}

	t.Setenv("GIT_DIR", filepath.Join(other, ".git"))
	t.Setenv("GIT_CONFIG_COUNT", "1")
	t.Setenv("GIT_CONFIG_KEY_0", "user.name")
	t.Setenv("GIT_CONFIG_VALUE_0", "ambient-injection")

	e := New()
	defer func() { _ = e.Close() }()
	ctx := context.Background()

	out, err := e.Run(ctx, Invocation{Dir: repo, DirAsFlag: true, Args: []string{"rev-parse", "--show-toplevel"}})
	if err != nil {
		t.Fatalf("rev-parse --show-toplevel: %v: %s", err, out.Stderr)
	}
	top, err := filepath.EvalSymlinks(strings.TrimSpace(string(out.Stdout)))
	if err != nil {
		t.Fatalf("resolve reported toplevel: %v", err)
	}
	want, err := filepath.EvalSymlinks(repo)
	if err != nil {
		t.Fatalf("resolve fixture repo: %v", err)
	}
	if !strings.EqualFold(filepath.Clean(top), filepath.Clean(want)) {
		t.Fatalf("brokered git resolved to %q, want %q: an ambient GIT_DIR re-aimed the call", top, want)
	}

	out, err = e.Run(ctx, Invocation{Dir: repo, DirAsFlag: true, Args: []string{"config", "--get", "user.name"}})
	if err != nil {
		t.Fatalf("config --get user.name: %v: %s", err, out.Stderr)
	}
	if name := strings.TrimSpace(string(out.Stdout)); name != "fak test" {
		t.Fatalf("brokered git read user.name = %q, want %q: ambient GIT_CONFIG_* injected config", name, "fak test")
	}
}

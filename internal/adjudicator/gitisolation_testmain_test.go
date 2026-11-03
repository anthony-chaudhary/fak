package adjudicator

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Vendored from internal/testgit (the shared harness is still consumed across
// the repo; the local copy keeps this lane's census coupling bounded); stdlib
// exec replaces sysproc.CommandContext whose only addition is Windows
// console-window suppression, cosmetic for a short-lived git config probe.

// repoLocatingGitEnv names the variables a git hook exports to point git at the
// repository under commit. A fixture git child must never inherit them.
var repoLocatingGitEnv = []string{
	"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
	"GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX",
	"GIT_NAMESPACE",
}

// isolateGit makes this process's git children hermetic: it unsets
// repoLocatingGitEnv, ignores the system config (GIT_CONFIG_NOSYSTEM=1), and
// points GIT_CONFIG_GLOBAL at a private copy of the caller's global config, so
// reads see the same settings (identity, safe.directory, core.longpaths) while
// a `git config --global` write lands in the copy. HOME is left alone: Go's
// module and build caches hang off it. The returned func removes the copy.
func isolateGit() (func(), error) {
	for _, name := range repoLocatingGitEnv {
		if err := os.Unsetenv(name); err != nil {
			return func() {}, fmt.Errorf("unset %s: %w", name, err)
		}
	}
	dir, err := os.MkdirTemp("", "fak-test-gitconfig-")
	if err != nil {
		return func() {}, fmt.Errorf("create isolated git config dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	global := filepath.Join(dir, "gitconfig")
	seed, err := globalGitConfigBytes()
	if err != nil {
		cleanup()
		return func() {}, err
	}
	if err := os.WriteFile(global, seed, 0o600); err != nil {
		cleanup()
		return func() {}, fmt.Errorf("write isolated git config: %w", err)
	}
	for k, v := range map[string]string{"GIT_CONFIG_GLOBAL": global, "GIT_CONFIG_NOSYSTEM": "1"} {
		if err := os.Setenv(k, v); err != nil {
			cleanup()
			return func() {}, fmt.Errorf("set %s: %w", k, err)
		}
	}
	return cleanup, nil
}

// globalGitConfigBytes returns the content git would read as the global config:
// GIT_CONFIG_GLOBAL when set, else $HOME/.gitconfig, else the XDG file. A
// missing file is an empty config.
func globalGitConfigBytes() ([]byte, error) {
	var candidates []string
	if p := os.Getenv("GIT_CONFIG_GLOBAL"); p != "" {
		candidates = append(candidates, p)
	} else {
		home := os.Getenv("HOME")
		if home == "" {
			home, _ = os.UserHomeDir()
		}
		if home != "" {
			candidates = append(candidates, filepath.Join(home, ".gitconfig"))
		}
		xdg := os.Getenv("XDG_CONFIG_HOME")
		if xdg == "" && home != "" {
			xdg = filepath.Join(home, ".config")
		}
		if xdg != "" {
			candidates = append(candidates, filepath.Join(xdg, "git", "config"))
		}
	}
	for _, p := range candidates {
		data, err := os.ReadFile(p)
		if err == nil {
			return data, nil
		}
		if !os.IsNotExist(err) {
			return nil, fmt.Errorf("read global git config %s: %w", p, err)
		}
	}
	return nil, nil
}

// runGitIsolated is a TestMain body: isolateGit, run the tests, then fail the
// run if the repo-local identity (user.* in the local or worktree scope) of the
// checkout enclosing the package directory changed while they ran. That
// tripwire names a fixture write that escaped isolation some other way (a
// helper whose cmd.Dir was empty, so git acted on the package's own checkout).
func runGitIsolated(m *testing.M) int {
	cleanup, err := isolateGit()
	if err != nil {
		fmt.Fprintf(os.Stderr, "testgit: isolate git: %v\n", err)
		return 1
	}
	defer cleanup()
	before, probed := checkoutIdentity(".")
	code := m.Run()
	if after, ok := checkoutIdentity("."); probed && ok && after != before {
		fmt.Fprintf(os.Stderr, "testgit: the enclosing checkout's repo-local git identity changed while these tests ran (this or a concurrent test binary wrote it):\n--- before\n%s--- after\n%s", before, after)
		if code == 0 {
			code = 1
		}
	}
	return code
}

// checkoutIdentity returns the user.* entries of the local and worktree config
// scopes of the repository enclosing dir, one "scope key value" per line, and
// whether the probe ran. Outside a repository it reports false.
func checkoutIdentity(dir string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "config", "--show-scope", "--get-regexp", `^user\.`)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		// Exit 1 is "no matching key": an empty identity is a valid snapshot.
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			return "", false
		}
	}
	var b strings.Builder
	for _, line := range strings.Split(strings.ReplaceAll(out.String(), "\r\n", "\n"), "\n") {
		scope, _, _ := strings.Cut(line, "\t")
		if scope == "local" || scope == "worktree" {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	return b.String(), true
}

// TestMain runs this package's tests with hermetic git: fixture
// `git config user.*` writes and commits can never reach the real checkout
// or the operator's global config, even under a git hook's exported GIT_DIR.
func TestMain(m *testing.M) { os.Exit(runGitIsolated(m)) }

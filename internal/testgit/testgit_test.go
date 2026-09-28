package testgit

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func gitIn(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out)
	}
	return strings.TrimSpace(string(out))
}

// TestIsolateGitContainsFixtureIdentityWrites is the isolation witness: with a
// hook-style GIT_DIR aimed at a decoy "real" checkout, a fixture helper's
// `git config user.name fak-test` must land in its own cmd.Dir repo, and a
// `--global` write must land in the isolated copy, never the seed.
func TestIsolateGitContainsFixtureIdentityWrites(t *testing.T) {
	decoy, fixture := t.TempDir(), t.TempDir()
	gitIn(t, decoy, "init", "-q")
	gitIn(t, fixture, "init", "-q")
	seed := filepath.Join(t.TempDir(), "seed.gitconfig")
	const seedText = "[core]\n\tlongpaths = true\n"
	if err := os.WriteFile(seed, []byte(seedText), 0o600); err != nil {
		t.Fatal(err)
	}
	// t.Setenv restores each variable IsolateGit rewrites.
	t.Setenv("GIT_CONFIG_GLOBAL", seed)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "")
	t.Setenv("GIT_DIR", filepath.Join(decoy, ".git"))
	t.Setenv("GIT_INDEX_FILE", filepath.Join(decoy, ".git", "index"))

	cleanup, err := IsolateGit()
	if err != nil {
		t.Fatalf("IsolateGit: %v", err)
	}
	t.Cleanup(cleanup)

	gitIn(t, fixture, "config", "user.name", "fak-test")
	gitIn(t, fixture, "config", "--global", "user.email", "test@fak.local")

	if got, ok := CheckoutIdentity(decoy); !ok || got != "" {
		t.Errorf("decoy checkout identity = %q (probed %v); want empty: GIT_DIR leaked a fixture write", got, ok)
	}
	if got, _ := CheckoutIdentity(fixture); !strings.Contains(got, "user.name fak-test") {
		t.Errorf("fixture identity = %q; want the fixture repo to hold user.name fak-test", got)
	}
	if data, _ := os.ReadFile(seed); string(data) != seedText {
		t.Errorf("seed global config was written: %q", data)
	}
	if got := gitIn(t, fixture, "config", "--global", "core.longpaths"); got != "true" {
		t.Errorf("isolated global config core.longpaths = %q; want the seed's settings carried over", got)
	}
}

func TestCheckoutIdentityOmitsGlobalScope(t *testing.T) {
	repo := t.TempDir()
	gitIn(t, repo, "init", "-q")
	seed := filepath.Join(t.TempDir(), "seed.gitconfig")
	if err := os.WriteFile(seed, []byte("[user]\n\tname = Global Person\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GIT_CONFIG_GLOBAL", seed)
	if got, ok := CheckoutIdentity(repo); !ok || got != "" {
		t.Fatalf("identity with only a global user = %q (probed %v); want empty", got, ok)
	}
	gitIn(t, repo, "config", "user.email", "test@fak.local")
	if got, _ := CheckoutIdentity(repo); got != "local\tuser.email test@fak.local\n" {
		t.Fatalf("identity = %q; want the local user.email line", got)
	}
}

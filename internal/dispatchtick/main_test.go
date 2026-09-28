package dispatchtick

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testgit"
)

// TestMain clears the fleet-wide FAK_SESSIONS_PER_ACCOUNT knob before running the
// package tests so the default-cap assertions observe the committed default rather
// than an ambient fleet override. Tests that exercise the knob set it explicitly
// with t.Setenv. It also runs the tests with hermetic git: fixture
// `git config user.*` writes and commits can never reach the real checkout or
// the operator's global config, even under a git hook's exported GIT_DIR.
func TestMain(m *testing.M) {
	os.Unsetenv(SessionsPerAccountEnv)
	os.Exit(testgit.RunGitIsolated(m))
}

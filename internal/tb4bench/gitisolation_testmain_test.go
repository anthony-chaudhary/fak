package tb4bench

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testgit"
)

// TestMain runs this package's tests with hermetic git: fixture
// `git config user.*` writes and commits can never reach the real checkout
// or the operator's global config, even under a git hook's exported GIT_DIR.
func TestMain(m *testing.M) { os.Exit(testgit.RunGitIsolated(m)) }

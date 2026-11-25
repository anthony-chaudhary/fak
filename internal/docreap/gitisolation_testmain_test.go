package docreap

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testgit"
)

// TestMain runs this package's tests with hermetic git so fixture commits can never
// reach the real checkout or the operator's global config.
func TestMain(m *testing.M) { os.Exit(testgit.RunGitIsolated(m)) }

//go:build windows

package tokencache

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// TestCommonDirCmdSuppressesConsoleWindow is the #5129 regression witness: the
// `git rev-parse --git-common-dir` probe that commonDir spawns must carry the
// Windows no-window flags. An unsuppressed spawn flashes a console window under
// background automation and trips the pre-push DESKTOP_POPUP_REGRESSION guard,
// which — because the guard scans staged files — stalls every worker's push even
// on commits that never touch internal/tokencache.
func TestCommonDirCmdSuppressesConsoleWindow(t *testing.T) {
	cmd := commonDirCmd("")
	if cmd.SysProcAttr == nil {
		t.Fatal("commonDir git spawn has nil SysProcAttr; console window not suppressed (#5129)")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Fatal("commonDir git spawn HideWindow=false; console window not suppressed (#5129)")
	}
	if cmd.SysProcAttr.CreationFlags&windowgate.CreateNoWindow == 0 {
		t.Fatalf("commonDir git spawn CreationFlags=%#x missing CREATE_NO_WINDOW (#5129)", cmd.SysProcAttr.CreationFlags)
	}
}

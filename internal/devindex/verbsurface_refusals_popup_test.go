package devindex

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// prePopupFixShape mirrors the exact launch shape vsGitTracked used before #6484:
// a bare exec.Command console tool that reaches Run without the Windows no-window
// hook. It is the red half of the witness, kept as source rather than a git blob so
// the guard below can prove it is not vacuous without shelling out.
const prePopupFixShape = `package devindex

func vsGitTracked(root, rel string) bool {
	cmd := exec.Command("git", "-C", root, "ls-files", "--error-unmatch", "--", rel)
	return cmd.Run() == nil
}
`

// TestDevindexPopupGuardIsLive pins the red half: the windowgate scanner must still
// flag the pre-fix launch shape. Without this, a detector change could silently make
// TestDevindexGoLaunchesAreWindowSuppressed pass for the wrong reason.
func TestDevindexPopupGuardIsLive(t *testing.T) {
	findings := windowgate.GoExecCandidates("internal/devindex/verbsurface_refusals.go", prePopupFixShape)
	if len(findings) == 0 {
		t.Fatal("windowgate no longer flags the pre-#6484 exec.Command shape — the popup guard below is vacuous")
	}
	for _, f := range findings {
		if !strings.Contains(f, windowgate.ReasonGoUnsuppressedExec) {
			t.Errorf("finding missing the %s reason code: %s", windowgate.ReasonGoUnsuppressedExec, f)
		}
	}
}

// TestDevindexGoLaunchesAreWindowSuppressed is the green half: every helper
// subprocess in this package stays off the windowgate popup watchlist. `fak
// windowgate report --json` named vsGitTracked's `git ls-files` probe under
// UNSUPPRESSED_GO_EXEC (#6484) — a windowless Windows parent flashes a console child
// when the command reaches Run unconfigured. Scanning the whole package, not just
// the one file that regressed, means the next unguarded launch here fails here too.
func TestDevindexGoLaunchesAreWindowSuppressed(t *testing.T) {
	root := repoRootForSurface(t)
	pkgDir := filepath.Join(root, "internal", "devindex")
	entries, err := os.ReadDir(pkgDir)
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	var rels []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		rels = append(rels, "internal/devindex/"+e.Name())
	}
	if len(rels) == 0 {
		t.Fatal("no .go files found in internal/devindex")
	}
	report, err := windowgate.ScanFiles(root, rels)
	if err != nil {
		t.Fatalf("windowgate.ScanFiles: %v", err)
	}
	for _, violation := range report.GoExecs {
		t.Errorf("unsuppressed Go helper launch: %s", violation)
	}
	for _, candidate := range report.GoCandidates {
		t.Errorf("console-tool launch on the popup watchlist: %s", candidate)
	}
}

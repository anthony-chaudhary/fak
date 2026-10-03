package selfupdatecmd

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fakroot"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func isFullGitCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !strings.ContainsRune("0123456789abcdefABCDEF", c) {
			return false
		}
	}
	return true
}

func discoverGitCommonDir(root string) string {
	cmd := exec.Command("git", "-C", root, "rev-parse", "--git-common-dir")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	p := strings.TrimSpace(string(out))
	if p == "" {
		return ""
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(root, p)
	}
	return p
}
func verbFlagUsage(fs *flag.FlagSet, _ string) {
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage of %s:\n", fs.Name())
		fs.PrintDefaults()
	}
}

// The fak-repo-root discovery ladder now lives in internal/fakroot — the ONE canonical
// copy (this file used to own it). These three wrappers stay so every existing caller
// and test in this package keeps compiling unchanged, and so the exec-backed rung-1
// probe stays where it was: the ladder body spawns nothing by design.

// isFakRepoRoot is the thin wrapper over fakroot.IsRepoRoot.
func isFakRepoRoot(dir string) bool {
	return fakroot.IsRepoRoot(dir)
}

// parseGoWorkUseDirs is the thin wrapper over fakroot.ParseGoWorkUseDirs.
func parseGoWorkUseDirs(workPath string) []string {
	return fakroot.ParseGoWorkUseDirs(workPath)
}

// discoverRepoRoot supplies the ladder with the one input it deliberately does not
// compute itself — the `git rev-parse --show-toplevel` subprocess — and returns the
// discovered public fak checkout, or "" when none is discoverable.
func discoverRepoRoot() string {
	var gitRoot string
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.Output(); err == nil {
		gitRoot = strings.TrimSpace(string(out))
	}
	cwd, _ := os.Getwd()
	return fakroot.Ladder{GitRoot: gitRoot, Cwd: cwd}.Discover()
}

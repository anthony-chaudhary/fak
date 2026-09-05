package selfupdatecmd

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

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

func isFakRepoRoot(dir string) bool {
	if strings.TrimSpace(dir) == "" {
		return false
	}
	st, err := os.Stat(filepath.Join(dir, "cmd", "fak", "main.go"))
	return err == nil && !st.IsDir()
}

func parseGoWorkUseDirs(workPath string) []string {
	b, err := os.ReadFile(workPath)
	if err != nil {
		return nil
	}
	workDir := filepath.Dir(workPath)
	var dirs []string
	inUseBlock := false

	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if idx := strings.Index(line, "//"); idx >= 0 {
			line = strings.TrimSpace(line[:idx])
		}
		if line == "" {
			continue
		}
		if inUseBlock {
			if strings.HasPrefix(line, ")") {
				inUseBlock = false
				continue
			}
			entry := strings.Trim(line, `"'`+" \t\r")
			dirs = append(dirs, filepath.Clean(filepath.Join(workDir, entry)))
		} else if strings.HasPrefix(line, "use (") || line == "use (" {
			inUseBlock = true
		} else if strings.HasPrefix(line, "use ") {
			rest := strings.TrimSpace(strings.TrimPrefix(line, "use "))
			entry := strings.Trim(rest, `"'`+" \t\r")
			dirs = append(dirs, filepath.Clean(filepath.Join(workDir, entry)))
		}
	}
	return dirs
}

func discoverRepoRoot() string {
	// 1. Check if current git repo has cmd/fak/main.go.
	var gitRoot string
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.Output(); err == nil {
		gitRoot = strings.TrimSpace(string(out))
		if isFakRepoRoot(gitRoot) {
			return gitRoot
		}
	}

	// 2. Check $FAK_ROOT.
	if envRoot := strings.TrimSpace(os.Getenv("FAK_ROOT")); envRoot != "" {
		if abs, err := filepath.Abs(envRoot); err == nil && isFakRepoRoot(abs) {
			return abs
		}
		if isFakRepoRoot(envRoot) {
			return envRoot
		}
	}

	// 3. Check go.work in CWD and git root (parse `use` entries to locate a directory with cmd/fak/main.go).
	cwd, _ := os.Getwd()
	checkWorkDirs := func(workPath string) string {
		for _, dir := range parseGoWorkUseDirs(workPath) {
			if isFakRepoRoot(dir) {
				return dir
			}
		}
		return ""
	}
	if cwd != "" {
		if found := checkWorkDirs(filepath.Join(cwd, "go.work")); found != "" {
			return found
		}
	}
	if gitRoot != "" && !strings.EqualFold(gitRoot, cwd) {
		if found := checkWorkDirs(filepath.Join(gitRoot, "go.work")); found != "" {
			return found
		}
	}

	// 4. Check sibling ../fak or ..\fak.
	siblingCandidates := []string{
		filepath.Join("..", "fak"),
	}
	if cwd != "" {
		siblingCandidates = append(siblingCandidates, filepath.Join(cwd, "..", "fak"))
	}
	if gitRoot != "" {
		siblingCandidates = append(siblingCandidates, filepath.Join(gitRoot, "..", "fak"))
	}
	for _, cand := range siblingCandidates {
		abs, err := filepath.Abs(cand)
		if err == nil && isFakRepoRoot(abs) {
			return abs
		}
		if isFakRepoRoot(cand) {
			return cand
		}
	}

	// 5. Fall back to the CWD git root.
	return gitRoot
}

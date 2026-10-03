package fakroot

import (
	"os"
	"path/filepath"
	"strings"
)

// IsRepoRoot reports whether dir is a PUBLIC fak repository checkout. The proof is
// structural and on-disk: `cmd/fak/main.go` must exist and be a regular file. It is
// deliberately NOT satisfied by the private companion checkout (fak-private has no
// cmd/fak tree), which is what keeps every widening downstream asymmetric.
func IsRepoRoot(dir string) bool {
	if strings.TrimSpace(dir) == "" {
		return false
	}
	st, err := os.Stat(filepath.Join(dir, "cmd", "fak", "main.go"))
	return err == nil && !st.IsDir()
}

// ParseGoWorkUseDirs returns the `use` entries declared by the go.work file at
// workPath, each resolved to an absolute directory relative to the go.work's own
// directory. A missing/unreadable file yields nil.
func ParseGoWorkUseDirs(workPath string) []string {
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

// Ladder carries the resolution INPUTS of Discover so the ladder body itself stays a
// pure filesystem walk: a caller that already knows (or already ran) the git
// top-level hands it in, and a caller that must not spawn a subprocess simply leaves
// GitRoot empty.
type Ladder struct {
	// GitRoot is the `git rev-parse --show-toplevel` output, or "" when the caller
	// did not probe it (no subprocess) or the probe failed.
	GitRoot string
	// Cwd is the process working directory, or "" when unknown.
	Cwd string
	// Getenv reads environment variables; nil means os.Getenv.
	Getenv func(string) string
}

// Discover walks the canonical ladder in rung order and returns the first directory
// that IsRepoRoot proves to be the public fak checkout, or "" when none is
// discoverable. It never returns a directory that fails the proof.
func (l Ladder) Discover() string {
	getenv := l.Getenv
	if getenv == nil {
		getenv = os.Getenv
	}
	gitRoot := l.GitRoot
	cwd := l.Cwd

	// 1. Check if current git repo has cmd/fak/main.go.
	if IsRepoRoot(gitRoot) {
		return gitRoot
	}

	// 2. Check $FAK_ROOT.
	if envRoot := strings.TrimSpace(getenv("FAK_ROOT")); envRoot != "" {
		if abs, err := filepath.Abs(envRoot); err == nil && IsRepoRoot(abs) {
			return abs
		}
		if IsRepoRoot(envRoot) {
			return envRoot
		}
	}

	// 3. Check go.work in CWD and git root (parse `use` entries to locate a directory with cmd/fak/main.go).
	checkWorkDirs := func(workPath string) string {
		for _, dir := range ParseGoWorkUseDirs(workPath) {
			if IsRepoRoot(dir) {
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

	// 4. Check child directory "fak" in cwd and git root (e.g. running from parent workspace like C:\work).
	childDirs := []string{"fak"}
	if cwd != "" {
		childDirs = append(childDirs, filepath.Join(cwd, "fak"))
	}
	if gitRoot != "" && !strings.EqualFold(gitRoot, cwd) {
		childDirs = append(childDirs, filepath.Join(gitRoot, "fak"))
	}
	for _, cand := range childDirs {
		abs, err := filepath.Abs(cand)
		if err == nil && IsRepoRoot(abs) {
			return abs
		}
		if IsRepoRoot(cand) {
			return cand
		}
	}

	// 5. Check sibling ../fak or ..\fak.
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
		if err == nil && IsRepoRoot(abs) {
			return abs
		}
		if IsRepoRoot(cand) {
			return cand
		}
	}

	// 6. Only fall back to the git root if it is genuinely a fak repo root; never return a non-fak directory.
	if gitRoot != "" && IsRepoRoot(gitRoot) {
		return gitRoot
	}
	return ""
}

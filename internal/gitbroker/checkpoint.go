package gitbroker

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ShadowCheckpointer captures and restores shadow checkpoints of git working trees,
// preserving both tracked changes and untracked files into 3-parent stash commits
// without modifying trunk history (HEAD) or refs/stash.
type ShadowCheckpointer interface {
	CreateShadowCheckpoint(repoDir string) (string, error)
	RestoreShadowCheckpoint(repoDir, ref string) error
}

// GitShadowCheckpointer is a concrete, thread-safe implementation of ShadowCheckpointer.
type GitShadowCheckpointer struct {
	mu          sync.RWMutex
	Checkpoints map[string]string
}

// NewGitShadowCheckpointer returns an initialized GitShadowCheckpointer.
func NewGitShadowCheckpointer() *GitShadowCheckpointer {
	return &GitShadowCheckpointer{
		Checkpoints: make(map[string]string),
	}
}

// DefaultShadowCheckpointer is the package-level default shadow checkpointer.
var DefaultShadowCheckpointer = NewGitShadowCheckpointer()

// CreateShadowCheckpoint captures the current working tree, index, and untracked files
// in repoDir into a 3-parent shadow stash commit without touching trunk history (HEAD)
// or refs/stash.
func CreateShadowCheckpoint(repoDir string) (string, error) {
	return DefaultShadowCheckpointer.CreateShadowCheckpoint(repoDir)
}

// RestoreShadowCheckpoint restores tracked changes and untracked files from ref into
// repoDir without modifying trunk history (HEAD).
func RestoreShadowCheckpoint(repoDir, ref string) error {
	return DefaultShadowCheckpointer.RestoreShadowCheckpoint(repoDir, ref)
}

// GetCheckpoint retrieves the recorded checkpoint for a given key (repoDir or ref).
func (g *GitShadowCheckpointer) GetCheckpoint(key string) (string, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.Checkpoints == nil {
		return "", false
	}
	v, ok := g.Checkpoints[key]
	return v, ok
}

// CreateShadowCheckpoint captures working tree modifications, staged index changes,
// and untracked files into a stash commit with up to 3 parents:
// - parent 1 (ref^1): HEAD commit
// - parent 2 (ref^2): index commit (staged changes)
// - parent 3 (ref^3): untracked commit (if untracked files exist)
// Trunk history (HEAD) and refs/stash are left completely untouched.
func (g *GitShadowCheckpointer) CreateShadowCheckpoint(repoDir string) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if repoDir == "" {
		repoDir = "."
	}
	absDir, err := filepath.Abs(repoDir)
	if err != nil {
		return "", fmt.Errorf("resolve repo dir %q: %w", repoDir, err)
	}
	repoDir = absDir

	// Verify that HEAD commit exists.
	headSHA, err := runGit(repoDir, nil, nil, "rev-parse", "--verify", "HEAD")
	if err != nil {
		return "", fmt.Errorf("resolve HEAD in %s: %w", repoDir, err)
	}

	branch, _ := runGit(repoDir, nil, nil, "rev-parse", "--abbrev-ref", "HEAD")
	if branch == "" || branch == "HEAD" {
		branch = "(no branch)"
	}
	headShort := headSHA
	if len(headShort) > 7 {
		headShort = headShort[:7]
	}

	// 1. Index commit (tree of the current staged index, parent is HEAD).
	treeIndex, err := runGit(repoDir, nil, nil, "write-tree")
	if err != nil {
		return "", fmt.Errorf("write-tree for index: %w", err)
	}
	commitIndex, err := runGit(repoDir, nil, nil, "commit-tree", "-m", fmt.Sprintf("index on %s: %s", branch, headShort), "-p", headSHA, treeIndex)
	if err != nil {
		return "", fmt.Errorf("commit-tree for index: %w", err)
	}

	// 2. Working tree commit (tree of staged + unstaged modifications to tracked files).
	indexPath, err := runGit(repoDir, nil, nil, "rev-parse", "--git-path", "index")
	if err != nil {
		return "", fmt.Errorf("resolve git index path: %w", err)
	}
	if !filepath.IsAbs(indexPath) {
		indexPath = filepath.Join(repoDir, indexPath)
	}

	tempIdxWork, err := os.CreateTemp("", "fak_shadow_idx_work_*")
	if err != nil {
		return "", fmt.Errorf("create temp index for worktree: %w", err)
	}
	tempIdxWorkPath := tempIdxWork.Name()
	_ = tempIdxWork.Close()
	defer func() { _ = os.Remove(tempIdxWorkPath) }()

	if data, err := os.ReadFile(indexPath); err == nil && len(data) > 0 {
		if err := os.WriteFile(tempIdxWorkPath, data, 0o600); err != nil {
			return "", fmt.Errorf("copy index to temp file: %w", err)
		}
	} else {
		// Ensure non-existent index file rather than empty 0-byte file.
		_ = os.Remove(tempIdxWorkPath)
	}

	envWork := []string{"GIT_INDEX_FILE=" + tempIdxWorkPath}
	if _, err := runGit(repoDir, nil, envWork, "add", "-u"); err != nil {
		return "", fmt.Errorf("update temp index with tracked changes: %w", err)
	}
	treeWork, err := runGit(repoDir, nil, envWork, "write-tree")
	if err != nil {
		return "", fmt.Errorf("write-tree for worktree: %w", err)
	}

	// 3. Untracked files commit (tree of untracked files, parentless root commit).
	rawUntracked, err := runGitBytes(repoDir, nil, nil, "ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return "", fmt.Errorf("list untracked files: %w", err)
	}

	var commitUntracked string
	hasUntracked := false
	for _, entry := range bytes.Split(rawUntracked, []byte{0}) {
		if len(entry) > 0 {
			hasUntracked = true
			break
		}
	}

	if hasUntracked {
		tempIdxUntracked, err := os.CreateTemp("", "fak_shadow_idx_untracked_*")
		if err != nil {
			return "", fmt.Errorf("create temp index for untracked files: %w", err)
		}
		tempIdxUntrackedPath := tempIdxUntracked.Name()
		_ = tempIdxUntracked.Close()
		_ = os.Remove(tempIdxUntrackedPath) // Git requires index file to not exist if empty.
		defer func() { _ = os.Remove(tempIdxUntrackedPath) }()

		envUntracked := []string{"GIT_INDEX_FILE=" + tempIdxUntrackedPath}
		if _, err := runGit(repoDir, rawUntracked, envUntracked, "add", "--pathspec-from-file=-", "--pathspec-file-nul"); err != nil {
			return "", fmt.Errorf("add untracked files to temp index: %w", err)
		}
		treeUntracked, err := runGit(repoDir, nil, envUntracked, "write-tree")
		if err != nil {
			return "", fmt.Errorf("write-tree for untracked files: %w", err)
		}
		commitUntracked, err = runGit(repoDir, nil, nil, "commit-tree", "-m", fmt.Sprintf("untracked files on %s: %s", branch, headShort), treeUntracked)
		if err != nil {
			return "", fmt.Errorf("commit-tree for untracked files: %w", err)
		}
	}

	// 4. Stash commit.
	// Parents: HEAD (headSHA), Index commit (commitIndex), and optionally Untracked commit (commitUntracked).
	commitArgs := []string{"commit-tree", "-m", fmt.Sprintf("WIP on %s: %s", branch, headShort), "-p", headSHA, "-p", commitIndex}
	if commitUntracked != "" {
		commitArgs = append(commitArgs, "-p", commitUntracked)
	}
	commitArgs = append(commitArgs, treeWork)

	stashCommit, err := runGit(repoDir, nil, nil, commitArgs...)
	if err != nil {
		return "", fmt.Errorf("commit-tree for stash commit: %w", err)
	}

	if g.Checkpoints == nil {
		g.Checkpoints = make(map[string]string)
	}
	g.Checkpoints[repoDir] = stashCommit
	g.Checkpoints[stashCommit] = repoDir

	return stashCommit, nil
}

// RestoreShadowCheckpoint restores the working tree, index, and untracked files
// from the shadow checkpoint ref without altering trunk git commit history (HEAD).
func (g *GitShadowCheckpointer) RestoreShadowCheckpoint(repoDir, ref string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if repoDir == "" {
		repoDir = "."
	}
	absDir, err := filepath.Abs(repoDir)
	if err != nil {
		return fmt.Errorf("resolve repo dir %q: %w", repoDir, err)
	}
	repoDir = absDir

	// If ref matches a mapped key (e.g. repoDir), resolve mapped ref.
	if g.Checkpoints != nil {
		if mapped, ok := g.Checkpoints[ref]; ok {
			if _, err := runGit(repoDir, nil, nil, "rev-parse", "--verify", ref+"^{commit}"); err != nil {
				ref = mapped
			}
		}
	}

	commitSHA, err := runGit(repoDir, nil, nil, "rev-parse", "--verify", ref+"^{commit}")
	if err != nil {
		return fmt.Errorf("resolve checkpoint ref %q: %w", ref, err)
	}

	headSHA, err := runGit(repoDir, nil, nil, "rev-parse", "--verify", commitSHA+"^1")
	if err != nil {
		return fmt.Errorf("checkpoint commit %s has no parent 1: %w", commitSHA, err)
	}

	commitIndex, err := runGit(repoDir, nil, nil, "rev-parse", "--verify", commitSHA+"^2")
	if err != nil {
		return fmt.Errorf("checkpoint commit %s has no parent 2: %w", commitSHA, err)
	}

	commitUntracked, errUntracked := runGit(repoDir, nil, nil, "rev-parse", "--verify", commitSHA+"^3")
	hasUntracked := (errUntracked == nil && commitUntracked != "")

	// 1. Restore tracked files in the working tree from the stash commit.
	if _, err := runGit(repoDir, nil, nil, "read-tree", commitSHA); err != nil {
		return fmt.Errorf("read-tree checkpoint commit %s: %w", commitSHA, err)
	}
	if _, err := runGit(repoDir, nil, nil, "checkout-index", "-a", "-f"); err != nil {
		return fmt.Errorf("checkout-index checkpoint commit %s: %w", commitSHA, err)
	}

	// 2. Remove any tracked files that were deleted in the checkpoint relative to HEAD.
	diffRaw, err := runGitBytes(repoDir, nil, nil, "diff-tree", "-r", "--name-status", "-z", headSHA, commitSHA)
	if err != nil {
		return fmt.Errorf("diff-tree deleted files: %w", err)
	}
	diffTokens := bytes.Split(diffRaw, []byte{0})
	for i := 0; i < len(diffTokens); i++ {
		tok := string(diffTokens[i])
		if len(tok) == 0 {
			continue
		}
		status := tok[0]
		switch status {
		case 'R', 'C':
			i += 2
		default:
			if i+1 < len(diffTokens) {
				path := string(diffTokens[i+1])
				if status == 'D' {
					removeFileAndEmptyParents(repoDir, path)
				}
				i++
			}
		}
	}

	// 3. Restore the index to the staged changes state (parent 2).
	if _, err := runGit(repoDir, nil, nil, "read-tree", commitIndex); err != nil {
		return fmt.Errorf("read-tree index commit %s: %w", commitIndex, err)
	}

	// 4. Restore untracked files (parent 3).
	snapUntrackedMap := make(map[string]bool)
	if hasUntracked {
		lsTreeRaw, err := runGitBytes(repoDir, nil, nil, "ls-tree", "-r", "--name-only", "-z", commitUntracked)
		if err != nil {
			return fmt.Errorf("ls-tree untracked commit: %w", err)
		}
		for _, entry := range bytes.Split(lsTreeRaw, []byte{0}) {
			if len(entry) > 0 {
				snapUntrackedMap[string(entry)] = true
			}
		}
	}

	// Remove untracked files currently on disk that were not present in the checkpoint.
	curUntrackedRaw, err := runGitBytes(repoDir, nil, nil, "ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return fmt.Errorf("list current untracked files: %w", err)
	}
	for _, entry := range bytes.Split(curUntrackedRaw, []byte{0}) {
		if len(entry) > 0 {
			path := string(entry)
			if !snapUntrackedMap[path] {
				removeFileAndEmptyParents(repoDir, path)
			}
		}
	}

	// Extract untracked files from untracked commit into working tree.
	if hasUntracked {
		tempIdxUntracked, err := os.CreateTemp("", "fak_shadow_restore_untracked_*")
		if err != nil {
			return fmt.Errorf("create temp index for untracked restore: %w", err)
		}
		tempIdxUntrackedPath := tempIdxUntracked.Name()
		_ = tempIdxUntracked.Close()
		_ = os.Remove(tempIdxUntrackedPath)
		defer func() { _ = os.Remove(tempIdxUntrackedPath) }()

		envUntracked := []string{"GIT_INDEX_FILE=" + tempIdxUntrackedPath}
		if _, err := runGit(repoDir, nil, envUntracked, "read-tree", commitUntracked); err != nil {
			return fmt.Errorf("read-tree untracked commit %s: %w", commitUntracked, err)
		}
		if _, err := runGit(repoDir, nil, envUntracked, "checkout-index", "-a", "-f"); err != nil {
			return fmt.Errorf("checkout-index untracked files: %w", err)
		}
	}

	return nil
}

func removeFileAndEmptyParents(repoDir, relPath string) {
	fullPath := filepath.Join(repoDir, relPath)
	_ = os.Remove(fullPath)
	dir := filepath.Dir(fullPath)
	cleanRepo := filepath.Clean(repoDir)
	for dir != cleanRepo && dir != "." && dir != "/" && dir != "" {
		if err := os.Remove(dir); err != nil {
			break
		}
		dir = filepath.Dir(dir)
	}
}

func runGitBytes(dir string, stdin []byte, env []string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)

	cmdEnv := os.Environ()
	hasAuthor := false
	for _, e := range cmdEnv {
		if strings.HasPrefix(e, "GIT_AUTHOR_NAME=") {
			hasAuthor = true
			break
		}
	}
	if !hasAuthor {
		cmdEnv = append(cmdEnv,
			"GIT_AUTHOR_NAME=fak-checkpoint",
			"GIT_AUTHOR_EMAIL=fak@local",
			"GIT_COMMITTER_NAME=fak-checkpoint",
			"GIT_COMMITTER_EMAIL=fak@local",
		)
	}
	if len(env) > 0 {
		cmdEnv = append(cmdEnv, env...)
	}
	cmd.Env = cmdEnv

	if len(stdin) > 0 {
		cmd.Stdin = bytes.NewReader(stdin)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w (%s)", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func runGit(dir string, stdin []byte, env []string, args ...string) (string, error) {
	out, err := runGitBytes(dir, stdin, env, args...)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

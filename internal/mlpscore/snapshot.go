package mlpscore

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// GitSnapshot is an immutable view of one repository HEAD. It intentionally
// ignores staged, modified, and untracked files.
type GitSnapshot struct {
	root   string
	commit string
	files  map[string]struct{}
}

// LoadGitSnapshot captures the path set and short commit at HEAD.
func LoadGitSnapshot(root string) (*GitSnapshot, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace: %w", err)
	}
	raw, err := runGit(abs, "ls-tree", "-r", "--name-only", "-z", "HEAD")
	if err != nil {
		return nil, fmt.Errorf("read committed tree: %w", err)
	}
	commitRaw, err := runGit(abs, "rev-parse", "--short", "HEAD")
	if err != nil {
		return nil, fmt.Errorf("read HEAD: %w", err)
	}
	s := &GitSnapshot{
		root:   abs,
		commit: strings.TrimSpace(string(commitRaw)),
		files:  map[string]struct{}{},
	}
	for _, item := range bytes.Split(raw, []byte{0}) {
		if len(item) > 0 {
			s.files[string(item)] = struct{}{}
		}
	}
	return s, nil
}

// Commit returns the short commit the snapshot grades.
func (s *GitSnapshot) Commit() string { return s.commit }

func (s *GitSnapshot) Exists(rel string) bool {
	clean, ok := cleanRepoPath(rel)
	if !ok {
		return false
	}
	_, ok = s.files[clean]
	return ok
}

func (s *GitSnapshot) ReadFile(rel string) ([]byte, error) {
	clean, ok := cleanRepoPath(rel)
	if !ok {
		return nil, fmt.Errorf("invalid repository path %q", rel)
	}
	if !s.Exists(clean) {
		return nil, errors.New("path is not committed at HEAD")
	}
	raw, err := runGit(s.root, "show", "HEAD:"+clean)
	if err != nil {
		return nil, fmt.Errorf("read %s at HEAD: %w", clean, err)
	}
	return raw, nil
}

func runGit(root string, argv ...string) ([]byte, error) {
	cmd := exec.Command("git", argv...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %v: %s", strings.Join(argv, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

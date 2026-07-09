package shadowgit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Runner executes a git invocation and returns its stdout. It is injectable so the
// snapshot/diff logic is unit-testable without a real git, while production uses
// [ExecRunner] over the git binary. args are the git arguments AFTER the leading
// --git-dir/--work-tree pair, which ShadowGit prepends.
type Runner interface {
	Run(args ...string) (stdout string, err error)
}

// ShadowGit is a private git ledger over a worktree, reached through a separate git
// dir so the real repo's .git is never touched. Construct with [Open]; drive it with
// [ShadowGit.Baseline] once, then [ShadowGit.Snapshot] per agent step.
type ShadowGit struct {
	gitDir         string
	workTree       string
	git            Runner
	includeIgnored bool
	lastCommit     string // shadow sha of the most recent snapshot ("" until Baseline)
}

// Options tunes Open. The zero value is the safe default: honor .gitignore, real git.
type Options struct {
	// IncludeIgnored stages files the worktree's .gitignore would exclude, so writes
	// to ignored paths are still attributed. Off by default (avoids build-output noise).
	IncludeIgnored bool
	// Runner overrides the git executor (tests inject a fake). nil => ExecRunner.
	Runner Runner
}

// Change is one file's delta between two snapshots.
type Change struct {
	Status  string `json:"status"`             // A (added) | M (modified) | D (deleted) | R (renamed) | C (copied) | T (type)
	Path    string `json:"path"`               // the current path (destination for a rename)
	OldPath string `json:"old_path,omitempty"` // source path for a rename/copy
}

// Snapshot is one step's committed worktree state plus the changes it introduced
// relative to the previous snapshot. It is the record appended to state_changelog.jsonl.
type Snapshot struct {
	Step       int      `json:"step"`
	Label      string   `json:"label,omitempty"`
	Commit     string   `json:"commit"`                 // shadow-repo sha of this snapshot
	Parent     string   `json:"parent,omitempty"`       // shadow sha of the previous snapshot
	Baseline   bool     `json:"baseline,omitempty"`     // true for the reference snapshot (no attributed changes)
	Changes    []Change `json:"changes"`                // files changed since Parent (empty for a baseline)
	TSUnixNano int64    `json:"ts_unix_nano,omitempty"` // optional wall-clock anchor (0 => unset)
}

// ExecRunner runs the real git binary. It is the default Runner.
type ExecRunner struct{}

// Run implements Runner over os/exec.
func (ExecRunner) Run(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd) // no console flash when a windowless Windows parent runs git
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

// Open prepares a shadow git ledger at gitDir over the worktree at workTree,
// initializing the shadow repo (and its exclude rules) if it does not exist yet. It
// performs no snapshot — call Baseline next.
func Open(gitDir, workTree string, opts Options) (*ShadowGit, error) {
	absGit, err := filepath.Abs(gitDir)
	if err != nil {
		return nil, err
	}
	absWT, err := filepath.Abs(workTree)
	if err != nil {
		return nil, err
	}
	if fi, err := os.Stat(absWT); err != nil || !fi.IsDir() {
		return nil, fmt.Errorf("shadowgit: work-tree %q is not a directory", workTree)
	}
	runner := opts.Runner
	if runner == nil {
		runner = ExecRunner{}
	}
	s := &ShadowGit{gitDir: absGit, workTree: absWT, git: runner, includeIgnored: opts.IncludeIgnored}
	if err := s.ensureInit(); err != nil {
		return nil, err
	}
	return s, nil
}

// git runs a git subcommand with the shadow git-dir + work-tree prepended.
func (s *ShadowGit) git0(args ...string) (string, error) {
	full := append([]string{"--git-dir", s.gitDir, "--work-tree", s.workTree}, args...)
	return s.git.Run(full...)
}

// ensureInit creates the shadow repo if absent and writes its exclude rules so a
// snapshot never captures the real .git or the shadow dir itself.
func (s *ShadowGit) ensureInit() error {
	if _, err := os.Stat(s.gitDir); err == nil {
		return s.writeExcludes() // already initialized; keep excludes current
	}
	if err := os.MkdirAll(s.gitDir, 0o755); err != nil {
		return err
	}
	if _, err := s.git0("init", "-q"); err != nil {
		return err
	}
	// A private ledger has its own identity, so a snapshot never depends on the
	// operator's global git config (and never prompts).
	if _, err := s.git0("config", "user.email", "shadowgit@fak.local"); err != nil {
		return err
	}
	if _, err := s.git0("config", "user.name", "fak shadowgit"); err != nil {
		return err
	}
	return s.writeExcludes()
}

// writeExcludes stamps info/exclude so the real .git/ and (if nested) the shadow dir
// are never staged — the non-invasiveness guarantee, enforced in git's own ignore layer.
func (s *ShadowGit) writeExcludes() error {
	lines := []string{
		"# fak shadowgit: never snapshot the real repo metadata or this ledger",
		"/.git/",
		".git/",
	}
	if rel, err := filepath.Rel(s.workTree, s.gitDir); err == nil && !strings.HasPrefix(rel, "..") && rel != "." {
		lines = append(lines, "/"+filepath.ToSlash(rel)+"/")
	}
	info := filepath.Join(s.gitDir, "info")
	if err := os.MkdirAll(info, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(info, "exclude"), []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}

// stage adds the whole worktree to the shadow index (honoring .gitignore unless
// IncludeIgnored). It is the "index is the diff cache" step.
func (s *ShadowGit) stage() error {
	args := []string{"add", "-A"}
	if s.includeIgnored {
		args = append(args, "--force")
	}
	_, err := s.git0(args...)
	return err
}

// commit records the staged worktree as a snapshot and returns its sha. --allow-empty
// keeps step numbering contiguous even for a step that changed nothing.
func (s *ShadowGit) commit(message string) (string, error) {
	if _, err := s.git0("commit", "-q", "--allow-empty", "-m", message); err != nil {
		return "", err
	}
	sha, err := s.git0("rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(sha), nil
}

// Baseline records the reference snapshot (step 0). It carries no attributed changes —
// it is the point subsequent snapshots diff against. Calling it more than once re-bases.
func (s *ShadowGit) Baseline() (Snapshot, error) {
	if err := s.stage(); err != nil {
		return Snapshot{}, err
	}
	sha, err := s.commit("shadowgit baseline")
	if err != nil {
		return Snapshot{}, err
	}
	s.lastCommit = sha
	return Snapshot{Step: 0, Commit: sha, Baseline: true, Changes: []Change{}}, nil
}

// Snapshot stages the worktree, commits it, and returns the files changed since the
// previous snapshot — attributed to this step. Baseline must have been called first.
func (s *ShadowGit) Snapshot(step int, label string) (Snapshot, error) {
	if s.lastCommit == "" {
		return Snapshot{}, fmt.Errorf("shadowgit: call Baseline before Snapshot")
	}
	if err := s.stage(); err != nil {
		return Snapshot{}, err
	}
	parent := s.lastCommit
	msg := label
	if msg == "" {
		msg = fmt.Sprintf("shadowgit step %d", step)
	}
	sha, err := s.commit(msg)
	if err != nil {
		return Snapshot{}, err
	}
	changes, err := s.diff(parent, sha)
	if err != nil {
		return Snapshot{}, err
	}
	s.lastCommit = sha
	return Snapshot{Step: step, Label: label, Commit: sha, Parent: parent, Changes: changes}, nil
}

// CheckForWrites reports whether the worktree has any change not yet in the last
// snapshot — the cheap "did anything happen since the last step?" probe (a porcelain
// status against the shadow index). It does not stage or commit.
func (s *ShadowGit) CheckForWrites() (bool, error) {
	out, err := s.git0("status", "--porcelain", "--untracked-files=all")
	if err != nil {
		return false, err
	}
	return strings.TrimSpace(out) != "", nil
}

// diff parses `git diff --name-status a b` from the shadow repo into []Change.
func (s *ShadowGit) diff(a, b string) ([]Change, error) {
	out, err := s.git0("diff", "--name-status", "-z", a, b)
	if err != nil {
		return nil, err
	}
	return parseNameStatusZ(out), nil
}

// parseNameStatusZ parses git's NUL-delimited --name-status -z stream. For a plain
// status (A/M/D/T) the record is <status>\0<path>\0; for a rename/copy (R/C) it is
// <status>\0<old>\0<new>\0.
func parseNameStatusZ(z string) []Change {
	fields := strings.Split(z, "\x00")
	var out []Change
	for i := 0; i < len(fields); {
		status := fields[i]
		if status == "" {
			i++
			continue
		}
		i++
		if i >= len(fields) {
			break
		}
		code := status[0]
		if code == 'R' || code == 'C' {
			if i+1 >= len(fields) {
				break
			}
			out = append(out, Change{Status: string(code), OldPath: fields[i], Path: fields[i+1]})
			i += 2
			continue
		}
		out = append(out, Change{Status: string(code), Path: fields[i]})
		i++
	}
	return out
}

// WriteChangelogLine appends one snapshot to a state_changelog JSONL writer.
func WriteChangelogLine(w io.Writer, snap Snapshot) error {
	b, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}

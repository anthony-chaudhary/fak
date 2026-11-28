package shadowgit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
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
// keeps step numbering contiguous even for a step that changed nothing. Step metadata
// trailers are attached in the commit body so Summarize can reconstruct status from the
// shadow repo alone across process restarts.
func (s *ShadowGit) commitStep(subject string, step int, baseline bool, label string) (string, error) {
	cleanSubj := strings.TrimSpace(strings.ReplaceAll(strings.SplitN(subject, "\n", 2)[0], "\r", ""))
	if cleanSubj == "" {
		cleanSubj = fmt.Sprintf("shadowgit step %d", step)
	}
	trailer := fmt.Sprintf("Shadowgit-Step: %d\nShadowgit-Baseline: %t", step, baseline)
	if label != "" {
		trailer += "\nShadowgit-Label: " + strconv.Quote(label)
	}
	if _, err := s.git0("commit", "-q", "--allow-empty", "-m", cleanSubj, "-m", trailer); err != nil {
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
	sha, err := s.commitStep("shadowgit baseline", 0, true, "")
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
	sha, err := s.commitStep(msg, step, false, label)
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

// StatusReport is the witnessed status snapshot of a shadowgit ledger and its
// worktree, suitable for crash/resume inspection and peer health checks without
// staging files or mutating the shadow index.
type StatusReport struct {
	GitDir         string   `json:"git_dir"`
	WorkTree       string   `json:"work_tree"`
	Initialized    bool     `json:"initialized"`               // true once a baseline commit exists in the shadow ledger
	Dirty          bool     `json:"dirty"`                     // true when workTree has un-snapshotted writes
	PendingWrites  int      `json:"pending_writes"`            // count of un-snapshotted paths in workTree
	PendingPaths   []string `json:"pending_paths,omitempty"`   // sorted worktree-relative pending paths
	CommitCount    int      `json:"commit_count"`              // total commits in the shadow ledger
	SnapshotCount  int      `json:"snapshot_count"`            // non-baseline snapshot commits since the latest baseline
	LatestStep     int      `json:"latest_step"`               // step number of the latest commit (0 for baseline)
	LatestCommit   string   `json:"latest_commit,omitempty"`   // shadow sha of HEAD ("" until Baseline)
	LatestLabel    string   `json:"latest_label,omitempty"`    // label of the latest snapshot ("" if unlabeled or baseline)
	LatestBaseline bool     `json:"latest_baseline,omitempty"` // true when HEAD is a baseline commit
	LastChanges    []Change `json:"last_changes,omitempty"`    // attributed changes in the latest snapshot (parent -> HEAD)
	ChangesSummary string   `json:"changes_summary"`           // compact human-readable status and delta summary
}

// Status returns the witnessed StatusReport for the shadow repository and worktree.
// It delegates to [ShadowGit.Summarize].
func (s *ShadowGit) Status() (StatusReport, error) {
	return s.Summarize()
}

// Summarize reads the shadow repository's witnessed commit log, latest diff, and
// current worktree status, folding them into a StatusReport without staging or
// committing.
func (s *ShadowGit) Summarize() (StatusReport, error) {
	if s == nil || s.git == nil || s.gitDir == "" || s.workTree == "" {
		return StatusReport{}, fmt.Errorf("shadowgit: ledger is not initialized")
	}
	if fi, err := os.Stat(s.workTree); err != nil || !fi.IsDir() {
		return StatusReport{}, fmt.Errorf("shadowgit: work-tree %q is not a directory", s.workTree)
	}
	if fi, err := os.Stat(s.gitDir); err != nil || !fi.IsDir() {
		return StatusReport{}, fmt.Errorf("shadowgit: git-dir %q is not a directory", s.gitDir)
	}
	snapshots, err := s.readSnapshots()
	if err != nil {
		return StatusReport{}, err
	}
	pending, err := s.pendingPaths()
	if err != nil {
		return StatusReport{}, err
	}
	return FoldStatus(s.gitDir, s.workTree, snapshots, pending), nil
}

// FoldStatus folds an ordered slice of witnessed snapshots (oldest to newest) and
// any un-snapshotted worktree paths into a deterministic StatusReport. Pure: no I/O.
func FoldStatus(gitDir, workTree string, snapshots []Snapshot, pendingPaths []string) StatusReport {
	var cleanPending []string
	for _, p := range pendingPaths {
		if p != "" {
			cleanPending = append(cleanPending, p)
		}
	}
	sort.Strings(cleanPending)

	report := StatusReport{
		GitDir:        gitDir,
		WorkTree:      workTree,
		Dirty:         len(cleanPending) > 0,
		PendingWrites: len(cleanPending),
		PendingPaths:  cleanPending,
	}

	if len(snapshots) == 0 {
		report.ChangesSummary = "uninitialized (no baseline); " + pendingSummary(report.PendingWrites)
		return report
	}

	report.Initialized = true
	report.CommitCount = len(snapshots)
	for i := len(snapshots) - 1; i >= 0; i-- {
		if snapshots[i].Baseline {
			break
		}
		report.SnapshotCount++
	}

	latest := snapshots[len(snapshots)-1]
	report.LatestStep = latest.Step
	report.LatestCommit = latest.Commit
	report.LatestLabel = latest.Label
	report.LatestBaseline = latest.Baseline
	if !latest.Baseline && len(latest.Changes) > 0 {
		report.LastChanges = append([]Change(nil), latest.Changes...)
	}

	headDesc := fmt.Sprintf("step %d", latest.Step)
	if latest.Baseline {
		headDesc = fmt.Sprintf("baseline (step %d)", latest.Step)
	} else if latest.Label != "" {
		headDesc = fmt.Sprintf("step %d (%s)", latest.Step, latest.Label)
	}
	parts := []string{headDesc}
	if !latest.Baseline {
		parts = append(parts, summarizeChanges(report.LastChanges))
	}
	parts = append(parts, snapshotCountSummary(report.SnapshotCount), pendingSummary(report.PendingWrites))
	report.ChangesSummary = strings.Join(parts, "; ")
	return report
}

func summarizeChanges(changes []Change) string {
	if len(changes) == 0 {
		return "0 changes"
	}
	counts := make(map[string]int, len(changes))
	for _, c := range changes {
		st := c.Status
		if st == "" {
			st = "?"
		}
		counts[st]++
	}
	order := []string{"A", "M", "D", "R", "C", "T"}
	seen := make(map[string]bool, len(order))
	var breakdown []string
	for _, k := range order {
		if n := counts[k]; n > 0 {
			breakdown = append(breakdown, fmt.Sprintf("%d %s", n, k))
			seen[k] = true
		}
	}
	var extras []string
	for k := range counts {
		if !seen[k] {
			extras = append(extras, k)
		}
	}
	sort.Strings(extras)
	for _, k := range extras {
		breakdown = append(breakdown, fmt.Sprintf("%d %s", counts[k], k))
	}
	noun := "changes"
	if len(changes) == 1 {
		noun = "change"
	}
	return fmt.Sprintf("%d %s (%s)", len(changes), noun, strings.Join(breakdown, ", "))
}

func snapshotCountSummary(n int) string {
	if n == 1 {
		return "1 snapshot"
	}
	return fmt.Sprintf("%d snapshots", n)
}

func pendingSummary(n int) string {
	if n == 0 {
		return "clean"
	}
	if n == 1 {
		return "1 pending write"
	}
	return fmt.Sprintf("%d pending writes", n)
}

func (s *ShadowGit) readSnapshots() ([]Snapshot, error) {
	_, headErr := s.git0("rev-parse", "--verify", "HEAD")
	if headErr != nil {
		refsOut, refErr := s.git0("for-each-ref", "--format=%(objectname)", "refs/heads")
		if refErr != nil {
			return nil, fmt.Errorf("shadowgit: inspect shadow repo: %w", refErr)
		}
		if strings.TrimSpace(refsOut) == "" {
			return nil, nil // valid initialized repo, Baseline not yet called
		}
		return nil, fmt.Errorf("shadowgit: resolve HEAD: %w", headErr)
	}
	logOut, err := s.git0("log", "--reverse", "--format=%H%x1f%P%x1f%s%x1f%b%x1e", "HEAD")
	if err != nil {
		return nil, err
	}
	snapshots := parseShadowLog(logOut)
	if len(snapshots) > 0 {
		last := &snapshots[len(snapshots)-1]
		if !last.Baseline && last.Parent != "" {
			changes, err := s.diff(last.Parent, last.Commit)
			if err != nil {
				return nil, err
			}
			last.Changes = changes
		}
	}
	return snapshots, nil
}

func parseShadowLog(out string) []Snapshot {
	records := strings.Split(out, "\x1e")
	var snapshots []Snapshot
	for _, rec := range records {
		rec = strings.Trim(rec, "\r\n")
		if rec == "" {
			continue
		}
		parts := strings.SplitN(rec, "\x1f", 4)
		if len(parts) < 3 {
			continue
		}
		sha := strings.TrimSpace(parts[0])
		parents := strings.Fields(parts[1])
		parent := ""
		if len(parents) > 0 {
			parent = parents[0]
		}
		subject := strings.TrimSpace(parts[2])
		body := ""
		if len(parts) >= 4 {
			body = parts[3]
		}

		var (
			step        int
			hasStep     bool
			baseline    bool
			hasBaseline bool
			label       string
			hasLabel    bool
		)
		for _, line := range strings.Split(body, "\n") {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "Shadowgit-Step:"):
				if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "Shadowgit-Step:"))); err == nil {
					step = n
					hasStep = true
				}
			case strings.HasPrefix(line, "Shadowgit-Baseline:"):
				if b, err := strconv.ParseBool(strings.TrimSpace(strings.TrimPrefix(line, "Shadowgit-Baseline:"))); err == nil {
					baseline = b
					hasBaseline = true
				}
			case strings.HasPrefix(line, "Shadowgit-Label:"):
				raw := strings.TrimSpace(strings.TrimPrefix(line, "Shadowgit-Label:"))
				if u, err := strconv.Unquote(raw); err == nil {
					label = u
				} else {
					label = raw
				}
				hasLabel = true
			}
		}

		if !hasBaseline && subject == "shadowgit baseline" {
			baseline = true
		}
		if !hasStep {
			if baseline {
				step = 0
			} else if strings.HasPrefix(subject, "shadowgit step ") {
				if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(subject, "shadowgit step "))); err == nil {
					step = n
					hasStep = true
				}
			}
			if !hasStep && !baseline {
				step = len(snapshots)
			}
		}
		if !hasLabel && !baseline && subject != fmt.Sprintf("shadowgit step %d", step) {
			label = subject
		}

		snapshots = append(snapshots, Snapshot{
			Step:     step,
			Label:    label,
			Commit:   sha,
			Parent:   parent,
			Baseline: baseline,
		})
	}
	return snapshots
}

func (s *ShadowGit) pendingPaths() ([]string, error) {
	out, err := s.git0("status", "--porcelain", "-z", "--untracked-files=all")
	if err != nil {
		return nil, err
	}
	return parseStatusPorcelainZ(out), nil
}

func parseStatusPorcelainZ(z string) []string {
	fields := strings.Split(z, "\x00")
	var paths []string
	for i := 0; i < len(fields); i++ {
		entry := fields[i]
		if len(entry) < 4 {
			continue
		}
		x, y := entry[0], entry[1]
		paths = append(paths, entry[3:])
		if x == 'R' || x == 'C' || y == 'R' || y == 'C' {
			i++
		}
	}
	sort.Strings(paths)
	return paths
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

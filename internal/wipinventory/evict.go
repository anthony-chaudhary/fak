package wipinventory

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// QuarantineNamespace is the ref namespace where orphan residue is archived.
const QuarantineNamespace = "refs/fak/quarantine/"

// QuarantineRef records the durable snapshot minted when evicting orphan files.
type QuarantineRef struct {
	Ref       string   `json:"ref"`
	SHA       string   `json:"sha"`
	TreeSHA   string   `json:"tree_sha,omitempty"`
	Files     []string `json:"files"`
	Count     int      `json:"count"`
	ByteTotal int64    `json:"byte_total"`
	SessionID string   `json:"session_id,omitempty"`
	Timestamp int64    `json:"timestamp,omitempty"`
}

// EvictOptions configures orphan eviction.
type EvictOptions struct {
	SessionID string
	Targets   []string
	Now       time.Time
	DryRun    bool
	Reason    string
}

// EvictOrphans archives un-checkpointed orphan working-tree residue into a synthetic
// commit under refs/fak/quarantine/<timestamp> (or refs/fak/quarantine/<session>/<timestamp>)
// before removing them from the working tree. It returns the created QuarantineRef
// with commit SHA, ref name, files archived, count, and byte total.
func EvictOrphans(ctx context.Context, repoRoot string, runner Runner, opts ...EvictOptions) (*QuarantineRef, error) {
	if repoRoot == "" {
		repoRoot = "."
	}
	abs, err := filepath.Abs(repoRoot)
	if err == nil {
		repoRoot = abs
	}
	if runner == nil {
		runner = GitRunner{}
	}
	var opt EvictOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	now := opt.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}

	targets := opt.Targets
	if len(targets) == 0 {
		discovered, err := discoverOrphans(repoRoot, runner)
		if err != nil {
			return nil, fmt.Errorf("discover orphans: %w", err)
		}
		targets = discovered
	}

	var filesToArchive []string
	var byteTotal int64
	for _, p := range targets {
		norm := filepath.ToSlash(filepath.Clean(filepath.FromSlash(p)))
		if norm == "" || norm == "." || norm == ".." || strings.HasPrefix(norm, "../") || filepath.IsAbs(norm) {
			continue
		}
		full := filepath.Join(repoRoot, filepath.FromSlash(norm))
		st, err := os.Lstat(full)
		if err != nil {
			continue
		}
		if st.IsDir() {
			continue
		}
		filesToArchive = append(filesToArchive, norm)
		byteTotal += st.Size()
	}
	sort.Strings(filesToArchive)
	if len(filesToArchive) == 0 {
		return nil, nil
	}

	ts := now.Format("20060102-150405")
	var refName string
	if opt.SessionID != "" {
		refName = fmt.Sprintf("%s%s/%s", QuarantineNamespace, opt.SessionID, ts)
	} else {
		refName = fmt.Sprintf("%s%s", QuarantineNamespace, ts)
	}

	// If ref name exists, append nanosecond discriminator to prevent collisions.
	if _, err := runner.Run(repoRoot, "rev-parse", "--verify", refName); err == nil {
		refName = fmt.Sprintf("%s-%09d", refName, now.Nanosecond())
	}

	if opt.DryRun {
		return &QuarantineRef{
			Ref:       refName,
			Files:     filesToArchive,
			Count:     len(filesToArchive),
			ByteTotal: byteTotal,
			SessionID: opt.SessionID,
			Timestamp: now.Unix(),
		}, nil
	}

	tmpDir, err := os.MkdirTemp("", "fak-quarantine-idx-")
	if err != nil {
		return nil, fmt.Errorf("create temp index dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	tmpIndex := filepath.Join(tmpDir, "index")
	idxEnv := []string{
		"GIT_INDEX_FILE=" + tmpIndex,
		"GIT_AUTHOR_NAME=fak-quarantine",
		"GIT_AUTHOR_EMAIL=quarantine@fak.local",
		"GIT_COMMITTER_NAME=fak-quarantine",
		"GIT_COMMITTER_EMAIL=quarantine@fak.local",
	}

	const batchSize = 100
	for i := 0; i < len(filesToArchive); i += batchSize {
		end := i + batchSize
		if end > len(filesToArchive) {
			end = len(filesToArchive)
		}
		args := append([]string{"add", "-f", "--"}, filesToArchive[i:end]...)
		if _, err := runWithEnv(runner, repoRoot, idxEnv, args...); err != nil {
			return nil, fmt.Errorf("stage files to quarantine index: %w", err)
		}
	}

	treeBytes, err := runWithEnv(runner, repoRoot, idxEnv, "write-tree")
	if err != nil {
		return nil, fmt.Errorf("write quarantine tree: %w", err)
	}
	treeSHA := strings.TrimSpace(string(treeBytes))

	commitMsg := fmt.Sprintf("fak-quarantine: archived %d orphan file(s)\n\nFiles: %d\nBytes: %d\nSession: %s\nTimestamp: %s\nReason: %s\n",
		len(filesToArchive), len(filesToArchive), byteTotal, opt.SessionID, now.Format(time.RFC3339), opt.Reason)

	commitBytes, err := runWithEnv(runner, repoRoot, idxEnv, "commit-tree", treeSHA, "-m", commitMsg)
	if err != nil {
		return nil, fmt.Errorf("commit quarantine tree: %w", err)
	}
	commitSHA := strings.TrimSpace(string(commitBytes))

	if _, err := runWithEnv(runner, repoRoot, nil, "update-ref", refName, commitSHA); err != nil {
		return nil, fmt.Errorf("update quarantine ref %s: %w", refName, err)
	}

	for _, f := range filesToArchive {
		full := filepath.Join(repoRoot, filepath.FromSlash(f))
		if err := os.Remove(full); err == nil {
			cleanEmptyDirs(repoRoot, filepath.Dir(full))
		}
	}

	return &QuarantineRef{
		Ref:       refName,
		SHA:       commitSHA,
		TreeSHA:   treeSHA,
		Files:     filesToArchive,
		Count:     len(filesToArchive),
		ByteTotal: byteTotal,
		SessionID: opt.SessionID,
		Timestamp: now.Unix(),
	}, nil
}

// PurgeOrphans is an alias for EvictOrphans allowing clean/purge operations
// backed by a quarantine ref archive.
func PurgeOrphans(ctx context.Context, repoRoot string, runner Runner, opts ...EvictOptions) (*QuarantineRef, error) {
	return EvictOrphans(ctx, repoRoot, runner, opts...)
}

// RestoreQuarantine recovers files from a quarantine ref back into the working tree.
// If paths is empty, all files in the quarantine ref are restored.
func RestoreQuarantine(ctx context.Context, repoRoot string, runner Runner, ref string, paths ...string) error {
	if repoRoot == "" {
		repoRoot = "."
	}
	if runner == nil {
		runner = GitRunner{}
	}
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return fmt.Errorf("restore quarantine: empty ref")
	}
	args := []string{"checkout", ref, "--"}
	if len(paths) > 0 {
		args = append(args, paths...)
	} else {
		args = append(args, ".")
	}
	_, err := runner.Run(repoRoot, args...)
	if err != nil {
		return fmt.Errorf("restore quarantine from %s: %w", ref, err)
	}
	return nil
}

func discoverOrphans(repoRoot string, runner Runner) ([]string, error) {
	out, err := runner.Run(repoRoot, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return nil, fmt.Errorf("status untracked: %w", err)
	}
	var untracked []string
	for _, raw := range bytes.Split(out, []byte{0}) {
		if len(raw) < 4 {
			continue
		}
		line := string(raw)
		if strings.HasPrefix(line, "?? ") {
			untracked = append(untracked, filepath.ToSlash(line[3:]))
		}
	}
	if len(untracked) == 0 {
		return nil, nil
	}

	cpOut, _ := runner.Run(repoRoot, "for-each-ref", "--format=%(objectname)", "refs/fak/wip")
	protected := make(map[string]bool)
	for _, shaLine := range strings.Split(strings.TrimSpace(string(cpOut)), "\n") {
		sha := strings.TrimSpace(shaLine)
		if sha == "" {
			continue
		}
		diffOut, derr := runner.Run(repoRoot, "diff-tree", "--root", "--name-only", "-r", sha)
		if derr == nil {
			for _, row := range strings.Split(strings.TrimSpace(string(diffOut)), "\n") {
				row = strings.TrimSpace(row)
				if row != "" {
					protected[filepath.ToSlash(row)] = true
				}
			}
		}
	}

	var orphans []string
	for _, u := range untracked {
		if !protected[u] {
			orphans = append(orphans, u)
		}
	}
	return orphans, nil
}

func cleanEmptyDirs(repoRoot, dir string) {
	for dir != "" && dir != "." {
		if filepath.Clean(dir) == filepath.Clean(repoRoot) {
			break
		}
		entries, err := os.ReadDir(dir)
		if err != nil || len(entries) > 0 {
			break
		}
		if err := os.Remove(dir); err != nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
}

func runWithEnv(r Runner, dir string, env []string, args ...string) ([]byte, error) {
	if er, ok := r.(EnvRunner); ok {
		return er.RunWithEnv(dir, env, nil, args...)
	}
	return r.Run(dir, args...)
}

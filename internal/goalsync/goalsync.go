package goalsync

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const Schema = "fak-goal-sync/1"

type ArtifactKind string

const (
	KindSpec     ArtifactKind = "spec"
	KindSubagent ArtifactKind = "subagent"
	KindRegistry ArtifactKind = "registry"
	KindPark     ArtifactKind = "park"
)

type Artifact struct {
	RelPath string       `json:"rel_path"`
	AbsPath string       `json:"abs_path"`
	Kind    ArtifactKind `json:"kind"`
	ModTime time.Time    `json:"mod_time"`
	Size    int64        `json:"size"`
	Hash    string       `json:"hash"`
}

type SyncAction string

const (
	ActionPush     SyncAction = "push"
	ActionPull     SyncAction = "pull"
	ActionNoop     SyncAction = "noop"
	ActionConflict SyncAction = "conflict"
)

type ItemStatus struct {
	RelPath       string     `json:"rel_path"`
	Action        SyncAction `json:"action"`
	SourceModTime time.Time  `json:"source_mod_time,omitempty"`
	TargetModTime time.Time  `json:"target_mod_time,omitempty"`
	HashMatch     bool       `json:"hash_match"`
	Reason        string     `json:"reason,omitempty"`
}

type SyncStatus struct {
	TargetDir   string       `json:"target_dir"`
	Items       []ItemStatus `json:"items"`
	TotalCount  int          `json:"total_count"`
	InSyncCount int          `json:"in_sync_count"`
	PushCount   int          `json:"push_count"`
	PullCount   int          `json:"pull_count"`
}

type SyncReport struct {
	Schema      string    `json:"schema"`
	Timestamp   time.Time `json:"timestamp"`
	TargetDir   string    `json:"target_dir"`
	Transferred []string  `json:"transferred"`
	Skipped     []string  `json:"skipped"`
	Committed   bool      `json:"committed"`
	Pushed      bool      `json:"pushed"`
	Error       string    `json:"error,omitempty"`
}

// DefaultTarget returns the canonical target directory inside fak-private.
func DefaultTarget(wsRoot string) string {
	return filepath.Join(wsRoot, "..", "fak-private", "goals", "fak")
}

// computeFileHash calculates the SHA-256 hash, size, and mod time of a file.
func computeFileHash(path string) (string, int64, time.Time, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, time.Time{}, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return "", 0, time.Time{}, err
	}

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, time.Time{}, err
	}
	return hex.EncodeToString(h.Sum(nil)), fi.Size(), fi.ModTime(), nil
}

// DiscoverSource scans wsRoot for goal specs, subagents, the registry file, and goal-park records.
func DiscoverSource(wsRoot, registryPath, goalParkDir string) ([]Artifact, error) {
	var artifacts []Artifact

	// 1. goals/*.md
	goalsDir := filepath.Join(wsRoot, "goals")
	if entries, err := os.ReadDir(goalsDir); err == nil {
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
				continue
			}
			abs := filepath.Join(goalsDir, e.Name())
			hash, size, modTime, err := computeFileHash(abs)
			if err != nil {
				return nil, fmt.Errorf("read source spec %s: %w", abs, err)
			}
			artifacts = append(artifacts, Artifact{
				RelPath: filepath.ToSlash(filepath.Join("goals", e.Name())),
				AbsPath: abs,
				Kind:    KindSpec,
				ModTime: modTime,
				Size:    size,
				Hash:    hash,
			})
		}
	}

	// 2. goals/subagents/*.md
	subDir := filepath.Join(wsRoot, "goals", "subagents")
	if entries, err := os.ReadDir(subDir); err == nil {
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
				continue
			}
			abs := filepath.Join(subDir, e.Name())
			hash, size, modTime, err := computeFileHash(abs)
			if err != nil {
				return nil, fmt.Errorf("read source subagent %s: %w", abs, err)
			}
			artifacts = append(artifacts, Artifact{
				RelPath: filepath.ToSlash(filepath.Join("goals", "subagents", e.Name())),
				AbsPath: abs,
				Kind:    KindSubagent,
				ModTime: modTime,
				Size:    size,
				Hash:    hash,
			})
		}
	}

	// 3. registry file (relative path registry/goals.json)
	regFile := registryPath
	if regFile == "" {
		regFile = filepath.Join(wsRoot, ".fak", "goals.json")
	}
	if fi, err := os.Stat(regFile); err == nil && !fi.IsDir() {
		hash, size, modTime, err := computeFileHash(regFile)
		if err != nil {
			return nil, fmt.Errorf("read source registry %s: %w", regFile, err)
		}
		artifacts = append(artifacts, Artifact{
			RelPath: "registry/goals.json",
			AbsPath: regFile,
			Kind:    KindRegistry,
			ModTime: modTime,
			Size:    size,
			Hash:    hash,
		})
	}

	// 4. goal-park files (relative path goal-park/<filename>)
	parkDir := goalParkDir
	if parkDir == "" {
		parkDir = filepath.Join(wsRoot, ".fak", "goal-park")
	}
	if entries, err := os.ReadDir(parkDir); err == nil {
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			abs := filepath.Join(parkDir, e.Name())
			hash, size, modTime, err := computeFileHash(abs)
			if err != nil {
				return nil, fmt.Errorf("read source park %s: %w", abs, err)
			}
			artifacts = append(artifacts, Artifact{
				RelPath: filepath.ToSlash(filepath.Join("goal-park", e.Name())),
				AbsPath: abs,
				Kind:    KindPark,
				ModTime: modTime,
				Size:    size,
				Hash:    hash,
			})
		}
	}

	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].RelPath < artifacts[j].RelPath
	})
	return artifacts, nil
}

// DiscoverTarget scans targetDir for existing synchronized artifacts.
func DiscoverTarget(targetDir string) ([]Artifact, error) {
	if _, err := os.Stat(targetDir); os.IsNotExist(err) {
		return []Artifact{}, nil
	}

	var artifacts []Artifact
	err := filepath.WalkDir(targetDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(targetDir, path)
		if err != nil {
			return err
		}
		relSlash := filepath.ToSlash(rel)

		var kind ArtifactKind
		switch {
		case strings.HasPrefix(relSlash, "goals/subagents/") && strings.HasSuffix(relSlash, ".md"):
			kind = KindSubagent
		case strings.HasPrefix(relSlash, "goals/") && strings.HasSuffix(relSlash, ".md"):
			kind = KindSpec
		case relSlash == "registry/goals.json" || strings.HasPrefix(relSlash, "registry/"):
			kind = KindRegistry
		case strings.HasPrefix(relSlash, "goal-park/") && strings.HasSuffix(relSlash, ".json"):
			kind = KindPark
		default:
			kind = KindSpec
		}

		hash, size, modTime, err := computeFileHash(path)
		if err != nil {
			return fmt.Errorf("read target artifact %s: %w", path, err)
		}
		artifacts = append(artifacts, Artifact{
			RelPath: relSlash,
			AbsPath: path,
			Kind:    kind,
			ModTime: modTime,
			Size:    size,
			Hash:    hash,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].RelPath < artifacts[j].RelPath
	})
	return artifacts, nil
}

// Status compares discovered source and target artifacts.
func Status(wsRoot, targetDir, registryPath, goalParkDir string) (*SyncStatus, error) {
	srcs, err := DiscoverSource(wsRoot, registryPath, goalParkDir)
	if err != nil {
		return nil, fmt.Errorf("discover source: %w", err)
	}
	tgts, err := DiscoverTarget(targetDir)
	if err != nil {
		return nil, fmt.Errorf("discover target: %w", err)
	}

	srcMap := make(map[string]Artifact, len(srcs))
	for _, s := range srcs {
		srcMap[s.RelPath] = s
	}
	tgtMap := make(map[string]Artifact, len(tgts))
	for _, t := range tgts {
		tgtMap[t.RelPath] = t
	}

	keySet := make(map[string]bool)
	for k := range srcMap {
		keySet[k] = true
	}
	for k := range tgtMap {
		keySet[k] = true
	}

	keys := make([]string, 0, len(keySet))
	for k := range keySet {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var items []ItemStatus
	var inSyncCount, pushCount, pullCount int

	for _, rel := range keys {
		src, hasSrc := srcMap[rel]
		tgt, hasTgt := tgtMap[rel]

		item := ItemStatus{RelPath: rel}
		if hasSrc && !hasTgt {
			item.Action = ActionPush
			item.SourceModTime = src.ModTime
			item.HashMatch = false
			item.Reason = "only in source"
			pushCount++
		} else if !hasSrc && hasTgt {
			item.Action = ActionPull
			item.TargetModTime = tgt.ModTime
			item.HashMatch = false
			item.Reason = "only in target"
			pullCount++
		} else {
			item.SourceModTime = src.ModTime
			item.TargetModTime = tgt.ModTime
			if src.Hash == tgt.Hash {
				item.Action = ActionNoop
				item.HashMatch = true
				item.Reason = "in sync"
				inSyncCount++
			} else {
				item.HashMatch = false
				if src.ModTime.After(tgt.ModTime) {
					item.Action = ActionPush
					item.Reason = "source newer"
					pushCount++
				} else if tgt.ModTime.After(src.ModTime) {
					item.Action = ActionPull
					item.Reason = "target newer"
					pullCount++
				} else {
					item.Action = ActionConflict
					item.Reason = "content mismatch"
				}
			}
		}
		items = append(items, item)
	}

	return &SyncStatus{
		TargetDir:   targetDir,
		Items:       items,
		TotalCount:  len(items),
		InSyncCount: inSyncCount,
		PushCount:   pushCount,
		PullCount:   pullCount,
	}, nil
}

// readWithConcurrencyProtection reads absPath while verifying the file was not modified during read.
func readWithConcurrencyProtection(absPath string) ([]byte, os.FileInfo, error) {
	fi1, err := os.Stat(absPath)
	if err != nil {
		return nil, nil, err
	}
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, nil, err
	}
	fi2, err := os.Stat(absPath)
	if err != nil {
		return nil, nil, err
	}
	if !fi1.ModTime().Equal(fi2.ModTime()) || fi1.Size() != fi2.Size() {
		return nil, nil, fmt.Errorf("concurrency protection: file modified during read (%s)", absPath)
	}
	return data, fi1, nil
}

// atomicWriteFile safely writes data to destPath using os.CreateTemp and os.Rename.
func atomicWriteFile(destPath string, data []byte, perm os.FileMode, modTime time.Time) error {
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(destDir, ".tmp-goalsync-*")
	if err != nil {
		return err
	}
	tmpName := tmpFile.Name()
	cleanTmp := true
	defer func() {
		if cleanTmp {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmpFile.Write(data); err != nil {
		_ = tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}
	_ = os.Chmod(tmpName, perm)
	if err := os.Rename(tmpName, destPath); err != nil {
		return err
	}
	cleanTmp = false
	if !modTime.IsZero() {
		_ = os.Chtimes(destPath, modTime, modTime)
	}
	return nil
}

func isolatedGitEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, ok := strings.Cut(kv, "=")
		if ok {
			u := strings.ToUpper(k)
			if u == "GIT_DIR" || u == "GIT_WORK_TREE" || u == "GIT_INDEX_FILE" || u == "GIT_OBJECT_DIRECTORY" || u == "GIT_PREFIX" || u == "GIT_COMMON_DIR" {
				continue
			}
		}
		env = append(env, kv)
	}
	return env
}

func configureGitCommand(cmd *exec.Cmd) {
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = isolatedGitEnv()
}

// findGitRoot discovers the root of the git repository containing dir.
func findGitRoot(dir string) (string, error) {
	cmd := exec.Command("git", "-C", dir, "rev-parse", "--show-toplevel")
	configureGitCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git rev-parse in %s failed: %w", dir, err)
	}
	return strings.TrimSpace(string(out)), nil
}

// Push synchronizes artifacts from wsRoot to targetDir.
func Push(wsRoot, targetDir, registryPath, goalParkDir string, commit, push, dryRun bool) (*SyncReport, error) {
	report := &SyncReport{
		Schema:      Schema,
		Timestamp:   time.Now().UTC(),
		TargetDir:   targetDir,
		Transferred: []string{},
		Skipped:     []string{},
	}

	srcs, err := DiscoverSource(wsRoot, registryPath, goalParkDir)
	if err != nil {
		report.Error = err.Error()
		return report, fmt.Errorf("discover source: %w", err)
	}
	tgts, err := DiscoverTarget(targetDir)
	if err != nil {
		report.Error = err.Error()
		return report, fmt.Errorf("discover target: %w", err)
	}

	tgtMap := make(map[string]Artifact, len(tgts))
	for _, t := range tgts {
		tgtMap[t.RelPath] = t
	}

	for _, src := range srcs {
		if tgt, exists := tgtMap[src.RelPath]; exists && tgt.Hash == src.Hash {
			report.Skipped = append(report.Skipped, src.RelPath)
			continue
		}

		if dryRun {
			report.Transferred = append(report.Transferred, src.RelPath)
			continue
		}

		destPath := filepath.Join(targetDir, filepath.FromSlash(src.RelPath))
		data, fi, err := readWithConcurrencyProtection(src.AbsPath)
		if err != nil {
			report.Error = err.Error()
			return report, err
		}

		if err := atomicWriteFile(destPath, data, fi.Mode().Perm(), fi.ModTime()); err != nil {
			report.Error = err.Error()
			return report, fmt.Errorf("atomic write to %s: %w", destPath, err)
		}
		report.Transferred = append(report.Transferred, src.RelPath)
	}

	if dryRun {
		return report, nil
	}

	if commit || push {
		root, err := findGitRoot(targetDir)
		if err != nil {
			report.Error = err.Error()
			return report, fmt.Errorf("git root discovery: %w", err)
		}
		resolvedTarget, err := filepath.EvalSymlinks(targetDir)
		if err != nil {
			resolvedTarget = targetDir
		}
		resolvedRoot, err := filepath.EvalSymlinks(root)
		if err != nil {
			resolvedRoot = root
		}
		targetRel, err := filepath.Rel(resolvedRoot, resolvedTarget)
		if err != nil {
			targetRel = "."
		}

		if commit {
			addCmd := exec.Command("git", "-C", resolvedRoot, "add", targetRel)
			configureGitCommand(addCmd)
			if out, err := addCmd.CombinedOutput(); err != nil {
				err := fmt.Errorf("git add failed: %s: %w", strings.TrimSpace(string(out)), err)
				report.Error = err.Error()
				return report, err
			}

			diffCmd := exec.Command("git", "-C", resolvedRoot, "diff", "--cached", "--quiet")
			configureGitCommand(diffCmd)
			if err := diffCmd.Run(); err != nil {
				commitCmd := exec.Command("git", "-C", resolvedRoot, "commit", "-m", "chore(goals): sync goal artifacts (fak)")
				configureGitCommand(commitCmd)
				if out, err := commitCmd.CombinedOutput(); err != nil {
					err := fmt.Errorf("git commit failed: %s: %w", strings.TrimSpace(string(out)), err)
					report.Error = err.Error()
					return report, err
				}
				report.Committed = true
			}
		}

		if push {
			pushCmd := exec.Command("git", "-C", resolvedRoot, "push")
			configureGitCommand(pushCmd)
			if out, err := pushCmd.CombinedOutput(); err != nil {
				err := fmt.Errorf("git push failed: %s: %w", strings.TrimSpace(string(out)), err)
				report.Error = err.Error()
				return report, err
			}
			report.Pushed = true
		}
	}

	return report, nil
}

// resolveLocalPath determines where a synchronized artifact lives inside wsRoot.
func resolveLocalPath(wsRoot, registryPath, goalParkDir, relPath string) string {
	if strings.HasPrefix(relPath, "goals/") {
		return filepath.Join(wsRoot, filepath.FromSlash(relPath))
	}
	if relPath == "registry/goals.json" {
		if registryPath != "" {
			return registryPath
		}
		return filepath.Join(wsRoot, ".fak", "goals.json")
	}
	if strings.HasPrefix(relPath, "goal-park/") {
		fn := strings.TrimPrefix(relPath, "goal-park/")
		if goalParkDir != "" {
			return filepath.Join(goalParkDir, fn)
		}
		return filepath.Join(wsRoot, ".fak", "goal-park", fn)
	}
	return filepath.Join(wsRoot, filepath.FromSlash(relPath))
}

// Pull restores artifacts from targetDir back to wsRoot.
// Refuses to overwrite newer local files unless force is true.
func Pull(wsRoot, targetDir, registryPath, goalParkDir string, force, dryRun bool) (*SyncReport, error) {
	report := &SyncReport{
		Schema:      Schema,
		Timestamp:   time.Now().UTC(),
		TargetDir:   targetDir,
		Transferred: []string{},
		Skipped:     []string{},
	}

	tgts, err := DiscoverTarget(targetDir)
	if err != nil {
		report.Error = err.Error()
		return report, fmt.Errorf("discover target: %w", err)
	}
	srcs, err := DiscoverSource(wsRoot, registryPath, goalParkDir)
	if err != nil {
		report.Error = err.Error()
		return report, fmt.Errorf("discover source: %w", err)
	}

	srcMap := make(map[string]Artifact, len(srcs))
	for _, s := range srcs {
		srcMap[s.RelPath] = s
	}

	// Preflight conflict check if !force
	if !force {
		for _, tgt := range tgts {
			if src, exists := srcMap[tgt.RelPath]; exists {
				if src.Hash != tgt.Hash && src.ModTime.After(tgt.ModTime) {
					err := fmt.Errorf("conflict: refusing to overwrite newer local file %q (use --force)", tgt.RelPath)
					report.Error = err.Error()
					return report, err
				}
			}
		}
	}

	for _, tgt := range tgts {
		src, exists := srcMap[tgt.RelPath]
		if exists && src.Hash == tgt.Hash {
			report.Skipped = append(report.Skipped, tgt.RelPath)
			continue
		}

		if dryRun {
			report.Transferred = append(report.Transferred, tgt.RelPath)
			continue
		}

		localPath := resolveLocalPath(wsRoot, registryPath, goalParkDir, tgt.RelPath)
		data, fi, err := readWithConcurrencyProtection(tgt.AbsPath)
		if err != nil {
			report.Error = err.Error()
			return report, err
		}

		if err := atomicWriteFile(localPath, data, fi.Mode().Perm(), fi.ModTime()); err != nil {
			report.Error = err.Error()
			return report, fmt.Errorf("atomic write to local %s: %w", localPath, err)
		}
		report.Transferred = append(report.Transferred, tgt.RelPath)
	}

	return report, nil
}

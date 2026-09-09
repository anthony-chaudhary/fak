package safesync

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// HealOptions configures the checkout healing operation.
type HealOptions struct {
	Repo   string `json:"repo"`
	DryRun bool   `json:"dry_run,omitempty"`
	Runner Runner `json:"-"`
}

// HealResult reports the outcome of the heal operation.
type HealResult struct {
	OK              bool     `json:"ok"`
	DryRun          bool     `json:"dry_run,omitempty"`
	MissingFiles    []string `json:"missing_files,omitempty"`
	StagedDeleted   []string `json:"staged_deleted,omitempty"`
	UnstagedDeleted []string `json:"unstaged_deleted,omitempty"`
	RestoredFiles   []string `json:"restored_files,omitempty"`
	PreservedDirty  []string `json:"preserved_dirty,omitempty"`
	HealedCount     int      `json:"healed_count"`
	Message         string   `json:"message,omitempty"`
}

func normalizeHealOptions(opts HealOptions) HealOptions {
	if strings.TrimSpace(opts.Repo) == "" {
		opts.Repo = "."
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	return opts
}

// Heal identifies missing files on disk that exist in HEAD, un-stages phantom
// deletions in .git/index, checks out missing files from HEAD, and refreshes the
// index while strictly preserving uncommitted edits in existing dirty files.
func Heal(ctx context.Context, opts HealOptions) (HealResult, error) {
	opts = normalizeHealOptions(opts)
	run := opts.Runner

	_, err := rev(ctx, run, opts.Repo, "HEAD")
	if err != nil {
		return HealResult{
			OK:      false,
			Message: "HEAD not found: " + err.Error(),
		}, nil
	}

	resAll, err := checked(ctx, run, opts.Repo, "diff-index", "--name-only", "-z", "--diff-filter=D", "HEAD")
	if err != nil {
		return HealResult{OK: false}, fmt.Errorf("diff-index HEAD: %w", err)
	}

	resCached, err := checked(ctx, run, opts.Repo, "diff-index", "--cached", "--name-only", "-z", "--diff-filter=D", "HEAD")
	if err != nil {
		return HealResult{OK: false}, fmt.Errorf("diff-index --cached HEAD: %w", err)
	}

	stagedDeletionsMap := make(map[string]bool)
	for _, p := range splitNUL(resCached) {
		clean := filepath.Clean(filepath.ToSlash(p))
		stagedDeletionsMap[clean] = true
	}

	candidateSet := make(map[string]bool)
	for _, p := range splitNUL(resAll) {
		clean := filepath.Clean(filepath.ToSlash(p))
		candidateSet[clean] = true
	}
	for _, p := range splitNUL(resCached) {
		clean := filepath.Clean(filepath.ToSlash(p))
		candidateSet[clean] = true
	}

	var missingFiles []string
	for p := range candidateSet {
		if p == "" || p == "." {
			continue
		}
		fullPath := filepath.Join(opts.Repo, filepath.FromSlash(p))
		statInfo, statErr := os.Lstat(fullPath)
		_ = statInfo
		if statErr != nil && os.IsNotExist(statErr) {
			catRes := run(ctx, opts.Repo, "cat-file", "-e", "HEAD:"+p)
			if catRes.Err == nil && catRes.Code == 0 {
				missingFiles = append(missingFiles, p)
			}
		}
	}
	sort.Strings(missingFiles)

	var stagedDeleted []string
	var unstagedDeleted []string
	for _, p := range missingFiles {
		if stagedDeletionsMap[p] {
			stagedDeleted = append(stagedDeleted, p)
		} else {
			unstagedDeleted = append(unstagedDeleted, p)
		}
	}
	sort.Strings(stagedDeleted)
	sort.Strings(unstagedDeleted)

	dirtyPaths, _ := workingTreeDirtyPaths(ctx, run, opts.Repo)
	missingSet := make(map[string]bool, len(missingFiles))
	for _, m := range missingFiles {
		missingSet[m] = true
	}
	var preservedDirty []string
	for _, d := range dirtyPaths {
		if !missingSet[d] {
			preservedDirty = append(preservedDirty, d)
		}
	}
	sort.Strings(preservedDirty)

	if len(missingFiles) == 0 {
		return HealResult{
			OK:             true,
			DryRun:         opts.DryRun,
			PreservedDirty: preservedDirty,
			HealedCount:    0,
			Message:        "checkout is clean; no phantom deletions to heal",
		}, nil
	}

	if opts.DryRun {
		return HealResult{
			OK:              true,
			DryRun:          true,
			MissingFiles:    missingFiles,
			StagedDeleted:   stagedDeleted,
			UnstagedDeleted: unstagedDeleted,
			RestoredFiles:   missingFiles,
			PreservedDirty:  preservedDirty,
			HealedCount:     len(missingFiles),
			Message:         fmt.Sprintf("dry run: %d phantom deletion(s) identified for healing", len(missingFiles)),
		}, nil
	}

	const batchSize = 100

	if len(stagedDeleted) > 0 {
		for i := 0; i < len(stagedDeleted); i += batchSize {
			end := i + batchSize
			if end > len(stagedDeleted) {
				end = len(stagedDeleted)
			}
			batch := stagedDeleted[i:end]
			args := append([]string{"reset", "HEAD", "--"}, batch...)
			if _, err := checked(ctx, run, opts.Repo, args...); err != nil {
				return HealResult{OK: false}, fmt.Errorf("git reset HEAD: %w", err)
			}
		}
	}

	for i := 0; i < len(missingFiles); i += batchSize {
		end := i + batchSize
		if end > len(missingFiles) {
			end = len(missingFiles)
		}
		batch := missingFiles[i:end]
		args := append([]string{"checkout", "HEAD", "--"}, batch...)
		if _, err := checked(ctx, run, opts.Repo, args...); err != nil {
			return HealResult{OK: false}, fmt.Errorf("git checkout HEAD: %w", err)
		}
	}

	if _, err := checked(ctx, run, opts.Repo, "update-index", "-q", "--refresh"); err != nil {
		return HealResult{OK: false}, fmt.Errorf("git update-index --refresh: %w", err)
	}

	dirtyAfter, _ := workingTreeDirtyPaths(ctx, run, opts.Repo)
	return HealResult{
		OK:              true,
		DryRun:          false,
		MissingFiles:    missingFiles,
		StagedDeleted:   stagedDeleted,
		UnstagedDeleted: unstagedDeleted,
		RestoredFiles:   missingFiles,
		PreservedDirty:  dirtyAfter,
		HealedCount:     len(missingFiles),
		Message:         fmt.Sprintf("healed %d phantom deletion(s)", len(missingFiles)),
	}, nil
}

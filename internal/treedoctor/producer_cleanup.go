package treedoctor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

const (
	ScratchProducerReceiptSchema = "fak-tree-doctor-scratch-producer/1"
	ScratchProducerReaped        = "reaped"
	ScratchProducerAbsent        = "absent"
	ScratchProducerRefused       = "refused"
	ScratchProducerFailed        = "failed"
)

// ErrUnsafeScratchProducer marks a request that was refused before deletion because its
// producer name or resolved filesystem tree was not one exact, contained scratch directory.
var ErrUnsafeScratchProducer = errors.New("unsafe scratch producer cleanup")

// ScratchProducerReceipt is the human/JSON contract for one exact producer cleanup.
// RemovedCount includes the selected producer directory itself and every removed descendant.
type ScratchProducerReceipt struct {
	Schema         string `json:"schema"`
	Producer       string `json:"producer"`
	ResolvedTarget string `json:"resolved_target,omitempty"`
	Verdict        string `json:"verdict"`
	RemovedCount   int    `json:"removed_count"`
	Error          string `json:"error,omitempty"`
}

// CleanScratchProducer removes exactly repoRoot/_scratch/<producer>. The producer must be one
// literal top-level name: paths, traversal, glob syntax, and the namespace root are refused.
// The producer root is resolved and checked before the first removal. Descendant symlinks and
// Windows reparse points are enumerated as leaf entries and unlinked without traversing their
// targets. Deletion uses exact paths bottom-up, never a recursive wildcard or ignored-ancestor
// git pathspec.
func CleanScratchProducer(repoRoot, producer string) (ScratchProducerReceipt, error) {
	receipt := ScratchProducerReceipt{
		Schema:   ScratchProducerReceiptSchema,
		Producer: producer,
		Verdict:  ScratchProducerRefused,
	}
	if err := validateScratchProducerName(producer); err != nil {
		return refuseScratchProducer(receipt, err)
	}

	repo, err := filepath.Abs(repoRoot)
	if err != nil {
		return failScratchProducer(receipt, fmt.Errorf("resolve repository root: %w", err))
	}
	repo, err = filepath.EvalSymlinks(repo)
	if err != nil {
		return failScratchProducer(receipt, fmt.Errorf("resolve repository root: %w", err))
	}
	repoInfo, err := os.Stat(repo)
	if err != nil {
		return failScratchProducer(receipt, fmt.Errorf("inspect repository root: %w", err))
	}
	if !repoInfo.IsDir() {
		return refuseScratchProducer(receipt, fmt.Errorf("repository root %q is not a directory", repo))
	}

	scratchRoot := filepath.Join(repo, scratchNamespace)
	target := filepath.Join(scratchRoot, producer)
	receipt.ResolvedTarget = target
	if err := requireDirectScratchChild(scratchRoot, target, producer); err != nil {
		return refuseScratchProducer(receipt, err)
	}

	receipt, terminal, err := inspectScratchDirectory(receipt, scratchRoot, "scratch root")
	if terminal {
		return receipt, err
	}
	resolvedScratch, err := filepath.EvalSymlinks(scratchRoot)
	if err != nil {
		return failScratchProducer(receipt, fmt.Errorf("resolve scratch root: %w", err))
	}
	if !sameFilesystemPath(resolvedScratch, scratchRoot) {
		return refuseScratchProducer(receipt, fmt.Errorf("scratch root resolves outside its declared path: %s", resolvedScratch))
	}

	receipt, terminal, err = inspectScratchDirectory(receipt, target, "producer target")
	if terminal {
		return receipt, err
	}
	resolvedTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		return failScratchProducer(receipt, fmt.Errorf("resolve producer target: %w", err))
	}
	receipt.ResolvedTarget = resolvedTarget
	if !sameFilesystemPath(resolvedTarget, target) {
		return refuseScratchProducer(receipt, fmt.Errorf("producer target resolves outside its declared path: %s", resolvedTarget))
	}
	if err := requireDirectScratchChild(resolvedScratch, resolvedTarget, producer); err != nil {
		return refuseScratchProducer(receipt, err)
	}

	files, directories, err := enumerateScratchProducer(resolvedTarget)
	if err != nil {
		if errors.Is(err, ErrUnsafeScratchProducer) {
			return refuseScratchProducer(receipt, err)
		}
		return failScratchProducer(receipt, err)
	}
	removed, err := removeScratchProducerExact(files, directories)
	receipt.RemovedCount = removed
	if err != nil {
		return failScratchProducer(receipt, err)
	}
	receipt.Verdict = ScratchProducerReaped
	return receipt, nil
}

func validateScratchProducerName(producer string) error {
	if producer == "" || strings.TrimSpace(producer) != producer {
		return fmt.Errorf("producer name is empty or has surrounding whitespace")
	}
	if producer == "." || producer == ".." || producer == scratchNamespace {
		return fmt.Errorf("producer %q names a protected root", producer)
	}
	if filepath.IsAbs(producer) || filepath.VolumeName(producer) != "" {
		return fmt.Errorf("producer %q must be a repository-relative name", producer)
	}
	if strings.ContainsAny(producer, `/\\`) {
		return fmt.Errorf("producer %q must be one top-level name", producer)
	}
	if strings.ContainsAny(producer, "*?[") {
		return fmt.Errorf("producer %q contains glob syntax", producer)
	}
	if filepath.Clean(producer) != producer || filepath.Base(producer) != producer {
		return fmt.Errorf("producer %q is not one clean path component", producer)
	}
	return nil
}

func requireDirectScratchChild(scratchRoot, target, producer string) error {
	rel, err := filepath.Rel(scratchRoot, target)
	if err != nil {
		return fmt.Errorf("resolve producer relative to scratch root: %w", err)
	}
	if filepath.Dir(rel) != "." || !sameFilesystemPath(filepath.Base(rel), producer) {
		return fmt.Errorf("producer target %q is not directly beneath %q", target, scratchRoot)
	}
	return nil
}

func inspectScratchDirectory(receipt ScratchProducerReceipt, path, label string) (ScratchProducerReceipt, bool, error) {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		receipt.Verdict = ScratchProducerAbsent
		return receipt, true, nil
	}
	if err != nil {
		receipt, err = failScratchProducer(receipt, fmt.Errorf("inspect %s: %w", label, err))
		return receipt, true, err
	}
	if err := refuseReparse(path, info); err != nil {
		receipt, err = refuseScratchProducer(receipt, err)
		return receipt, true, err
	}
	if !info.IsDir() {
		receipt, err = refuseScratchProducer(receipt, fmt.Errorf("%s %q is not a directory", label, path))
		return receipt, true, err
	}
	return receipt, false, nil
}
func enumerateScratchProducer(root string) (files []string, directories []string, retErr error) {
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("%w: enumerated path escaped producer root: %s", ErrUnsafeScratchProducer, path)
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		reparse, err := goTmpIsReparse(path, info)
		if err != nil {
			return fmt.Errorf("inspect producer entry %s for reparse points: %w", path, err)
		}
		if reparse {
			files = append(files, path)
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if info.IsDir() {
			directories = append(directories, path)
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("enumerate producer tree: %w", err)
	}
	return files, directories, nil
}

func removeScratchProducerExact(files, directories []string) (int, error) {
	removed := 0
	for _, path := range files {
		info, err := os.Lstat(path)
		if err != nil {
			return removed, fmt.Errorf("recheck producer entry %s: %w", path, err)
		}
		if _, err := goTmpIsReparse(path, info); err != nil {
			return removed, fmt.Errorf("inspect producer entry %s for reparse points: %w", path, err)
		}
		if err := os.Remove(path); err != nil {
			return removed, fmt.Errorf("remove producer entry %s: %w", path, err)
		}
		removed++
	}
	sort.Slice(directories, func(i, j int) bool {
		return len(directories[i]) > len(directories[j])
	})
	for _, path := range directories {
		info, err := os.Lstat(path)
		if err != nil {
			return removed, fmt.Errorf("recheck producer directory %s: %w", path, err)
		}
		if _, err := goTmpIsReparse(path, info); err != nil {
			return removed, fmt.Errorf("inspect producer directory %s for reparse points: %w", path, err)
		}
		if err := os.Remove(path); err != nil {
			return removed, fmt.Errorf("remove producer directory %s: %w", path, err)
		}
		removed++
	}
	return removed, nil
}

func refuseReparse(path string, info os.FileInfo) error {
	reparse, err := goTmpIsReparse(path, info)
	if err != nil {
		return fmt.Errorf("inspect producer entry %s for reparse points: %w", path, err)
	}
	if reparse {
		return fmt.Errorf("%w: producer tree contains a symlink or reparse point: %s", ErrUnsafeScratchProducer, path)
	}
	return nil
}

func refuseScratchProducer(receipt ScratchProducerReceipt, cause error) (ScratchProducerReceipt, error) {
	err := cause
	if !errors.Is(err, ErrUnsafeScratchProducer) {
		err = fmt.Errorf("%w: %v", ErrUnsafeScratchProducer, cause)
	}
	receipt.Verdict = ScratchProducerRefused
	receipt.RemovedCount = 0
	receipt.Error = err.Error()
	return receipt, err
}

func failScratchProducer(receipt ScratchProducerReceipt, err error) (ScratchProducerReceipt, error) {
	receipt.Verdict = ScratchProducerFailed
	receipt.Error = err.Error()
	return receipt, err
}

func sameFilesystemPath(a, b string) bool {
	a = filepath.Clean(a)
	b = filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

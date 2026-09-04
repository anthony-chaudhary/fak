package treedoctor

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultScratchGoFilesThreshold is the maximum number of untracked .go files
// permitted in _scratch before a scratch hygiene warning is triggered to prevent
// language server memory explosion.
const DefaultScratchGoFilesThreshold = 10000

// ScratchHygieneReport classifies untracked .go files in _scratch.
type ScratchHygieneReport struct {
	ScratchUntrackedGoFiles int    `json:"scratch_untracked_go_files"`
	Threshold               int    `json:"threshold"`
	Exceeded                bool   `json:"exceeded"`
	Warning                 string `json:"warning,omitempty"`
}

// diagnoseScratchHygiene inspects repoRoot/_scratch for untracked .go files
// against DefaultScratchGoFilesThreshold (10,000).
func diagnoseScratchHygiene(repoRoot string) ScratchHygieneReport {
	return DiagnoseScratchHygieneThreshold(repoRoot, DefaultScratchGoFilesThreshold)
}

// DiagnoseScratchHygiene inspects repoRoot/_scratch for untracked .go files
// against DefaultScratchGoFilesThreshold.
func DiagnoseScratchHygiene(repoRoot string) ScratchHygieneReport {
	return DiagnoseScratchHygieneThreshold(repoRoot, DefaultScratchGoFilesThreshold)
}

// DiagnoseScratchHygieneThreshold inspects repoRoot/_scratch for untracked .go files
// against the provided threshold.
func DiagnoseScratchHygieneThreshold(repoRoot string, threshold int) ScratchHygieneReport {
	if threshold <= 0 {
		threshold = DefaultScratchGoFilesThreshold
	}
	if repoRoot == "" {
		return ScratchHygieneReport{Threshold: threshold}
	}
	scratchDir := filepath.Join(repoRoot, scratchNamespace)
	info, err := os.Stat(scratchDir)
	if err != nil || !info.IsDir() {
		return ScratchHygieneReport{Threshold: threshold}
	}

	count := 0
	_ = filepath.WalkDir(scratchDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(d.Name(), ".go") {
			count++
		}
		return nil
	})

	return BuildScratchHygieneReport(count, threshold)
}

// BuildScratchHygieneReport constructs a ScratchHygieneReport from an untracked
// .go file count and threshold.
func BuildScratchHygieneReport(count, threshold int) ScratchHygieneReport {
	if threshold <= 0 {
		threshold = DefaultScratchGoFilesThreshold
	}
	rep := ScratchHygieneReport{
		ScratchUntrackedGoFiles: count,
		Threshold:               threshold,
	}
	if count > threshold {
		rep.Exceeded = true
		if threshold == DefaultScratchGoFilesThreshold {
			rep.Warning = fmt.Sprintf("_scratch contains >10,000 untracked .go files (%d) without quarantine; isolate workspace scope or reap scratch to prevent LSP/gopls memory explosion", count)
		} else {
			rep.Warning = fmt.Sprintf("_scratch contains >%d untracked .go files (%d) without quarantine; isolate workspace scope or reap scratch to prevent LSP/gopls memory explosion", threshold, count)
		}
	}
	return rep
}

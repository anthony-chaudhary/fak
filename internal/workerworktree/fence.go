package workerworktree

import (
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
)

// ReasonOutOfLaneMutation is the stable token emitted when worker writes violate
// declared lease pathspec boundaries.
const ReasonOutOfLaneMutation = "OUT_OF_LANE_MUTATION"

// ErrOutOfLaneMutation is the sentinel error for pathspec fence violations.
var ErrOutOfLaneMutation = errors.New(ReasonOutOfLaneMutation)

// OutOfLaneMutationError records files modified outside leased globs.
type OutOfLaneMutationError struct {
	Violations []string
}

func (e *OutOfLaneMutationError) Error() string {
	return fmt.Sprintf("%s: out-of-lane mutation detected: %s", ReasonOutOfLaneMutation, strings.Join(e.Violations, ", "))
}

func (e *OutOfLaneMutationError) Unwrap() error {
	return ErrOutOfLaneMutation
}

// ValidateWorkerTreeDisjointness verifies that all changed files conform to declared
// leased pathspec globs. If leasedGlobs is empty, no fence is declared and nil is returned.
// Each file path in changedFiles is normalized to a slash-separated relative path and
// tested against leasedGlobs. It supports recursive globs ("pkg/**"), directory globs
// ("dir/*"), and exact file paths. If any file does not match any leased glob, it returns
// an error wrapping ReasonOutOfLaneMutation and listing the violating files.
func ValidateWorkerTreeDisjointness(changedFiles []string, leasedGlobs []string) error {
	if len(leasedGlobs) == 0 {
		return nil
	}
	seenViolations := make(map[string]bool)
	var violations []string
	for _, file := range changedFiles {
		normFile := normalizeFencePath(file)
		if normFile == "" {
			continue
		}
		matched := false
		for _, glob := range leasedGlobs {
			normGlob := normalizeFenceGlob(glob)
			if normGlob == "" {
				continue
			}
			if matchFenceGlob(normGlob, normFile) {
				matched = true
				break
			}
		}
		if !matched && !seenViolations[normFile] {
			seenViolations[normFile] = true
			violations = append(violations, normFile)
		}
	}
	if len(violations) > 0 {
		sort.Strings(violations)
		return &OutOfLaneMutationError{Violations: violations}
	}
	return nil
}

func normalizeFencePath(p string) string {
	p = strings.TrimSpace(p)
	p = strings.ReplaceAll(p, "\\", "/")
	p = path.Clean(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	if p == "." {
		return ""
	}
	return p
}

func normalizeFenceGlob(glob string) string {
	glob = strings.TrimSpace(glob)
	glob = strings.ReplaceAll(glob, "\\", "/")
	glob = strings.TrimPrefix(glob, "./")
	glob = strings.TrimPrefix(glob, "/")
	return glob
}

func matchFenceGlob(glob, filePath string) bool {
	if glob == "" || filePath == "" {
		return false
	}
	if glob == "**" || glob == "*" {
		return true
	}
	if glob == filePath {
		return true
	}

	// Exact directory prefix: "internal/workerworktree" or "internal/workerworktree/"
	// matches the exact directory or any child path.
	if !strings.ContainsAny(glob, "*?[") {
		clean := strings.TrimSuffix(glob, "/")
		if filePath == clean || strings.HasPrefix(filePath, clean+"/") {
			return true
		}
		return false
	}

	// "dir/**" matches "dir" itself or any descendant under "dir/"
	if strings.HasSuffix(glob, "/**") {
		prefix := strings.TrimSuffix(glob, "/**")
		prefix = strings.TrimSuffix(prefix, "/")
		if prefix == "" || filePath == prefix || strings.HasPrefix(filePath, prefix+"/") {
			return true
		}
	}

	// Single-level path.Match for globs without **
	if !strings.Contains(glob, "**") {
		if ok, err := path.Match(glob, filePath); err == nil && ok {
			return true
		}
	}

	// Segment-by-segment matching supporting recursive **
	pSegs := strings.Split(glob, "/")
	tSegs := strings.Split(filePath, "/")
	return matchFenceSegments(pSegs, tSegs)
}

func matchFenceSegments(pSegs, tSegs []string) bool {
	if len(pSegs) == 0 {
		return len(tSegs) == 0
	}
	if pSegs[0] == "**" {
		if len(pSegs) == 1 {
			return true
		}
		for i := 0; i <= len(tSegs); i++ {
			if matchFenceSegments(pSegs[1:], tSegs[i:]) {
				return true
			}
		}
		return false
	}
	if len(tSegs) == 0 {
		return false
	}
	matched, err := path.Match(pSegs[0], tSegs[0])
	if err != nil || !matched {
		return false
	}
	return matchFenceSegments(pSegs[1:], tSegs[1:])
}

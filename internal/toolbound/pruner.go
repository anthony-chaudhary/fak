package toolbound

import (
	"path/filepath"
	"strings"
	"sync"
)

// DirectoryPruner prunes paths matching configured directory prefixes.
type DirectoryPruner struct {
	mu      sync.RWMutex
	ignored []string
}

// NewDirectoryPruner creates a new DirectoryPruner configured with the specified ignored directories.
func NewDirectoryPruner(ignoredDirs ...string) *DirectoryPruner {
	dp := &DirectoryPruner{}
	dp.AddIgnored(ignoredDirs...)
	return dp
}

// AddIgnored registers additional directory prefixes to ignore.
func (dp *DirectoryPruner) AddIgnored(dirs ...string) {
	dp.mu.Lock()
	defer dp.mu.Unlock()

	for _, d := range dirs {
		norm := normalizeDir(d)
		if norm == "" {
			continue
		}
		exists := false
		for _, existing := range dp.ignored {
			if existing == norm {
				exists = true
				break
			}
		}
		if !exists {
			dp.ignored = append(dp.ignored, norm)
		}
	}
}

// Ignored returns a copy of the currently configured ignored directory prefixes.
func (dp *DirectoryPruner) Ignored() []string {
	dp.mu.RLock()
	defer dp.mu.RUnlock()

	out := make([]string, len(dp.ignored))
	copy(out, dp.ignored)
	return out
}

// IsIgnored reports whether path falls within any configured ignored directory.
func (dp *DirectoryPruner) IsIgnored(path string) bool {
	dp.mu.RLock()
	defer dp.mu.RUnlock()
	return dp.isIgnoredLocked(path)
}

func (dp *DirectoryPruner) isIgnoredLocked(path string) bool {
	if len(dp.ignored) == 0 {
		return false
	}
	cleanPath := normalizePath(path)
	if cleanPath == "" {
		return false
	}
	for _, dir := range dp.ignored {
		if cleanPath == dir || strings.HasPrefix(cleanPath, dir+"/") {
			return true
		}
	}
	return false
}

// Prune filters out paths that fall within any ignored directory, returning the remaining paths.
func (dp *DirectoryPruner) Prune(paths []string) []string {
	if paths == nil {
		return nil
	}
	if len(paths) == 0 {
		return []string{}
	}

	dp.mu.RLock()
	defer dp.mu.RUnlock()

	if len(dp.ignored) == 0 {
		out := make([]string, len(paths))
		copy(out, paths)
		return out
	}

	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if !dp.isIgnoredLocked(p) {
			out = append(out, p)
		}
	}
	return out
}

// PruneSearchPaths is a standalone helper that prunes paths matching ignored directories.
func PruneSearchPaths(paths []string, ignoredDirs []string) []string {
	pruner := NewDirectoryPruner(ignoredDirs...)
	return pruner.Prune(paths)
}

func normalizePath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" || p == "." {
		return ""
	}
	p = strings.ReplaceAll(p, "\\", "/")
	p = filepath.ToSlash(filepath.Clean(p))
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	if p == "." {
		return ""
	}
	return p
}

func normalizeDir(d string) string {
	d = strings.TrimSpace(d)
	if d == "" {
		return ""
	}
	d = strings.ReplaceAll(d, "\\", "/")
	d = filepath.ToSlash(filepath.Clean(d))
	d = strings.TrimPrefix(d, "./")
	d = strings.TrimPrefix(d, "/")
	d = strings.TrimSuffix(d, "/")
	if d == "." || d == "" {
		return ""
	}
	return d
}

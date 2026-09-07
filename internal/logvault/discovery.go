package logvault

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// DiscoveredInstance represents a log or state store instance discovered
// in the repository that is not registered in the logvault source registry.
type DiscoveredInstance struct {
	Path    string // absolute path to discovered file or directory
	RelPath string // repo-relative path (forward-slash normalized)
	Pattern string // pattern that matched (e.g. "**/.dos", "**/guard-audit/*.jsonl")
	IsDir   bool   // true if the instance is a directory
	Warning string // human-readable warning message
}

func (d DiscoveredInstance) String() string {
	return d.Warning
}

// DefaultDiscoveryPatterns are the canonical instance search patterns:
// - **/.dos: forked DOS trust-kernel state directories
// - **/docs/nightrun/*.jsonl: nightrun ledgers outside the canonical root
// - **/guard-audit/*.jsonl: guard-audit session journals
var DefaultDiscoveryPatterns = []string{
	"**/.dos",
	"**/docs/nightrun/*.jsonl",
	"**/guard-audit/*.jsonl",
}

// normalizePath converts p to an absolute path with normalized Windows drive letter.
func normalizePath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		abs = filepath.Clean(p)
	}
	if len(abs) >= 2 && abs[1] == ':' {
		abs = strings.ToUpper(abs[:1]) + abs[1:]
	}
	return abs
}

// pathWithinNormalized reports whether path is equal to or inside root.
func pathWithinNormalized(path, root string) bool {
	p1 := normalizePath(path)
	p2 := normalizePath(root)
	if strings.EqualFold(p1, p2) {
		return true
	}
	sep := string(os.PathSeparator)
	prefix := strings.ToLower(p2)
	if !strings.HasSuffix(prefix, sep) {
		prefix += sep
	}
	return strings.HasPrefix(strings.ToLower(p1), prefix)
}

// IsRegistered reports whether a discovered file or directory path is covered
// by any source in registered.
func IsRegistered(path string, isDir bool, repoRoot string, registered []Source) bool {
	absPath := normalizePath(path)

	for _, s := range registered {
		sRoot := s.Root
		if repoRoot != "" && !filepath.IsAbs(sRoot) {
			sRoot = filepath.Join(repoRoot, sRoot)
		}
		sRootAbs := normalizePath(sRoot)

		if isDir {
			if strings.EqualFold(absPath, sRootAbs) {
				return true
			}
			if pathWithinNormalized(absPath, sRootAbs) {
				relFromSrc, err := filepath.Rel(sRootAbs, absPath)
				if err == nil && !strings.HasPrefix(relFromSrc, "..") {
					relSlash := filepath.ToSlash(relFromSrc)
					if !excluded(s, relSlash+"/") && includesCouldReach(s, relSlash+"/") {
						return true
					}
				}
			}
		} else {
			if strings.EqualFold(absPath, sRootAbs) {
				return true
			}
			if pathWithinNormalized(absPath, sRootAbs) {
				relFromSrc, err := filepath.Rel(sRootAbs, absPath)
				if err == nil && !strings.HasPrefix(relFromSrc, "..") {
					relSlash := filepath.ToSlash(relFromSrc)
					if !excluded(s, relSlash) && admitted(s, relSlash) {
						return true
					}
				}
			}
		}
	}
	return false
}

// matchInstancePattern matches relPath against pat.
func matchInstancePattern(pat, relPath string, isDir bool) bool {
	patSlash := filepath.ToSlash(pat)
	relSlash := filepath.ToSlash(relPath)

	dirPattern := strings.HasSuffix(patSlash, "/")
	cleanPat := strings.TrimSuffix(patSlash, "/")

	if dirPattern && !isDir {
		return false
	}

	if strings.HasPrefix(cleanPat, "**/") {
		subPat := strings.TrimPrefix(cleanPat, "**/")
		if isDir {
			if relSlash == subPat || strings.HasSuffix(relSlash, "/"+subPat) {
				return true
			}
			matched, _ := filepath.Match(subPat, filepath.Base(relSlash))
			return matched
		}
		if strings.Contains(subPat, "/") {
			expectedDir := filepath.ToSlash(filepath.Dir(subPat))
			expectedFile := filepath.Base(subPat)
			relDir := filepath.ToSlash(filepath.Dir(relSlash))
			relBase := filepath.Base(relSlash)

			fileMatch, _ := filepath.Match(expectedFile, relBase)
			if !fileMatch {
				return false
			}
			return relDir == expectedDir || strings.HasSuffix(relDir, "/"+expectedDir)
		}
		fileMatch, _ := filepath.Match(subPat, filepath.Base(relSlash))
		return fileMatch
	}

	if isDir {
		return relSlash == cleanPat
	}
	matched, _ := filepath.Match(cleanPat, relSlash)
	return matched
}

// DiscoverInstances walks repoRoot searching for instance patterns and returns any
// discovered files or directories that are not already covered by registered sources.
func DiscoverInstances(repoRoot string, registered []Source, patterns ...string) ([]DiscoveredInstance, error) {
	if len(patterns) == 0 {
		patterns = DefaultDiscoveryPatterns
	}
	repoAbs := normalizePath(repoRoot)
	var discovered []DiscoveredInstance

	err := filepath.WalkDir(repoAbs, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil // skip unreadable subtrees gracefully
		}
		relOS, relErr := filepath.Rel(repoAbs, path)
		if relErr != nil || relOS == "." {
			return nil
		}
		relSlash := filepath.ToSlash(relOS)
		isDir := d.IsDir()

		if isDir {
			name := d.Name()
			if name == ".git" || name == ".gocache" || name == ".gotmp" || name == ".history" {
				return filepath.SkipDir
			}
		}

		for _, pat := range patterns {
			if matchInstancePattern(pat, relSlash, isDir) {
				if !IsRegistered(path, isDir, repoAbs, registered) {
					warning := fmt.Sprintf("unregistered instance discovered at %s (pattern %s)", relSlash, pat)
					discovered = append(discovered, DiscoveredInstance{
						Path:    path,
						RelPath: relSlash,
						Pattern: pat,
						IsDir:   isDir,
						Warning: warning,
					})
				}
				if isDir {
					return filepath.SkipDir
				}
				break
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return discovered, nil
}

// DiscoverWarnings runs DiscoverInstances and returns only the formatted warning strings.
func DiscoverWarnings(repoRoot string, registered []Source, patterns ...string) ([]string, error) {
	instances, err := DiscoverInstances(repoRoot, registered, patterns...)
	if err != nil {
		return nil, err
	}
	var warnings []string
	for _, inst := range instances {
		warnings = append(warnings, inst.Warning)
	}
	return warnings, nil
}

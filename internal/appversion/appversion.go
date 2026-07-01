// Package appversion resolves the fleet/FAK application version from the single
// repo-level VERSION marker, with build-time and environment fallbacks.
package appversion

import (
	"os"
	"path/filepath"
	"strings"
)

const fallback = "dev"

// BuildVersion may be set by release builds with:
//
//	-ldflags "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=0.8.0"
//
// BuildVersion wins over VERSION when present so a release binary reports the version it
// was built with instead of inheriting a parent checkout's marker.
var BuildVersion string

// Current returns the best available application version.
func Current() string {
	if v := strings.TrimSpace(os.Getenv("FAK_APP_VERSION")); v != "" {
		return v
	}
	if v := strings.TrimSpace(BuildVersion); v != "" {
		return v
	}
	for _, start := range candidateStarts() {
		if v, ok := FromDir(start); ok {
			return v
		}
	}
	return fallback
}

// FromDir walks upward from start until it finds a VERSION file, but it does not cross a
// repository boundary. A sibling checkout without VERSION must not inherit one from its
// parent directory.
func FromDir(start string) (string, bool) {
	if strings.TrimSpace(start) == "" {
		return "", false
	}
	dir, err := filepath.Abs(start)
	if err != nil {
		return "", false
	}
	if info, err := os.Stat(dir); err == nil && !info.IsDir() {
		dir = filepath.Dir(dir)
	}
	for {
		if v, ok := readVersionFile(filepath.Join(dir, "VERSION")); ok {
			return v, true
		}
		if hasRepoBoundary(dir) {
			return "", false
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func candidateStarts() []string {
	var starts []string
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	return starts
}

func readVersionFile(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	v := strings.TrimSpace(string(b))
	if v == "" {
		return "", false
	}
	return v, true
}

func hasRepoBoundary(dir string) bool {
	if strings.TrimSpace(dir) == "" {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
		return true
	}
	return false
}

// Package appversion resolves the FAK application version from build identity or
// from a VERSION marker that belongs to the running executable.
package appversion

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
)

const (
	fallback                = "dev"
	BenchmarkConceptVersion = "fak.benchmark-concept.v1"
)

// BuildVersion may be set by release builds with:
//
//	-ldflags "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=0.8.0"
//
// BuildVersion wins over VERSION when present so a release binary reports the version it
// was built with instead of inheriting a parent checkout's marker.
var BuildVersion string

// BuildCommit may be set by source-install builds when the Go toolchain cannot derive
// vcs.revision itself (notably from a linked/detached git worktree on Windows). It is the
// full, clean commit SHA selected by the installer and is build provenance, not a friendly
// application version.
var BuildCommit string

// Current returns the running binary's best available application version, resolved
// most-binary-bound first: $FAK_APP_VERSION, the -ldflags BuildVersion, the release tag Go
// embedded in the module build info, then a VERSION marker beside the executable.
//
// VERSION is searched only from the executable directory. In particular, Current must not
// read VERSION from the process working directory: an installed or stale fak is routinely
// launched while cwd is a newer checkout, and borrowing that checkout's marker makes the old
// binary claim the new version. The module-stamp step upholds that same invariant rather than
// weakening it: the tag is written into the binary at link time and travels with it, so a
// `go install <module>@vX.Y.Z` build — which carries no -ldflags stamp and no VERSION marker —
// reports X.Y.Z wherever it runs, while a `go build` from source records "(devel)" or a VCS
// pseudo-version and is refused. Release/dev-profile builds and self-update stamp BuildVersion;
// an unstamped non-release binary outside its source tree still honestly reports "dev".
func Current() string {
	if v := strings.TrimSpace(os.Getenv("FAK_APP_VERSION")); v != "" {
		return v
	}
	if v := strings.TrimSpace(BuildVersion); v != "" {
		return v
	}
	// A go test binary can inherit the module's release version even though it is not an
	// installed fak executable. Keep tests on the same unstamped fallback path as a local
	// development build; real binaries still use their embedded module release.
	base := strings.TrimSuffix(strings.ToLower(filepath.Base(os.Args[0])), ".exe")
	if !strings.HasSuffix(base, ".test") {
		if bi, ok := debug.ReadBuildInfo(); ok {
			if v, ok := releaseFromModuleVersion(bi.Main.Version); ok {
				return v
			}
		}
	}
	if !strings.HasSuffix(base, ".test") {
		if exe, err := os.Executable(); err == nil {
			if v, ok := FromDir(filepath.Dir(exe)); ok {
				return v
			}
		}
	}
	return fallback
}

// releaseFromModuleVersion converts the module version Go embeds in a binary into an
// application version, and reports whether that string names a real release.
//
// `go install <module>@vX.Y.Z` records the resolved tag as the main module's version, and for
// such a binary that is the ONLY identity it carries: no -ldflags stamp, and an install
// directory with no VERSION marker anywhere above it. Only an exact release-tag shape is
// accepted — "vMAJOR.MINOR.PATCH", the only shape this repo tags — so a non-release build
// cannot invent a version: a plain `go build` from a VCS tree records a pseudo-version such as
// "v0.41.1-0.20260729114657-6ef585379011+dirty", a build without VCS records "(devel)", and a
// binary with no module stamp records "". The leading "v" is trimmed so the answer matches the
// VERSION file and the published archives ("0.42.0", not "v0.42.0").
func releaseFromModuleVersion(moduleVersion string) (string, bool) {
	v := strings.TrimSpace(moduleVersion)
	if !strings.HasPrefix(v, "v") {
		return "", false
	}
	v = strings.TrimPrefix(v, "v")
	parts := strings.Split(v, ".")
	if len(parts) != 3 {
		return "", false
	}
	for _, p := range parts {
		if p == "" {
			return "", false
		}
		for _, r := range p {
			if r < '0' || r > '9' {
				return "", false
			}
		}
	}
	return v, true
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
		if v, found, valid := readVersionFile(filepath.Join(dir, "VERSION")); found {
			if !valid {
				return "", false
			}
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

func readVersionFile(path string) (value string, found, valid bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false, false
	}
	v := strings.TrimSpace(string(b))
	if v == "" {
		return "", false, false
	}
	if hasConflictMarker(v) {
		return "", true, false
	}
	return v, true, true
}

func hasConflictMarker(value string) bool {
	for _, line := range strings.Split(value, "\n") {
		line = strings.TrimSuffix(line, "\r")
		if strings.HasPrefix(line, strings.Repeat("<", 7)) ||
			strings.HasPrefix(line, strings.Repeat("=", 7)) ||
			strings.HasPrefix(line, strings.Repeat(">", 7)) {
			return true
		}
	}
	return false
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

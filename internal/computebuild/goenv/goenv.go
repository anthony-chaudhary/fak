// Package goenv holds the Go build-environment rules every fak-spawned go
// command shares so the build cache stays bounded.
//
// Without -trimpath cmd/go hashes each main-module package's absolute directory
// into its action ID, so every fresh checkout (a witness snapshot, a worker
// worktree) recompiles the whole graph into new cache entries that no other
// checkout can reuse.
package goenv

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	// NoTrimpathEnv set to "1" opts out of the appended -trimpath, e.g. to debug
	// with absolute source paths in binaries.
	NoTrimpathEnv = "FAK_NO_TRIMPATH"
	// SharedCacheEnv names an operator-owned absolute Go build cache.
	SharedCacheEnv = "FAK_SHARED_GOCACHE"
)

// TrimpathFlags returns inherited GOFLAGS with -trimpath appended unless a
// -trimpath flag in any form (including an explicit -trimpath=false) is present.
func TrimpathFlags(inherited string) string {
	fields := strings.Fields(inherited)
	for _, f := range fields {
		name := strings.TrimPrefix(strings.TrimPrefix(f, "-"), "-")
		if name == "trimpath" || strings.HasPrefix(name, "trimpath=") {
			return strings.Join(fields, " ")
		}
	}
	return strings.Join(append(fields, "-trimpath"), " ")
}

// GoFlags applies TrimpathFlags to base unless lookup reports NoTrimpathEnv=1,
// in which case base is returned unchanged.
func GoFlags(base string, lookup func(string) (string, bool)) string {
	if v, _ := lookup(NoTrimpathEnv); strings.TrimSpace(v) == "1" {
		return base
	}
	return TrimpathFlags(base)
}

// SharedCache returns the absolute directory lookup reports for
// SharedCacheEnv, or false when it is unset, relative, or malformed.
func SharedCache(lookup func(string) (string, bool)) (string, bool) {
	v, _ := lookup(SharedCacheEnv)
	v = strings.TrimSpace(v)
	if !filepath.IsAbs(v) || strings.ContainsRune(v, '\x00') {
		return "", false
	}
	return v, true
}

// EnvironLookup returns a lookup over a KEY=VALUE environment slice, matching
// keys case-insensitively on Windows like the OS does. The last entry wins.
func EnvironLookup(env []string) func(string) (string, bool) {
	return func(key string) (string, bool) {
		val, found := "", false
		for _, kv := range env {
			name, v, ok := strings.Cut(kv, "=")
			if ok && sameEnvKey(name, key) {
				val, found = v, true
			}
		}
		return val, found
	}
}

// ProcessLookup is os.LookupEnv; it exists so callers can pass the process
// environment where a lookup is expected.
func ProcessLookup(key string) (string, bool) { return os.LookupEnv(key) }

func sameEnvKey(a, b string) bool {
	if os.PathSeparator == '\\' {
		return strings.EqualFold(a, b)
	}
	return a == b
}

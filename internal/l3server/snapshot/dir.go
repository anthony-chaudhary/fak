package snapshot

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// ConfineDir resolves a client-supplied snapshot/restore directory against the
// server-configured base directory (issue #14345) and returns the cleaned
// directory to use. The request directory must stay INSIDE the base: it may be
// the base itself or any subdirectory of it, never a sibling, a parent, or an
// unrelated path. Three request shapes are accepted:
//
//   - "" or "." — the configured base directory itself,
//   - a relative subpath ("sub", "sub/deeper") — resolved against the BASE,
//     deliberately never against the server's working directory (CWD-relative
//     resolution is the client-chosen-path hole this closes), and
//   - an absolute path — accepted only when it resolves inside the base (the
//     normal client shape: it echoes back a directory a previous snapshot
//     response named).
//
// It returns an error for:
//   - an empty base (fail closed — there is nothing to be contained by),
//   - a request dir that escapes the base ("..", or ".."+separator prefixes),
//   - an absolute request dir outside the base, and
//   - an absolute request dir on another volume/root than the base (Windows
//     drive or UNC vs. local), where filepath.Rel cannot produce a contained
//     form.
//
// The check is lexical: it constrains the PATH, not the filesystem. A symlink
// inside the base that points outside it is not detected here; the server never
// creates symlinks itself, and operators who need that guarantee should point
// snapshot_dir at a directory they control. On Windows filepath.Rel compares
// case-insensitively, matching how the filesystem itself resolves paths there.
func ConfineDir(base, req string) (string, error) {
	if strings.TrimSpace(base) == "" {
		return "", errors.New("request dir refused: base snapshot directory is empty")
	}
	cleanBase := filepath.Clean(base)
	cleanReq := filepath.Clean(req)
	if cleanReq == "." {
		// "The configured directory itself." Handled before Rel because
		// Rel refuses mixed absolute/relative forms outright.
		return cleanBase, nil
	}
	if filepath.IsAbs(cleanReq) {
		rel, err := filepath.Rel(cleanBase, cleanReq)
		if err != nil {
			return "", fmt.Errorf("request dir %q refused: cannot confine under %q: %v", req, cleanBase, err)
		}
		if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return "", fmt.Errorf("request dir %q refused: outside snapshot directory %q", req, cleanBase)
		}
		if rel == "." {
			return cleanBase, nil
		}
		return filepath.Join(cleanBase, rel), nil
	}
	// Relative request: a subpath of the base. Clean has already collapsed
	// interior dot segments, so any surviving leading ".." is an escape.
	if cleanReq == ".." || strings.HasPrefix(cleanReq, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("request dir %q refused: outside snapshot directory %q", req, cleanBase)
	}
	return filepath.Join(cleanBase, cleanReq), nil
}

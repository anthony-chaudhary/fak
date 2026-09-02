// Package walkfiles is the shared swallow-and-scan filepath.WalkDir scaffold:
// walk a tree, tolerate walk-step errors, visit only regular files, and
// propagate visit errors to the caller.
//
// Discovery/refresh scans (latest-session discovery, source concatenation,
// markdown/lock inventories, ...) all re-wrote the same closure: swallow the
// walk-step error (a scan that misses one entry is still useful), skip
// directories, filter/collect in the body, and decide afterwards what the
// results are worth. That closure was the tree's largest copy-pasted clone
// family; this primitive owns it once. Because every cloning site swallowed
// the walk error, the scaffold surfaces ONLY visit errors — a missing root is
// a scan that found nothing, not a failure. Scans whose dir behavior differs
// structurally — e.g. subtree skipping via filepath.SkipDir — keep their own
// WalkDir closure; forcing them through a files-only walker would change what
// they collect.
package walkfiles

import (
	"io/fs"
	"path/filepath"
)

// Files walks root with filepath.WalkDir and calls visit for every regular
// file whose walk step carried no error. Walk-step errors — an unreadable
// directory, an entry that vanishes mid-walk, a missing root — are swallowed
// so the scan keeps going (and yields whatever it could see); the error
// returned is visit's alone. Directories are never visited — callers filter
// inside visit by name, suffix, or content. An error returned by visit aborts
// the walk and is returned unchanged.
func Files(root string, visit func(path string, d fs.DirEntry) error) error {
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		return visit(p, d)
	})
}

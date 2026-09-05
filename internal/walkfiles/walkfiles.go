// Package walkfiles traverses directory trees, visiting regular files and ignoring walk-step errors.
package walkfiles

import (
	"io/fs"
	"path/filepath"
)

// Files walks root and calls visit for each regular file, swallowing
// walk-step errors and aborting on visit failure.
func Files(root string, visit func(path string, d fs.DirEntry) error) error {
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		return visit(p, d)
	})
}

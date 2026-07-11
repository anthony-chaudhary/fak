package wiki

import (
	"os"
	"path/filepath"
	"strconv"
)

// repoName is the base name of the repo root, used as the wiki's Repo label. An
// empty or "." root degrades to "repo" so the Tree always carries a stable label.
func repoName(root string) string {
	if root == "" || root == "." {
		return "repo"
	}
	base := filepath.Base(filepath.Clean(root))
	if base == "." || base == string(filepath.Separator) || base == "" {
		return "repo"
	}
	return base
}

// fileExists reports whether rel resolves to a regular file (not a directory)
// under root. It is the same non-dangling test the citation resolver applies,
// reused so a structure page never cites a path that isn't there.
func fileExists(root, rel string) bool {
	info, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil && !info.IsDir()
}

// itoa is strconv.Itoa under a local name so structure.go reads without an extra
// import line in a file that otherwise needs none.
func itoa(n int) string { return strconv.Itoa(n) }

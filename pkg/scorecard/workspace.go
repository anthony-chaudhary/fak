package scorecard

import "path/filepath"

// WorkspaceRoot resolves the normalized root a scorecard records and scans.
// filepath.Abs can return a path together with an error; callers historically
// retained that path and fell back only when it was empty, so this helper does
// the same.
func WorkspaceRoot(root string) string {
	abs, _ := filepath.Abs(root)
	if abs == "" {
		return root
	}
	return abs
}

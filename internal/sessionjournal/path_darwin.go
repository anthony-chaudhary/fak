//go:build darwin

package sessionjournal

import (
	"os"
	"path/filepath"
)

func hostJournalPath() string {
	// Guard normally runs in an interactive user's session on macOS. Keep its
	// journal under that user's standard Application Support root so creating
	// the first registration never requires root access.
	root, err := os.UserConfigDir()
	if err != nil || root == "" {
		// A session without a resolvable home is not an ordinary interactive
		// user. Preserve the historical machine-store fallback so service
		// installations retain an explicit, stable path and Append reports any
		// missing privilege instead of silently writing relative to the CWD.
		return filepath.Join("/var", "lib", "fak", "session-journal", "events.jsonl")
	}
	return filepath.Join(root, "fak", "session-journal", "events.jsonl")
}

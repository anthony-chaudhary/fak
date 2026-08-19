//go:build !windows

package sessionjournal

import "path/filepath"

func hostJournalPath() string {
	return filepath.Join("/var", "lib", "fak", "session-journal", "events.jsonl")
}

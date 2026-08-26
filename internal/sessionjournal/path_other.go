//go:build !windows && !darwin

package sessionjournal

import "path/filepath"

func hostJournalPath() string {
	// Non-Darwin Unix installations retain the machine-wide journal used by
	// service-managed agents and shared host-account discovery.
	return filepath.Join("/var", "lib", "fak", "session-journal", "events.jsonl")
}

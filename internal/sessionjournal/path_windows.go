//go:build windows

package sessionjournal

import (
	"os"
	"path/filepath"
)

func hostJournalPath() string {
	// Windows retains the machine-wide ProgramData journal shared by services
	// and interactive accounts on the host.
	root := os.Getenv("ProgramData")
	if root == "" {
		root = filepath.Join(os.Getenv("SystemDrive")+string(os.PathSeparator), "ProgramData")
	}
	return filepath.Join(root, "fak", "session-journal", "events.jsonl")
}

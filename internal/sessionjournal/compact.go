package sessionjournal

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// Compact rewrites the journal to its valid rows under the same cross-process
// lock used by Append. The old file is closed before replacement, allowing
// Windows delete/rename semantics without racing a heartbeat writer.
func Compact(path string) (int, error) {
	if path == "" {
		path = DefaultPath()
	}
	kept := 0
	err := withJournalLock(path, func() error {
		b, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		events := ParseEvents(string(b))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		tmp, err := os.CreateTemp(filepath.Dir(path), ".session-journal-compact-*")
		if err != nil {
			return err
		}
		name := tmp.Name()
		defer os.Remove(name)
		enc := json.NewEncoder(tmp)
		for _, event := range events {
			if err = enc.Encode(event); err != nil {
				break
			}
			kept++
		}
		if err == nil {
			err = tmp.Sync()
		}
		if closeErr := tmp.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			return err
		}
		// Windows cannot atomically replace an existing destination with Rename;
		// the journal lock keeps the remove/rename interval invisible to writers.
		if err = os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return os.Rename(name, path)
	})
	return kept, err
}
